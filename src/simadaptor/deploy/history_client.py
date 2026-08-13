"""
Client-side helper to talk to the history controller over ZMQ.

Responsibilities:
- Subscribe to the controller's history stream (PUB -> SUB).
- Maintain a rolling observation buffer for downstream tasks (RL policy,
  trajectory follower, etc.).
- Push control commands / gains / embeddings back to the controller
  (PUSH -> PULL on the controller side).

Message shapes match ``examples/history_controller.py``:
  History publish: {"type": "history", "window": [{t, q, qd, tau_applied, tau_base,
                                                  tau_adaptor_delta, tau_measured,
                                                  valid_for_history, synthetic_padding,
                                                  publish_ready, sample_dt_sec}, ...]}
  Reset publish (from controller): {"type": "reset", "reason": "..."}
  Command publish (from client): {"target_q": [...], "target_dq": [...],
                                  "stiffness": [...], "damping": [...],
                                  "filter": float, "embedding": [...],
                                  "feedforward": [...], "command_source": str,
                                  "command_id": int}
  Reliable (REQ/REP) commands (from client): {"cmd": "...", ...}
    - enable_adaptor: {"cmd": "enable_adaptor", "enabled": bool}
    - load_bin_blob: {"cmd": "load_bin_blob", "name": str, "blob": base64 str}
    - set_embedding: {"cmd": "set_embedding", "embedding": [...]}
    - set_ideal_model_has_gravity: {"cmd": "set_ideal_model_has_gravity", "enabled": bool}
"""

from collections import deque
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
import uuid

import numpy as np
import zmq
from simadaptor.deploy.fast_policy_transport import (
    FAST_ACTION_ENDPOINT,
    FAST_STATE_ENDPOINT,
    FastPolicyTransport,
)

# Default endpoints must align with examples/history_controller.py
HISTORY_ENDPOINT = "tcp://192.168.1.101:5555"  # controller binds PUB, client SUB connects
COMMAND_ENDPOINT = "tcp://192.168.1.101:5556"  # controller binds PULL, client PUSH connects
REQUEST_ENDPOINT = "tcp://192.168.1.101:5557"  # client REQ connects to controller REP


def _load_viser_util():
    raise RuntimeError(
        "Viser visualization support is not included in the minimal TAM release."
    )


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonify_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_value(v) for v in value]
    return value


class _HistoryLogger:
    """Lightweight logger to accumulate history samples keyed by field."""

    def __init__(self, fields: Optional[List[str]] = None, dedup: bool = True):
        # Always track timestamps even if not explicitly requested.
        default_fields = [
            "t",
            "t_raw",
            "q",
            "dq",
            "tau_cmd",
            "tau_applied",
            "tau_base",
            "tau_commanded",
            "tau_measured",
            "gravity",
            "coriolis",
            "tau_adaptor_delta",
            "tau_tam_residual",
            "history_embedding_seq",
            "adaptor_active",
            "valid_for_history",
            "synthetic_padding",
            "publish_ready",
            "sample_dt_sec",
        ]
        self.fields = fields if fields is not None else default_fields
        if "t" not in self.fields:
            self.fields.append("t")
        if "t_raw" not in self.fields:
            self.fields.append("t_raw")
        self.dedup = dedup
        self.reset()

    def reset(self) -> None:
        self._log: Dict[str, List[Any]] = {k: [] for k in self.fields}
        self._last_ts: float = -float("inf")
        self._t0: Optional[float] = None

    def append(self, sample: Dict[str, Any]) -> bool:
        ts = float(sample.get("t", -float("inf")))
        if self.dedup and ts <= self._last_ts:
            return False
        if self._t0 is None:
            self._t0 = ts

        if "t_raw" in self._log:
            self._log["t_raw"].append(ts)
        if "t" in self._log:
            self._log["t"].append(ts - self._t0)

        for key in self._log:
            if key in ("t", "t_raw"):
                continue
            if key in sample and sample[key] is not None:
                self._log[key].append(sample[key])
        self._last_ts = ts
        return True

    def as_lists(self) -> Dict[str, List[Any]]:
        return {k: list(v) for k, v in self._log.items()}

    def as_numpy(self) -> Dict[str, Any]:
        return {k: (np.asarray(v) if len(v) else np.empty((0,))) for k, v in self._log.items()}


class HistoryControllerClient:
    def __init__(
        self,
        history_endpoint: str = HISTORY_ENDPOINT,
        command_endpoint: str = COMMAND_ENDPOINT,
        history_buffer: int = 500,
        request_endpoint: Optional[str] = None,
        conflate: bool = False,
        fast_transport_enabled: bool = False,
        fast_state_endpoint: str = FAST_STATE_ENDPOINT,
        fast_action_endpoint: str = FAST_ACTION_ENDPOINT,
        fast_state_max_age_s: float = 0.05,
        fast_action_requires_state: bool = False,
        fast_poll_timeout_ms: int = 0,
    ):
        self.ctx = zmq.Context.instance()
        self._history_endpoint = str(history_endpoint)
        self._command_endpoint = str(command_endpoint)
        self._history_conflate = bool(conflate)

        # Subscribe to history/notification stream from controller.
        self.sub = self._make_history_socket()

        # Push commands to the controller (controller PULL binds the endpoint).
        self.push = self._make_command_socket()

        # Optional REQ socket for reliable commands (load bin, set embedding, enable adaptor).
        self.req = None
        self._request_endpoint = request_endpoint
        self._request_rcvtimeo_ms = 5000
        self._request_sndtimeo_ms = 2000
        if request_endpoint is not None:
            self.req = self._make_req_socket(request_endpoint)

        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_buffer)
        self._last_ts: float = -np.inf
        self._last_reset: Optional[Dict[str, Any]] = None
        self._viewer: Optional[Any] = None
        self._logger: Optional[_HistoryLogger] = None
        self._command_source = uuid.uuid4().hex
        self._next_command_id = 0
        self.fast_transport: Optional[FastPolicyTransport] = None
        self._fast_state_max_age_s = float(fast_state_max_age_s)
        self._fast_action_requires_state = bool(fast_action_requires_state)
        self._fast_poll_timeout_ms = int(max(0, fast_poll_timeout_ms))
        if bool(fast_transport_enabled):
            self.fast_transport = FastPolicyTransport(
                state_endpoint=str(fast_state_endpoint),
                action_endpoint=str(fast_action_endpoint),
                conflate=True,
            )

    def close(self):
        self.sub.close(linger=0)
        self.push.close(linger=0)
        if self.req is not None:
            self.req.close(linger=0)
        if self.fast_transport is not None:
            self.fast_transport.close()
            self.fast_transport = None

    def _make_history_socket(self) -> zmq.Socket:
        sub = self.ctx.socket(zmq.SUB)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        if self._history_conflate:
            sub.setsockopt(zmq.RCVHWM, 1)
            sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt(zmq.LINGER, 0)
        sub.connect(self._history_endpoint)
        return sub

    def _make_command_socket(self) -> zmq.Socket:
        push = self.ctx.socket(zmq.PUSH)
        push.connect(self._command_endpoint)
        return push

    def reset_transport_sockets(self, *, reason: Optional[str] = None) -> None:
        for sock in (self.sub, self.push):
            try:
                sock.close(linger=0)
            except Exception:
                pass
        self.sub = self._make_history_socket()
        self.push = self._make_command_socket()
        self.reset_request_socket()
        self.reset_history()
        self._last_reset = None
        if reason:
            print(f"[history_client] Recreated history/command sockets: {reason}")

    def reset_history(self):
        """Clear cached history and last timestamp (useful before a new session)."""
        self._history.clear()
        self._last_ts = -np.inf

    # --- Logging helpers ---
    def start_logging(self, fields: Optional[List[str]] = None, dedup: bool = True) -> None:
        """Enable logging of samples seen in poll_history/log_sample."""
        self._logger = _HistoryLogger(fields=fields, dedup=dedup)

    def stop_logging(self, to_numpy: bool = True) -> Optional[Dict[str, Any]]:
        """Return accumulated log and disable logging."""
        if self._logger is None:
            return None
        log = self._logger.as_numpy() if to_numpy else self._logger.as_lists()
        self._logger = None
        return log

    def get_log(self, to_numpy: bool = True) -> Optional[Dict[str, Any]]:
        """Return current log without stopping logging."""
        if self._logger is None:
            return None
        return self._logger.as_numpy() if to_numpy else self._logger.as_lists()

    def log_sample(self, sample: Dict[str, Any]) -> None:
        """Manually append a sample to the logger."""
        if self._logger is not None:
            self._logger.append(sample)

    def _require_req(self) -> zmq.Socket:
        if self.req is None:
            raise RuntimeError("Request endpoint is not configured for this HistoryControllerClient.")
        return self.req

    def _make_req_socket(self, endpoint: str) -> zmq.Socket:
        req = self.ctx.socket(zmq.REQ)
        req.RCVTIMEO = int(self._request_rcvtimeo_ms)
        req.SNDTIMEO = int(self._request_sndtimeo_ms)
        req.LINGER = 0
        req.connect(endpoint)
        return req

    def reset_request_socket(self, *, reason: Optional[str] = None) -> None:
        if self._request_endpoint is None:
            return
        if self.req is not None:
            try:
                self.req.close(linger=0)
            except Exception:
                pass
        self.req = self._make_req_socket(self._request_endpoint)
        if reason:
            print(f"[history_client] Reset request socket after failure: {reason}")

    def _request_json(self, payload: Dict[str, Any], *, cmd_name: str) -> Dict[str, Any]:
        req = self._require_req()
        try:
            req.send_json(payload)
            resp = req.recv_json()
        except Exception as exc:
            self.reset_request_socket(reason=f"{cmd_name}: {exc}")
            raise RuntimeError(f"{cmd_name} request failed: {exc}") from exc
        if not isinstance(resp, dict):
            raise RuntimeError(f"{cmd_name} returned a non-dict response: {type(resp).__name__}")
        return resp

    def _next_async_metadata(self) -> Dict[str, Any]:
        self._next_command_id += 1
        return {
            "command_source": self._command_source,
            "command_id": self._next_command_id,
        }

    def send_async_payload(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("Async payload must be a dict.")
        payload_json = _jsonify_value(payload)
        if not payload_json:
            return
        payload_json.update(self._next_async_metadata())
        self.push.send_json(payload_json)

    def poll_fast_state(
        self,
        *,
        max_age_s: Optional[float] = None,
        poll_timeout_ms: Optional[int] = None,
    ):
        """Poll the optional fast latest-state socket and return a fresh sample."""
        if self.fast_transport is None:
            return None
        return self.fast_transport.get_recent_state(
            max_age_s=float(self._fast_state_max_age_s if max_age_s is None else max_age_s),
            poll_timeout_ms=int(self._fast_poll_timeout_ms if poll_timeout_ms is None else max(0, poll_timeout_ms)),
        )

    def wait_for_fast_state(
        self,
        *,
        timeout_s: float = 2.0,
        max_age_s: Optional[float] = None,
        poll_timeout_ms: int = 20,
    ):
        """Wait for the fast bridge to publish at least one fresh state sample."""
        if self.fast_transport is None:
            raise RuntimeError("Fast transport is not enabled for this HistoryControllerClient.")
        deadline = time.perf_counter() + float(max(0.0, timeout_s))
        while time.perf_counter() <= deadline:
            state = self.poll_fast_state(
                max_age_s=max_age_s,
                poll_timeout_ms=int(max(0, poll_timeout_ms)),
            )
            if state is not None:
                return state
        raise RuntimeError(
            "No fresh fast state received before startup timeout "
            f"({float(timeout_s):.3f}s) from {self.fast_transport.state_endpoint}."
        )

    def _observed_fast_state_seq_for_action(self) -> int:
        state = self.poll_fast_state()
        if state is None and self._fast_action_requires_state:
            state = self.poll_fast_state(
                poll_timeout_ms=max(int(self._fast_poll_timeout_ms), 10),
            )
        if state is None:
            if self._fast_action_requires_state:
                endpoint = "<disabled>" if self.fast_transport is None else self.fast_transport.state_endpoint
                raise RuntimeError(
                    "No fresh fast state available for action send from "
                    f"{endpoint} after waiting {max(int(self._fast_poll_timeout_ms), 10)} ms."
                )
            return 0
        return int(state.seq)

    def _poll_messages(
        self,
        timeout_ms: int = 0,
        *,
        latest_only: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)
        socks = dict(poller.poll(timeout=timeout_ms))
        if self.sub not in socks:
            return None

        msgs: List[Dict[str, Any]] = []
        last_history = None
        last_history_idx = -1
        last_reset = None
        last_reset_idx = -1
        idx = 0
        while True:
            try:
                msg = self.sub.recv_json(flags=zmq.NOBLOCK)
            except Exception:
                break
            if isinstance(msg, dict):
                if latest_only:
                    if msg.get("type") == "history":
                        last_history = msg
                        last_history_idx = idx
                    elif msg.get("type") == "reset":
                        last_reset = msg
                        last_reset_idx = idx
                else:
                    msgs.append(msg)
            idx += 1

        if not msgs and latest_only:
            if last_history is not None:
                msgs.append((last_history_idx, last_history))
            if last_reset is not None:
                msgs.append((last_reset_idx, last_reset))
            msgs.sort(key=lambda x: x[0])
            msgs = [m for _, m in msgs]

        return msgs if msgs else None

    def _consume_messages(
        self,
        msgs: Sequence[Dict[str, Any]],
        *,
        log: bool = False,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
        last_window: Optional[List[Dict[str, Any]]] = None
        last_raw_window: Optional[List[Dict[str, Any]]] = None
        for msg in msgs:
            if msg.get("type") == "history":
                window = msg.get("window", [])
                if not isinstance(window, list):
                    window = []
                last_raw_window = window
                for sample in window:
                    if not isinstance(sample, dict):
                        continue
                    ts = float(sample.get("t", -np.inf))
                    if ts > self._last_ts:
                        self._history.append(sample)
                        self._last_ts = ts
                        if log and self._logger is not None:
                            self._logger.append(sample)
                last_window = list(self._history)[-len(window):] if window else []
            elif msg.get("type") == "reset":
                self._last_reset = {
                    "reason": msg.get("reason", "unknown"),
                    "f_norm": msg.get("f_norm", None),
                    "timestamp": msg.get("timestamp", None),
                }
                if last_window is None:
                    last_window = []
        return last_window, last_raw_window

    # --- Receiving history ---
    def poll_history(
        self,
        timeout_ms: int = 0,
        log: bool = False,
        *,
        return_raw: bool = False,
        latest_only: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Poll once; returns the latest window if received, else None.

        By default, this returns a deduplicated window built from the internal rolling buffer
        (dedup keyed by sample["t"]). If your controller publishes windows where `t` resets
        frequently (e.g., window-local time), prefer ``poll_window`` or use
        ``return_raw=True`` to return the raw window from the most recent history message
        without applying the monotonic-`t` filter.
        """
        msgs = self._poll_messages(timeout_ms=timeout_ms, latest_only=latest_only)
        if msgs is None:
            return None
        last_window, last_raw_window = self._consume_messages(msgs, log=log)
        if return_raw and last_raw_window is not None:
            return last_raw_window
        return last_window if last_window is not None else None

    def poll_window(
        self,
        timeout_ms: int = 0,
        log: bool = False,
        *,
        latest_only: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        """Poll once and return the raw controller-published history window."""
        msgs = self._poll_messages(timeout_ms=timeout_ms, latest_only=latest_only)
        if msgs is None:
            return None
        _, last_raw_window = self._consume_messages(msgs, log=log)
        return last_raw_window

    def get_stack(self, n: int = 1) -> List[Dict[str, Any]]:
        """Return the most recent n samples from the rolling buffer."""
        n = min(n, len(self._history))
        if n == 0:
            return []
        return list(self._history)[-n:]

    def pop_last_reset(self) -> Optional[Dict[str, Any]]:
        """Return and clear the last reset event sent by the server, if any."""
        evt = self._last_reset
        self._last_reset = None
        return evt

    # --- Sending commands ---
    def send_command(
        self,
        target_q: Optional[np.ndarray] = None,
        target_dq: Optional[np.ndarray] = None,
        stiffness: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        filter_coeff: Optional[float] = None,
        embedding: Optional[np.ndarray] = None,
        feedforward: Optional[np.ndarray] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        prefer_fast_transport: bool = True,
    ) -> None:
        payload: Dict[str, Any] = {}
        has_fast_action_fields = any(
            value is not None
            for value in (target_q, target_dq, stiffness, damping, feedforward)
        )
        if bool(prefer_fast_transport) and self.fast_transport is not None and has_fast_action_fields:
            self.fast_transport.send_action(
                observed_state_seq=self._observed_fast_state_seq_for_action(),
                target_q=target_q,
                target_dq=target_dq,
                stiffness=stiffness,
                damping=damping,
                feedforward=feedforward,
            )
        else:
            if target_q is not None:
                payload["target_q"] = np.asarray(target_q, dtype=float).tolist()
            if target_dq is not None:
                payload["target_dq"] = np.asarray(target_dq, dtype=float).tolist()
            if stiffness is not None:
                payload["stiffness"] = np.asarray(stiffness, dtype=float).tolist()
            if damping is not None:
                payload["damping"] = np.asarray(damping, dtype=float).tolist()
            if feedforward is not None:
                payload["feedforward"] = np.asarray(feedforward, dtype=float).tolist()
        if filter_coeff is not None:
            payload["filter"] = float(filter_coeff)
        if embedding is not None:
            payload["embedding"] = np.asarray(embedding, dtype=float).tolist()
        if extra_fields is not None:
            payload.update(_jsonify_value(extra_fields))

        if payload:
            self.send_async_payload(payload)

    def enable_adaptor(self, input_bool: bool) -> None:
        """Send a command to enable the adaptor on the controller side."""
        self.enable_adaptor_best_effort(input_bool)
        print("Sent enable_adaptor command to controller.")

    def send_reset_command(self) -> None:
        """Send a reset command to the controller."""
        self.reset_best_effort()
        print("Sent reset command to controller.")

    def go_to_home(
        self,
        home_q: Sequence[float],
        *,
        duration_s: float = 3.0,
        dt: float = 0.01,
        stiffness: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        warmup_poll_s: float = 2.0,
        poll_timeout_ms: int = 5,
        settle_s: float = 0.0,
        require_measurement: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearly interpolate from the current measured q to home_q and stream commands.

        Returns:
            (q_start, q_goal) used for the interpolation.
        """
        q_goal = np.asarray(home_q, dtype=float).reshape(-1)
        if q_goal.size == 0:
            raise ValueError("home_q must be non-empty.")

        def _extract_q(window: Optional[List[Dict[str, Any]]]) -> Optional[np.ndarray]:
            if not window:
                return None
            for s in reversed(window):
                if not isinstance(s, dict):
                    continue
                q0 = s.get("q", None)
                if q0 is None:
                    continue
                q_arr = np.asarray(q0, dtype=float).reshape(-1)
                if q_arr.size >= q_goal.size:
                    return q_arr[: q_goal.size]
            return None

        # Always prefer the raw-most-recent controller window (controller timestamps can reset,
        # which makes the deduped rolling buffer stale).
        q_start: Optional[np.ndarray] = None
        for _ in range(3):
            q_start = _extract_q(self.poll_window(timeout_ms=0))
            if q_start is not None:
                break

        if q_start is None:
            deadline = time.perf_counter() + float(max(0.0, warmup_poll_s))
            while time.perf_counter() < deadline and q_start is None:
                q_start = _extract_q(self.poll_window(timeout_ms=int(poll_timeout_ms)))

        if q_start is None:
            if require_measurement:
                raise RuntimeError("Failed to read current joint state from history stream; refusing to go_to_home.")
            q_start = q_goal.copy()

        if stiffness is not None or damping is not None:
            self.send_command(
                stiffness=stiffness,
                damping=damping,
                prefer_fast_transport=False,
            )
            time.sleep(0.05)

        duration_s = float(duration_s)
        dt = float(dt)
        if duration_s <= 0.0:
            self.send_command(target_q=q_goal, target_dq=np.zeros_like(q_goal))
            return q_start, q_goal
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}.")

        n_steps = int(np.ceil(duration_s / dt)) + 1
        n_steps = max(2, n_steps)
        t0 = time.perf_counter()
        for i in range(n_steps):
            alpha = float(i) / float(n_steps - 1)
            q_cmd = (1.0 - alpha) * q_start + alpha * q_goal
            dq_cmd = np.zeros_like(q_cmd)
            self.send_command(target_q=q_cmd, target_dq=dq_cmd)
            # Keep receiving side alive while moving.
            self.poll_history(timeout_ms=0)

            target_wall = t0 + i * dt
            sleep_dt = target_wall - time.perf_counter()
            while sleep_dt > 0.0:
                time.sleep(min(0.0015, sleep_dt))
                sleep_dt = target_wall - time.perf_counter()

        if settle_s > 0.0:
            settle_deadline = time.perf_counter() + float(settle_s)
            while time.perf_counter() < settle_deadline:
                self.send_command(target_q=q_goal, target_dq=np.zeros_like(q_goal))
                self.poll_history(timeout_ms=0)
                time.sleep(min(0.01, max(0.0, settle_deadline - time.perf_counter())))

        return q_start, q_goal

    def send_simadaptor_bin(self, bin_path: str) -> None:
        """Notify controller to load a new SimAdaptor .bin file."""
        self.load_bin_path_best_effort(bin_path)
        print(f"Sent simadaptor bin path to controller: {bin_path}")

    def send_simadaptor_bin_blob(self, bin_name: str, bin_b64: str) -> None:
        """Send adaptor weights as a base64 blob for the controller to load."""
        self.load_bin_blob_best_effort(bin_name, bin_b64)
        print(f"Sent simadaptor bin blob ({bin_name}, {len(bin_b64)} bytes b64) to controller")

    def set_embedding_best_effort(self, embedding: np.ndarray) -> None:
        self.send_command(embedding=np.asarray(embedding, dtype=float).reshape(-1))

    def enable_adaptor_best_effort(self, enabled: bool) -> None:
        self.send_async_payload({"enable_adaptor": bool(enabled)})

    def load_bin_blob_best_effort(
        self,
        bin_name: str,
        bin_b64: str,
        *,
        enable_after_load: Optional[bool] = None,
    ) -> None:
        payload = {"simadaptor_bin_name": str(bin_name), "simadaptor_bin_blob": str(bin_b64)}
        if enable_after_load is not None:
            payload["simadaptor_enable_after_load"] = bool(enable_after_load)
        self.send_async_payload(payload)

    def load_bin_path_best_effort(
        self,
        bin_path: str,
        *,
        enable_after_load: Optional[bool] = None,
    ) -> None:
        payload = {"load_simadaptor_bin": str(bin_path)}
        if enable_after_load is not None:
            payload["simadaptor_enable_after_load"] = bool(enable_after_load)
        self.send_async_payload(payload)

    def reset_best_effort(self) -> None:
        self.send_async_payload({"reset": True})

    # Reliable (REQ/REP) helpers.
    def request_load_bin_blob_response(
        self,
        bin_name: str,
        bin_b64: str,
        *,
        enable_after_load: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload = {"cmd": "load_bin_blob", "name": bin_name, "blob": bin_b64}
        if enable_after_load is not None:
            payload["enable_after_load"] = bool(enable_after_load)
        resp = self._request_json(
            payload,
            cmd_name="load_bin_blob",
        )
        ok = resp.get("ok", False)
        if not ok:
            print(f"load_bin_blob failed: {resp}")
        return resp

    def request_load_bin_blob(
        self,
        bin_name: str,
        bin_b64: str,
        *,
        enable_after_load: Optional[bool] = None,
    ) -> bool:
        resp = self.request_load_bin_blob_response(
            bin_name,
            bin_b64,
            enable_after_load=enable_after_load,
        )
        ok = resp.get("ok", False)
        return ok

    def request_set_embedding(self, embedding: np.ndarray) -> bool:
        resp = self._request_json(
            {"cmd": "set_embedding", "embedding": np.asarray(embedding, dtype=float).tolist()},
            cmd_name="set_embedding",
        )
        ok = resp.get("ok", False)
        if not ok:
            print(f"set_embedding failed: {resp}")
        return ok

    def request_enable_adaptor(self, enabled: bool) -> bool:
        resp = self._request_json(
            {"cmd": "enable_adaptor", "enabled": bool(enabled)},
            cmd_name="enable_adaptor",
        )
        ok = resp.get("ok", False)
        if not ok:
            print(f"enable_adaptor failed: {resp}")
        return ok

    def request_set_ideal_model_has_gravity(self, enabled: bool) -> bool:
        resp = self._request_json(
            {"cmd": "set_ideal_model_has_gravity", "enabled": bool(enabled)},
            cmd_name="set_ideal_model_has_gravity",
        )
        ok = resp.get("ok", False)
        if not ok:
            print(f"set_ideal_model_has_gravity failed: {resp}")
        return ok

    def request_controller_status(self) -> Dict[str, Any]:
        """Read controller-side mode flags without changing controller state."""
        return self._request_json({}, cmd_name="controller_status")
    
    # --- Visualization ---
    def enable_viewer(
        self,
        xml_path: str,
        *,
        host: str = "0.0.0.0",
        port: int = 4242,
        prefix: str = "/robot",
        fps: float = 30.0,
        tcp_link: Optional[str] = None,
        force_scale: float = 1.0,
        torque_scale: float = 1.0,
        arrow_radius: float = 0.05,
        invert_vectors: bool = False,
    ):
        """Create a lightweight Viser visualizer to mirror incoming joint states."""
        self._viewer = _load_viser_util().ViserSceneVisualizer(
            xml_path,
            host=host,
            port=port,
            prefix=prefix,
            fps=fps,
            tcp_link=tcp_link,
            force_scale=force_scale,
            torque_scale=torque_scale,
            arrow_radius=arrow_radius,
            invert_vectors=invert_vectors,
        )

    def visualize_sample(self, sample: Dict[str, Any]):
        """Visualize one history sample if a viewer is enabled."""
        if self._viewer is None:
            return
        q = np.asarray(sample.get("q", []), dtype=float)
        if q.size == 0:
            return
        ts = float(sample.get("t", 0.0))
        self._viewer.update_with_timestamp(
            q,
            timestamp=ts,
        )
        ft_world = sample.get("ft_O", None)
        pose_world = sample.get("O_T_EE", None)
        if ft_world is not None and pose_world is not None:
            self._viewer.update_ft(q, ft_world, ft_in_world=True, pose_world=pose_world)

        adaptor_active = sample.get("adaptor_active", False)
        tau_adaptor_delta = sample.get("tau_adaptor_delta", None)
        if adaptor_active and tau_adaptor_delta is not None:
            self._viewer.update_adaptor_torque(q, np.asarray(tau_adaptor_delta, dtype=float))

# Usage patterns:
# - Use HistoryControllerClient directly, or subclass it to encapsulate task-specific logic.
#   e.g., class RLPolicyClient(HistoryControllerClient): def step(): obs = self.get_stack(k); act = policy(obs); self.send_command(...)
#   e.g., class TrajectoryFollower(HistoryControllerClient): def step(): self.send_command(target_q=traj(t), target_dq=traj_dq(t))


def _example_loop():
    """
    Minimal event loop:
    - receive history
    - compute a dummy command
    - send command
    """
    client = HistoryControllerClient(
        history_endpoint="tcp://192.168.1.101:5555",
        command_endpoint="tcp://192.168.1.101:5556",
        history_buffer=500,
        request_endpoint="tcp://192.168.1.101:5557",
    )
    client.enable_viewer(
        "assets/franka_panda/panda_pandagripper.xml",
        prefix="/panda",
        fps=30.0,
        tcp_link="fts300_sensor_body",  # adjust if your TCP link name differs
        force_scale=2.0,
        torque_scale=2.0,
        arrow_radius=0.08,
        invert_vectors=True,
    )
    try:
        while True:
            window = client.poll_history(timeout_ms=5)
            if window:
                latest = window[-1] # (ft_K: ft wrt ee frame, ft_O: ft wrt world frame, O_T_EE: ee pose wrt world frame - 4x4 homogeneous matrix)
                # q = np.asarray(latest.get("q", np.zeros(7)))
                client.visualize_sample(latest)
                # Example: hold current joint positions
                # client.send_command(target_q=q)
                # print(q)
    finally:
        client.close()


if __name__ == "__main__":
    _example_loop()
