from __future__ import annotations

import codecs
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from harness.endpoints import LocalEndpoint, validate_local_openai_endpoint
from harness.reasoning_policy import (
    REASONING_POLICY_CONTRACT,
    ReasoningPolicy,
    ReasoningPolicyError,
    endpoint_api_mode,
    normalize_ollama_request,
    ollama_reasoning_control,
    parse_reasoning_policy,
)


MODEL_TRANSPORT_CONTRACT = "ollama-model-transport-v1"
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_JSON_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_OBSERVATION_BYTES = 64 * 1024 * 1024
ALLOWED_PATHS = {
    "/v1/models",
    "/v1/chat/completions",
    "/api/chat",
    "/api/show",
    "/api/tags",
    "/api/version",
}


class ModelGatewayError(RuntimeError):
    pass


def _response_presence(value: Mapping[str, Any]) -> dict[str, Any]:
    candidates: list[Mapping[str, Any]] = []
    finish: str | None = None
    choices = value.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            if isinstance(choice.get("finish_reason"), str):
                finish = str(choice["finish_reason"])
            for key in ("message", "delta"):
                candidate = choice.get(key)
                if isinstance(candidate, Mapping):
                    candidates.append(candidate)
    message = value.get("message")
    if isinstance(message, Mapping):
        candidates.append(message)
    if isinstance(value.get("done_reason"), str):
        finish = str(value["done_reason"])
    elif value.get("done") is True and finish is None:
        finish = "done"

    visible = reasoning = tools = False
    for candidate in candidates:
        visible = visible or bool(candidate.get("content"))
        reasoning = reasoning or any(
            bool(candidate.get(key))
            for key in ("reasoning", "reasoning_content", "thinking")
        )
        tools = tools or bool(candidate.get("tool_calls"))
    return {
        "visible_content_returned": visible,
        "reasoning_content_returned": reasoning,
        "tool_call_returned": tools,
        "finish_reason": finish,
    }


class _ResponseObserver:
    def __init__(self, *, api_mode: str, streaming: bool) -> None:
        self.api_mode = api_mode
        self.streaming = streaming
        self.visible = False
        self.reasoning = False
        self.tools = False
        self.finish_reasons: set[str] = set()
        self.done_marker = False
        self.parser_errors: list[str] = []
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._buffer = ""
        self._sse_data: list[str] = []
        self._json_body = bytearray()

    def _merge(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            self.parser_errors.append("response event is not a JSON object")
            return
        presence = _response_presence(value)
        self.visible = self.visible or presence["visible_content_returned"]
        self.reasoning = self.reasoning or presence["reasoning_content_returned"]
        self.tools = self.tools or presence["tool_call_returned"]
        finish = presence["finish_reason"]
        if isinstance(finish, str):
            self.finish_reasons.add(finish)
        if value.get("done") is True:
            self.done_marker = True

    def _parse_json_text(self, text: str) -> None:
        if not text:
            return
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            self.parser_errors.append("malformed response JSON event")
            return
        self._merge(value)

    def _finish_sse_event(self) -> None:
        if not self._sse_data:
            return
        data = "\n".join(self._sse_data).strip()
        self._sse_data.clear()
        if data == "[DONE]":
            self.done_marker = True
            return
        self._parse_json_text(data)

    def _line(self, line: str) -> None:
        line = line.rstrip("\r")
        if self.api_mode == "openai-chat-completions":
            if not line:
                self._finish_sse_event()
            elif line.startswith("data:"):
                self._sse_data.append(line[5:].lstrip())
            elif not line.startswith(":"):
                self.parser_errors.append("malformed SSE response line")
        elif line.strip():
            self._parse_json_text(line.strip())

    def feed(self, chunk: bytes) -> None:
        if not self.streaming:
            if len(self._json_body) + len(chunk) > MAX_JSON_RESPONSE_BYTES:
                self.parser_errors.append("JSON response exceeds observation limit")
                return
            self._json_body.extend(chunk)
            return
        try:
            self._buffer += self._decoder.decode(chunk)
        except UnicodeDecodeError:
            self.parser_errors.append("response stream is not valid UTF-8")
            return
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._line(line)
        if len(self._buffer) > MAX_JSON_RESPONSE_BYTES:
            self.parser_errors.append("unterminated response event exceeds limit")
            self._buffer = ""

    def finish(self) -> dict[str, Any]:
        if self.streaming:
            try:
                self._buffer += self._decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                self.parser_errors.append("response stream ended inside UTF-8 data")
            if self._buffer:
                self._line(self._buffer)
                self._buffer = ""
            if self.api_mode == "openai-chat-completions":
                self._finish_sse_event()
        elif self._json_body:
            try:
                self._merge(json.loads(self._json_body.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.parser_errors.append("malformed JSON response body")
        return {
            "visible_content_returned": self.visible,
            "reasoning_content_returned": self.reasoning,
            "tool_call_returned": self.tools,
            "finish_reasons": sorted(self.finish_reasons),
            "finish_state_returned": bool(self.finish_reasons or self.done_marker),
            "stream_done_marker": self.done_marker,
            "parser_errors": list(dict.fromkeys(self.parser_errors)),
        }


class _GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False

    def __init__(self, gateway: "ModelGateway") -> None:
        self.gateway = gateway
        super().__init__(("127.0.0.1", 0), _GatewayHandler)


class _GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "HermesBenchGateway/1"
    sys_version = ""

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        self._proxy()

    def do_CONNECT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        self.send_error(405, "CONNECT is not permitted")

    def _error(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _proxy(self) -> None:
        gateway = self.server.gateway  # type: ignore[attr-defined]
        parsed_path = urlsplit(self.path)
        if parsed_path.query or parsed_path.fragment:
            self._error(400, "query and fragment are not permitted")
            return
        path = parsed_path.path.rstrip("/") or "/"
        if path not in ALLOWED_PATHS:
            self._error(404, "path is outside the model gateway allowlist")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "invalid Content-Length")
            return
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._error(413, "request body exceeds model gateway limit")
            return
        if self.headers.get("Transfer-Encoding"):
            self._error(400, "chunked request bodies are not supported")
            return
        body = self.rfile.read(length) if length else b""
        api_mode = endpoint_api_mode(path)
        request_id: str | None = None
        streaming = False
        if api_mode is not None:
            try:
                payload = json.loads(body.decode("utf-8"))
                normalized, metadata = normalize_ollama_request(
                    api_mode, payload, gateway.policy
                )
                body = json.dumps(
                    normalized,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                ReasoningPolicyError,
                TypeError,
                ValueError,
            ) as exc:
                gateway.record_observer_error(api_mode, type(exc).__name__)
                self._error(400, "inference request could not be normalized")
                return
            streaming = normalized.get("stream") is True
            request_id = gateway.record_request(
                path=path,
                streaming=streaming,
                metadata=metadata,
            )

        headers = {
            "Accept": self.headers.get("Accept", "application/json"),
            "Authorization": "Bearer ollama",
        }
        if body:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        connection = http.client.HTTPConnection(
            gateway.target.host,
            gateway.target.port,
            timeout=gateway.upstream_timeout,
        )
        observer = (
            _ResponseObserver(api_mode=api_mode, streaming=streaming)
            if api_mode is not None
            else None
        )
        status: int | None = None
        client_open = True
        response_recorded = False
        try:
            connection.request(self.command, path, body=body or None, headers=headers)
            response = connection.getresponse()
            status = response.status
            self.send_response(response.status, response.reason)
            response_headers = {key.lower(): value for key, value in response.getheaders()}
            content_type = response_headers.get("content-type")
            if content_type:
                self.send_header("Content-Type", content_type)
            content_length = response_headers.get("content-length")
            if content_length:
                self.send_header("Content-Length", content_length)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                # read1() preserves streaming latency: read(amt) may wait for a
                # much larger buffer even after Ollama has emitted an SSE event.
                chunk = response.read1(65536)
                if not chunk:
                    break
                protocol_complete = False
                if observer is not None:
                    observer.feed(chunk)
                    if observer.done_marker and request_id is not None:
                        gateway.record_response(
                            request_id=request_id,
                            path=path,
                            status=status,
                            observation=observer.finish(),
                            transport_error=None,
                        )
                        response_recorded = True
                        observer = None
                        protocol_complete = True
                if client_open:
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        client_open = False
                if protocol_complete:
                    break
            if (
                observer is not None
                and request_id is not None
                and not response_recorded
            ):
                gateway.record_response(
                    request_id=request_id,
                    path=path,
                    status=status,
                    observation=observer.finish(),
                    transport_error=None,
                )
        except (OSError, http.client.HTTPException) as exc:
            if (
                observer is not None
                and request_id is not None
                and not response_recorded
            ):
                gateway.record_response(
                    request_id=request_id,
                    path=path,
                    status=status,
                    observation=observer.finish(),
                    transport_error=type(exc).__name__,
                )
            if status is None and client_open:
                self._error(502, "model gateway upstream transport failed")
        finally:
            connection.close()
            self.close_connection = True


class ModelGateway:
    def __init__(
        self,
        *,
        target_base_url: str,
        policy: ReasoningPolicy,
        stage: str,
        observations_path: Path,
        upstream_timeout: float,
    ) -> None:
        self.target = validate_local_openai_endpoint(target_base_url)
        self.policy = policy
        self.stage = stage
        self.observations_path = observations_path
        self.upstream_timeout = upstream_timeout
        self._lock = threading.Lock()
        self._request_counter = 0
        self._server: _GatewayServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> LocalEndpoint:
        if self._server is None:
            raise ModelGatewayError("model gateway is not running")
        port = int(self._server.server_address[1])
        return validate_local_openai_endpoint(f"http://127.0.0.1:{port}/v1")

    @property
    def metadata(self) -> dict[str, Any]:
        endpoint = self.endpoint
        return {
            "contract": MODEL_TRANSPORT_CONTRACT,
            "reasoning_policy_contract": REASONING_POLICY_CONTRACT,
            "stage": self.stage,
            "boundary": endpoint.base_url,
            "target": self.target.base_url,
            "target_policy": "fixed configured local endpoint",
            "allowed_paths": sorted(ALLOWED_PATHS),
            "secrets_recorded": False,
            "request_content_recorded": False,
            "response_content_recorded": False,
            "observations_path": str(self.observations_path),
        }

    def _append(self, value: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        with self._lock:
            self.observations_path.parent.mkdir(parents=True, exist_ok=True)
            with self.observations_path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()

    def record_request(
        self,
        *,
        path: str,
        streaming: bool,
        metadata: Mapping[str, Any],
    ) -> str:
        with self._lock:
            self._request_counter += 1
            request_id = f"{self.stage}-{self._request_counter}"
        self._append(
            {
                "event": "request",
                "transport_contract": MODEL_TRANSPORT_CONTRACT,
                "stage": self.stage,
                "request_id": request_id,
                "endpoint_path": path,
                "streaming": streaming,
                **metadata,
            }
        )
        return request_id

    def record_response(
        self,
        *,
        request_id: str,
        path: str,
        status: int | None,
        observation: Mapping[str, Any],
        transport_error: str | None,
    ) -> None:
        self._append(
            {
                "event": "response",
                "transport_contract": MODEL_TRANSPORT_CONTRACT,
                "stage": self.stage,
                "request_id": request_id,
                "endpoint_path": path,
                "status_code": status,
                "transport_error": transport_error,
                **observation,
            }
        )

    def record_observer_error(self, api_mode: str, error: str) -> None:
        self._append(
            {
                "event": "observer_error",
                "transport_contract": MODEL_TRANSPORT_CONTRACT,
                "stage": self.stage,
                "api_mode": api_mode,
                "error": error,
            }
        )

    def start(self) -> "ModelGateway":
        if self._server is not None:
            raise ModelGatewayError("model gateway is already running")
        self._server = _GatewayServer(self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"hermesbench-gateway-{self.stage}",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> "ModelGateway":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def read_model_transport_observations(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    malformed_rows = 0
    read_error: str | None = None
    if path.is_file():
        if path.is_symlink():
            lines = []
            read_error = "SymlinkRejected"
        elif path.stat().st_size > MAX_OBSERVATION_BYTES:
            lines = []
            read_error = "ObservationSizeLimit"
        else:
            try:
                lines = path.read_text(
                    encoding="utf-8", errors="strict"
                ).splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                lines = []
                read_error = type(exc).__name__
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                malformed_rows += 1
    requests = [row for row in rows if row.get("event") == "request"]
    responses = [row for row in rows if row.get("event") == "response"]
    observer_errors = [row for row in rows if row.get("event") == "observer_error"]
    unknown_event_rows = sum(
        row.get("event") not in {"request", "response", "observer_error"}
        for row in rows
    )
    request_ids = [row.get("request_id") for row in requests]
    response_ids = [row.get("request_id") for row in responses]
    duplicate_request_ids = sorted(
        {value for value in request_ids if request_ids.count(value) > 1 and value}
    )
    duplicate_response_ids = sorted(
        {value for value in response_ids if response_ids.count(value) > 1 and value}
    )
    orphan_response_ids = sorted(
        {str(value) for value in response_ids if value not in request_ids}
    )
    missing_response_ids = sorted(
        {str(value) for value in request_ids if value not in response_ids}
    )
    parser_errors = [
        str(error)
        for row in responses
        for error in row.get("parser_errors", [])
        if isinstance(error, str)
    ]
    transport_errors = [
        str(row["transport_error"])
        for row in responses
        if isinstance(row.get("transport_error"), str)
    ]
    policies = sorted(
        {
            str(row["requested_reasoning_policy"])
            for row in requests
            if isinstance(row.get("requested_reasoning_policy"), str)
        }
    )
    invalid_contract_rows = 0
    for row in requests + responses + observer_errors:
        invalid = row.get("transport_contract") != MODEL_TRANSPORT_CONTRACT
        if row.get("event") == "request":
            try:
                policy = parse_reasoning_policy(
                    str(row["requested_reasoning_policy"])
                )
                api_mode = str(row["api_mode"])
                control = ollama_reasoning_control(api_mode, policy)
                field = next(iter(control), None)
            except (KeyError, ReasoningPolicyError):
                invalid = True
            else:
                invalid = invalid or any(
                    (
                        row.get("contract") != REASONING_POLICY_CONTRACT,
                        row.get("reasoning_cohort") != policy.cohort,
                        endpoint_api_mode(str(row.get("endpoint_path")))
                        != api_mode,
                        row.get("serialized_control_field") != field,
                        row.get("serialized_control_value")
                        != (control.get(field) if field else None),
                        row.get("legacy_enable_thinking_present") is not False,
                        not isinstance(row.get("conflicting_fields_removed"), list),
                    )
                )
        invalid_contract_rows += int(invalid)
    policy_inconsistent = bool(requests) and (
        len(policies) != 1
        or sum(
            isinstance(row.get("requested_reasoning_policy"), str)
            for row in requests
        )
        != len(requests)
    )
    request_observer_ok = not any(
        (
            malformed_rows,
            unknown_event_rows,
            read_error,
            invalid_contract_rows,
            policy_inconsistent,
            observer_errors,
            duplicate_request_ids,
            duplicate_response_ids,
            orphan_response_ids,
            parser_errors,
            transport_errors,
        )
    )
    observer_ok = request_observer_ok and not missing_response_ids
    return {
        "contract": MODEL_TRANSPORT_CONTRACT,
        "reasoning_policy_contract": REASONING_POLICY_CONTRACT,
        "requested_reasoning_policy": policies[0] if len(policies) == 1 else None,
        "request_count": len(requests),
        "response_count": len(responses),
        "requests": requests,
        "responses": responses,
        "visible_content_returned": any(
            row.get("visible_content_returned") is True for row in responses
        ),
        "reasoning_content_returned": any(
            row.get("reasoning_content_returned") is True for row in responses
        ),
        "tool_call_returned": any(
            row.get("tool_call_returned") is True for row in responses
        ),
        "finish_state_returned": any(
            row.get("finish_state_returned") is True for row in responses
        ),
        "finish_reasons": sorted(
            {
                reason
                for row in responses
                for reason in row.get("finish_reasons", [])
                if isinstance(reason, str)
            }
        ),
        "response_completed_count": sum(
            row.get("finish_state_returned") is True for row in responses
        ),
        "malformed_row_count": malformed_rows,
        "unknown_event_row_count": unknown_event_rows,
        "invalid_contract_row_count": invalid_contract_rows,
        "policy_inconsistent": policy_inconsistent,
        "observer_errors": observer_errors,
        "read_error": read_error,
        "parser_errors": list(dict.fromkeys(parser_errors)),
        "transport_errors": transport_errors,
        "duplicate_request_ids": duplicate_request_ids,
        "duplicate_response_ids": duplicate_response_ids,
        "orphan_response_ids": orphan_response_ids,
        "missing_response_ids": missing_response_ids,
        "request_observer_ok": request_observer_ok,
        "observer_ok": observer_ok,
    }


def combine_model_transport_observations(
    observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(observations.values())
    policies = {
        row.get("requested_reasoning_policy")
        for row in rows
        if isinstance(row.get("requested_reasoning_policy"), str)
    }
    return {
        "contract": MODEL_TRANSPORT_CONTRACT,
        "reasoning_policy_contract": REASONING_POLICY_CONTRACT,
        "requested_reasoning_policy": (
            next(iter(policies)) if len(policies) == 1 else None
        ),
        "request_count": sum(int(row.get("request_count") or 0) for row in rows),
        "response_count": sum(int(row.get("response_count") or 0) for row in rows),
        "visible_content_returned": any(
            row.get("visible_content_returned") is True for row in rows
        ),
        "reasoning_content_returned": any(
            row.get("reasoning_content_returned") is True for row in rows
        ),
        "tool_call_returned": any(
            row.get("tool_call_returned") is True for row in rows
        ),
        "finish_state_returned": all(
            row.get("finish_state_returned") is True for row in rows
        ) if rows else False,
        "observer_ok": bool(rows) and all(
            row.get("observer_ok") is True for row in rows
        ),
        "stages": {name: dict(value) for name, value in observations.items()},
    }
