from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


class EndpointPolicyError(ValueError):
    """Raised when an endpoint is outside the local qualification boundary."""


@dataclass(frozen=True)
class LocalEndpoint:
    base_url: str
    host: str
    port: int

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"


def validate_local_openai_endpoint(value: str) -> LocalEndpoint:
    """Validate and normalize an explicit local OpenAI-compatible ``/v1`` URL.

    Only clear-text loopback or RFC1918 numeric addresses are accepted.  The
    model endpoint is intentionally not allowed to depend on public DNS, URL
    credentials, redirects, or an implicit port.
    """

    if not isinstance(value, str) or not value.strip():
        raise EndpointPolicyError("local endpoint is missing")

    parsed = urlsplit(value.strip())
    if parsed.scheme != "http":
        raise EndpointPolicyError("local endpoint must use http")
    if parsed.username is not None or parsed.password is not None:
        raise EndpointPolicyError("endpoint credentials must not be embedded in the URL")
    if parsed.query or parsed.fragment:
        raise EndpointPolicyError("endpoint must not contain a query string or fragment")
    if parsed.path.rstrip("/") != "/v1":
        raise EndpointPolicyError("OpenAI-compatible endpoint path must be exactly /v1")

    host = parsed.hostname or ""
    if host.lower() == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise EndpointPolicyError(
            "endpoint host must be localhost or a numeric private address"
        ) from exc
    private = address.is_loopback
    if isinstance(address, ipaddress.IPv4Address):
        private = private or any(
            address in network
            for network in (
                ipaddress.ip_network("10.0.0.0/8"),
                ipaddress.ip_network("172.16.0.0/12"),
                ipaddress.ip_network("192.168.0.0/16"),
            )
        )
    else:
        private = private or address in ipaddress.ip_network("fc00::/7")
    if not private:
        raise EndpointPolicyError("endpoint host must be loopback or RFC1918/private")

    try:
        port = parsed.port
    except ValueError as exc:
        raise EndpointPolicyError("endpoint port is invalid") from exc
    if port is None:
        raise EndpointPolicyError("endpoint must include an explicit port")

    normalized_host = address.compressed
    netloc = f"[{normalized_host}]:{port}" if address.version == 6 else f"{normalized_host}:{port}"
    return LocalEndpoint(
        base_url=urlunsplit(("http", netloc, "/v1", "", "")),
        host=normalized_host,
        port=port,
    )
