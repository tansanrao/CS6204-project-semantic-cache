"""Flask blueprint exposing OpenAI-compatible proxy endpoints."""

from __future__ import annotations

from flask import Blueprint, Response, current_app, request

from .service import ProxyService


def create_blueprint() -> Blueprint:
    """Return the proxy blueprint registered on the Flask app."""
    blueprint = Blueprint("proxy", __name__)

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]

    @blueprint.route("/v1", defaults={"path": ""}, methods=methods)
    @blueprint.route("/v1/<path:path>", methods=methods)
    def proxy(path: str) -> Response:
        service: ProxyService | None = current_app.config.get("proxy_service")
        if service is None:
            raise RuntimeError("Proxy service is not configured.")
        return service.forward(path=path, flask_request=request)

    return blueprint
