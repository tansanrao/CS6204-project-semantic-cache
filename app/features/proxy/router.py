"""FastAPI router exposing the OpenAI-compatible proxy endpoints."""

from fastapi import APIRouter, Depends, Request, Response

from .service import ProxyService


def get_proxy_service(request: Request) -> ProxyService:
    """Retrieve the proxy service instance from the FastAPI app state."""
    proxy_service: ProxyService | None = getattr(
        request.app.state, "proxy_service", None
    )
    if proxy_service is None:
        raise RuntimeError("Proxy service is not configured on application state.")
    return proxy_service


def build_router() -> APIRouter:
    """Construct the router for forwarding OpenAI-compatible requests."""
    router = APIRouter()

    @router.api_route(
        "/v1",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    async def proxy_root(
        request: Request, service: ProxyService = Depends(get_proxy_service)
    ) -> Response:
        return await service.forward(path="", request=request)

    @router.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    async def proxy_path(
        path: str,
        request: Request,
        service: ProxyService = Depends(get_proxy_service),
    ) -> Response:
        return await service.forward(path=path, request=request)

    return router
