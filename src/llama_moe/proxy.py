import os
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse, StreamingResponse

UPSTREAM = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
UPSTREAM_BASE_PATH = os.getenv("UPSTREAM_BASE_PATH", "")

app = FastAPI()

# 较大的超时，避免长生成被误判为超时
client = httpx.AsyncClient(
    timeout=httpx.Timeout(300.0, read=300.0, write=300.0, connect=10.0)
)

# hop-by-hop 头不应被代理透传
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def build_upstream_url(path: str, query: str) -> str:
    path = path if path.startswith("/") else "/" + path
    base = urljoin(UPSTREAM, UPSTREAM_BASE_PATH + "/")
    url = urljoin(base, path.lstrip("/"))
    return url + (("?" + query) if query else "")


def pass_req_headers(headers: httpx.Headers) -> dict:
    out = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk == "host" or lk in HOP_BY_HOP:
            continue
        out[k] = v
    return out


def pass_resp_headers(headers: httpx.Headers) -> dict:
    out = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP:
            continue
        out[k] = v
    return out


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy(full_path: str, request: Request):
    upstream_url = build_upstream_url(full_path, request.url.query)
    body = await request.body()
    req_headers = pass_req_headers(request.headers)

    try:
        # 用 stream 模式发起上游请求，便于原样逐块转发
        async with client.stream(
            request.method,
            upstream_url,
            headers=req_headers,
            content=body,
        ) as upstream_resp:
            status = upstream_resp.status_code
            resp_headers = pass_resp_headers(upstream_resp.headers)
            ctype = upstream_resp.headers.get("content-type", "")

            # 只要是流式（常见为 text/event-stream 或 chunked），就逐块转发
            if (
                "text/event-stream" in ctype
                or upstream_resp.headers.get("transfer-encoding") == "chunked"
            ):

                async def aiter():
                    async for chunk in upstream_resp.aiter_raw():
                        # 不做任何改写，原样把上游发来的字节转回客户端
                        yield chunk

                return StreamingResponse(
                    aiter(),
                    status_code=status,
                    headers=resp_headers,
                    media_type=ctype or None,
                )

            # 非流式：聚合一次转发
            content = await upstream_resp.aread()
            return Response(content=content, status_code=status, headers=resp_headers)

    except httpx.RequestError as e:
        return PlainTextResponse(f"Upstream error: {e}", status_code=502)
