# /// script
# dependencies = [
#   "mcp>=1.0.0",
#   "httpx>=0.28.0",
# ]
# ///
"""MCP server for face clustering.

Run standalone:
  uv run scripts/mcp_server.py

Then any MCP host (Claude Code, Cursor, etc.) can call the
`cluster_faces` tool to group face images by identity.

This is a thin proxy over the FastAPI backend — it delegates
the heavy lifting to the running HTTP service and wraps the
result as an MCP tool response.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx

# ---- try importing mcp; fall back to a minimal stdio protocol ----
try:
    from mcp.server.fastmcp import FastMCP

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

BASE_URL = os.getenv("FC_API_URL", "http://localhost:8000")


def _cluster(files: list[str], threshold: float = 0.6,
             backend: str = "agglomerative") -> dict[str, Any]:
    """Call the FastAPI backend and return structured results."""
    with httpx.Client(base_url=BASE_URL, timeout=300) as client:
        file_objs = []
        for path in files:
            if not os.path.isfile(path):
                return {"error": f"file not found: {path}"}
            file_objs.append(("files", (os.path.basename(path), open(path, "rb"), "image/jpeg")))

        resp = client.post("/cluster", files=file_objs,
                           data={"threshold": str(threshold), "backend": backend})

        for _, fobj in file_objs:
            fobj[1].close()

        return resp.json()


def run_mcp() -> None:
    """Start the FastMCP server (stdio transport)."""
    mcp = FastMCP("face-cluster-service", version="2.0.0")

    @mcp.tool()
    def cluster_faces(files: list[str], threshold: float = 0.6,
                      backend: str = "agglomerative") -> str:
        """Group face images by identity.

        Args:
            files: Absolute paths to image files (JPEG/PNG).
            threshold: Cosine similarity threshold (0.0-1.0, default 0.6).
            backend: "agglomerative" (default) or "dbscan".

        Returns:
            JSON with cluster assignments, one file per line.
        """
        result = _cluster(files, threshold, backend)
        return json.dumps(result, indent=2, ensure_ascii=False)

    mcp.run()


def run_stdio() -> None:
    """Minimal stdio-based JSON-RPC server when `mcp` package is absent."""
    import sys

    def _respond(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    _respond({"jsonrpc": "2.0", "method": "tools/list", "params": {
        "tools": [
            {
                "name": "cluster_faces",
                "description": "Group face images by identity. Upload N images, receive clusters.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string"},
                                  "description": "Absolute paths to image files"},
                        "threshold": {"type": "number", "default": 0.6,
                                      "description": "Cosine threshold (0-1)"},
                        "backend": {"type": "string", "default": "agglomerative",
                                    "description": "agglomerative or dbscan"},
                    },
                    "required": ["files"],
                },
            }
        ]
    }})

    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("method") == "tools/call" and msg["params"]["name"] == "cluster_faces":
            args = msg["params"]["arguments"]
            result = _cluster(args.get("files", []), args.get("threshold", 0.6),
                              args.get("backend", "agglomerative"))
            _respond({"jsonrpc": "2.0", "id": msg.get("id"), "result": result})
        elif msg.get("method") in ("initialize", "notifications/initialized"):
            _respond({"jsonrpc": "2.0", "id": msg.get("id"), "result": {"protocolVersion": "0.1.0", "capabilities": {}}})


if __name__ == "__main__":
    if MCP_AVAILABLE:
        run_mcp()
    else:
        run_stdio()
