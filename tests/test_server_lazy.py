from __future__ import annotations


def test_server_import_keeps_heavy_resources_lazy() -> None:
    import pytest

    pytest.importorskip("mcp")
    from blazing_rag_mcp import server

    assert server._app._embeddings is None
    assert server._app._vector_index is None
    assert server._app._store is None


def test_mcp_initialize_and_tools_list_without_heavy_runtime(tmp_path) -> None:
    import os
    import sys

    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def exercise() -> None:
        env = dict(os.environ)
        env.update(
            {
                "BRAG_ROOTS": tmp_path.as_posix(),
                "BRAG_DB_DIR": (tmp_path / ".brag").as_posix(),
                "BRAG_DEVICE": "cpu",
                "BRAG_EMBEDDING_ALLOW_HASH_FALLBACK": "true",
                "BRAG_READ_ONLY": "true",
                "BRAG_LOG_LEVEL": "WARNING",
            }
        )
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "blazing_rag_mcp.server"],
            env=env,
        )
        async with (
            stdio_client(params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            initialized = await session.initialize()
            tools = await session.list_tools()
            assert initialized.serverInfo.name == "blazing-code-rag"
            names = {tool.name for tool in tools.tools}
            assert {
                "code_search",
                "code_update_index",
                "document_search",
                "document_outline",
                "document_fetch",
                "document_update_index",
                "rag_doctor",
            } <= names

    anyio.run(exercise)
