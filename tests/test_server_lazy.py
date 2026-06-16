from __future__ import annotations


def test_server_import_does_not_construct_runtime() -> None:
    import pytest
    pytest.importorskip("mcp")
    from blazing_rag_mcp import server

    assert server._runtime is None
