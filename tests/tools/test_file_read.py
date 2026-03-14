"""Tests for the file_read tool.

Covers FileReadTool: execute() reading real files, error on missing files,
line range support, allowed_dirs, and sensitive file blocking.
Uses tmp_path to create real test files.
"""

from __future__ import annotations

import pytest

from openjarvis.tools.file_read import FileReadTool


class TestFileReadTool:
    @pytest.mark.spec("REQ-tools.file-read.spec")
    @pytest.mark.spec("REQ-tools.file-read")
    def test_spec_name(self):
        tool = FileReadTool()
        assert tool.spec.name == "file_read"

    @pytest.mark.spec("REQ-tools.file-read.spec")
    def test_spec_category(self):
        tool = FileReadTool()
        assert tool.spec.category == "filesystem"

    @pytest.mark.spec("REQ-tools.file-read.spec")
    def test_spec_has_path_param(self):
        tool = FileReadTool()
        assert "path" in tool.spec.parameters["properties"]

    @pytest.mark.spec("REQ-tools.file-read.spec")
    def test_spec_path_required(self):
        tool = FileReadTool()
        assert "path" in tool.spec.parameters["required"]

    @pytest.mark.spec("REQ-tools.file-read.spec")
    def test_spec_has_max_lines_param(self):
        tool = FileReadTool()
        assert "max_lines" in tool.spec.parameters["properties"]

    @pytest.mark.spec("REQ-tools.file-read.spec")
    def test_tool_id(self):
        tool = FileReadTool()
        assert tool.tool_id == "file_read"

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_no_path(self):
        tool = FileReadTool()
        result = tool.execute(path="")
        assert result.success is False
        assert "No path" in result.content

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_no_path_param(self):
        tool = FileReadTool()
        result = tool.execute()
        assert result.success is False
        assert "No path" in result.content

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_file_not_found(self):
        tool = FileReadTool()
        result = tool.execute(path="/nonexistent/file.txt")
        assert result.success is False
        assert "File not found" in result.content

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_read_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world\nsecond line\n", encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f))
        assert result.success is True
        assert "hello world" in result.content
        assert result.metadata["size_bytes"] > 0

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_read_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f))
        assert result.success is True

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_max_lines(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f), max_lines=2)
        assert result.success is True
        assert "line1" in result.content
        assert "line2" in result.content
        assert "line3" not in result.content

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_max_lines_zero_returns_all(self, tmp_path):
        """max_lines=0 should return all content (falsy check)."""
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f), max_lines=0)
        assert result.success is True
        assert "line3" in result.content

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_max_lines_greater_than_file_length(self, tmp_path):
        f = tmp_path / "short.txt"
        f.write_text("only\n", encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f), max_lines=100)
        assert result.success is True
        assert "only" in result.content

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_metadata_path_resolved(self, tmp_path):
        f = tmp_path / "meta.txt"
        f.write_text("content", encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f))
        assert result.success is True
        assert result.metadata["path"] == str(f.resolve())

    @pytest.mark.spec("REQ-tools.file-read.allowed-dirs")
    def test_allowed_dirs_blocks(self, tmp_path):
        f = tmp_path / "secret.txt"
        f.write_text("secret data", encoding="utf-8")
        tool = FileReadTool(allowed_dirs=["/some/other/dir"])
        result = tool.execute(path=str(f))
        assert result.success is False
        assert "Access denied" in result.content

    @pytest.mark.spec("REQ-tools.file-read.allowed-dirs")
    def test_allowed_dirs_permits(self, tmp_path):
        f = tmp_path / "ok.txt"
        f.write_text("ok data", encoding="utf-8")
        tool = FileReadTool(allowed_dirs=[str(tmp_path)])
        result = tool.execute(path=str(f))
        assert result.success is True
        assert "ok data" in result.content

    @pytest.mark.spec("REQ-tools.file-read.allowed-dirs")
    def test_no_allowed_dirs_permits_all(self, tmp_path):
        f = tmp_path / "any.txt"
        f.write_text("any content", encoding="utf-8")
        tool = FileReadTool(allowed_dirs=None)
        result = tool.execute(path=str(f))
        assert result.success is True

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_directory_not_file(self, tmp_path):
        tool = FileReadTool()
        result = tool.execute(path=str(tmp_path))
        assert result.success is False
        assert "Not a file" in result.content

    @pytest.mark.spec("REQ-tools.file-read.security")
    def test_blocks_env_file(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("SECRET=foo", encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f))
        assert result.success is False
        assert "sensitive" in result.content.lower()

    @pytest.mark.spec("REQ-tools.file-read.security")
    def test_blocks_pem_file(self, tmp_path):
        f = tmp_path / "server.pem"
        f.write_text("-----BEGIN CERTIFICATE-----", encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f))
        assert result.success is False
        assert "sensitive" in result.content.lower()

    @pytest.mark.spec("REQ-tools.file-read.security")
    def test_blocks_credentials_json(self, tmp_path):
        f = tmp_path / "credentials.json"
        f.write_text('{"token": "abc"}', encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f))
        assert result.success is False
        assert "sensitive" in result.content.lower()

    @pytest.mark.spec("REQ-tools.file-read.security")
    def test_blocks_key_file(self, tmp_path):
        f = tmp_path / "private.key"
        f.write_text("key data", encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f))
        assert result.success is False
        assert "sensitive" in result.content.lower()

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_allows_normal_py_files(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("print('hello')", encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f))
        assert result.success is True

    @pytest.mark.spec("REQ-tools.file-read.execute")
    def test_tool_name_in_result(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("data", encoding="utf-8")
        tool = FileReadTool()
        result = tool.execute(path=str(f))
        assert result.tool_name == "file_read"

    @pytest.mark.spec("REQ-tools.file-read.openai")
    def test_openai_function(self):
        tool = FileReadTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "file_read"
