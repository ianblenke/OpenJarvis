"""Tests for the file_write tool.

Covers FileWriteTool: execute() writing files, creating directories,
append mode, allowed_dirs, sensitive file blocking, and error cases.
Uses tmp_path for real file system operations.
"""

from __future__ import annotations

import pytest

from openjarvis.tools.file_write import FileWriteTool


class TestFileWriteTool:
    @pytest.mark.spec("REQ-tools.file-write.spec")
    def test_spec_name(self):
        tool = FileWriteTool()
        assert tool.spec.name == "file_write"

    @pytest.mark.spec("REQ-tools.file-write.spec")
    def test_spec_category(self):
        tool = FileWriteTool()
        assert tool.spec.category == "filesystem"

    @pytest.mark.spec("REQ-tools.file-write.spec")
    def test_spec_required_capabilities(self):
        tool = FileWriteTool()
        assert "file:write" in tool.spec.required_capabilities

    @pytest.mark.spec("REQ-tools.file-write.spec")
    def test_spec_has_path_param(self):
        tool = FileWriteTool()
        assert "path" in tool.spec.parameters["properties"]

    @pytest.mark.spec("REQ-tools.file-write.spec")
    def test_spec_has_content_param(self):
        tool = FileWriteTool()
        assert "content" in tool.spec.parameters["properties"]

    @pytest.mark.spec("REQ-tools.file-write.spec")
    def test_spec_required_fields(self):
        tool = FileWriteTool()
        required = tool.spec.parameters["required"]
        assert "path" in required
        assert "content" in required

    @pytest.mark.spec("REQ-tools.file-write.spec")
    def test_tool_id(self):
        tool = FileWriteTool()
        assert tool.tool_id == "file_write"

    @pytest.mark.spec("REQ-tools.file-write.spec")
    def test_spec_has_mode_param(self):
        tool = FileWriteTool()
        assert "mode" in tool.spec.parameters["properties"]

    @pytest.mark.spec("REQ-tools.file-write.spec")
    def test_spec_has_create_dirs_param(self):
        tool = FileWriteTool()
        assert "create_dirs" in tool.spec.parameters["properties"]

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_no_path(self):
        tool = FileWriteTool()
        result = tool.execute(path="", content="hello")
        assert result.success is False
        assert "No path" in result.content

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_no_content(self):
        tool = FileWriteTool()
        result = tool.execute(path="/tmp/test.txt")
        assert result.success is False
        assert "No content" in result.content

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_write_file(self, tmp_path):
        f = tmp_path / "test.txt"
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content="hello world\n")
        assert result.success is True
        assert f.read_text(encoding="utf-8") == "hello world\n"
        assert result.metadata["size_bytes"] > 0
        assert result.metadata["path"] == str(f.resolve())

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_write_creates_new_file(self, tmp_path):
        f = tmp_path / "new_file.txt"
        assert not f.exists()
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content="brand new")
        assert result.success is True
        assert f.exists()
        assert f.read_text(encoding="utf-8") == "brand new"

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_overwrite_existing_file(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old content", encoding="utf-8")
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content="new content")
        assert result.success is True
        assert f.read_text(encoding="utf-8") == "new content"

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_append_mode(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\n", encoding="utf-8")
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content="line2\n", mode="append")
        assert result.success is True
        assert f.read_text(encoding="utf-8") == "line1\nline2\n"

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_append_to_new_file(self, tmp_path):
        f = tmp_path / "new_append.txt"
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content="first line\n", mode="append")
        assert result.success is True
        assert f.read_text(encoding="utf-8") == "first line\n"

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_invalid_mode(self, tmp_path):
        f = tmp_path / "test.txt"
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content="data", mode="invalid")
        assert result.success is False
        assert "Invalid mode" in result.content

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_create_dirs(self, tmp_path):
        f = tmp_path / "sub" / "deep" / "test.txt"
        tool = FileWriteTool()
        result = tool.execute(
            path=str(f), content="nested", create_dirs=True,
        )
        assert result.success is True
        assert f.read_text(encoding="utf-8") == "nested"

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_create_dirs_false_missing_parent(self, tmp_path):
        f = tmp_path / "nonexistent" / "test.txt"
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content="data")
        assert result.success is False
        assert "Parent directory does not exist" in result.content

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_file_size_limit(self, tmp_path):
        f = tmp_path / "big.txt"
        # 10 MB + 1 byte exceeds the limit
        big_content = "x" * 10_485_761
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content=big_content)
        assert result.success is False
        assert "too large" in result.content.lower()

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_success_message(self, tmp_path):
        f = tmp_path / "msg.txt"
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content="data")
        assert result.success is True
        assert "Successfully wrote" in result.content

    @pytest.mark.spec("REQ-tools.file-write.security")
    def test_blocks_env_file(self, tmp_path):
        f = tmp_path / ".env"
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content="SECRET=foo")
        assert result.success is False
        assert "sensitive" in result.content.lower()

    @pytest.mark.spec("REQ-tools.file-write.security")
    def test_blocks_pem_file(self, tmp_path):
        f = tmp_path / "server.pem"
        tool = FileWriteTool()
        result = tool.execute(
            path=str(f), content="-----BEGIN CERTIFICATE-----",
        )
        assert result.success is False
        assert "sensitive" in result.content.lower()

    @pytest.mark.spec("REQ-tools.file-write.security")
    def test_blocks_credentials_json(self, tmp_path):
        f = tmp_path / "credentials.json"
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content='{"token": "abc"}')
        assert result.success is False
        assert "sensitive" in result.content.lower()

    @pytest.mark.spec("REQ-tools.file-write.security")
    def test_blocks_key_file(self, tmp_path):
        f = tmp_path / "private.key"
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content="key data")
        assert result.success is False
        assert "sensitive" in result.content.lower()

    @pytest.mark.spec("REQ-tools.file-write.allowed-dirs")
    def test_allowed_dirs_blocks(self, tmp_path):
        f = tmp_path / "test.txt"
        tool = FileWriteTool(allowed_dirs=["/some/other/dir"])
        result = tool.execute(path=str(f), content="data")
        assert result.success is False
        assert "Access denied" in result.content

    @pytest.mark.spec("REQ-tools.file-write.allowed-dirs")
    def test_allowed_dirs_permits(self, tmp_path):
        f = tmp_path / "ok.txt"
        tool = FileWriteTool(allowed_dirs=[str(tmp_path)])
        result = tool.execute(path=str(f), content="ok data")
        assert result.success is True
        assert f.read_text(encoding="utf-8") == "ok data"

    @pytest.mark.spec("REQ-tools.file-write.allowed-dirs")
    def test_no_allowed_dirs_permits_all(self, tmp_path):
        f = tmp_path / "any.txt"
        tool = FileWriteTool(allowed_dirs=None)
        result = tool.execute(path=str(f), content="any data")
        assert result.success is True

    @pytest.mark.spec("REQ-tools.file-write.execute")
    def test_tool_name_in_result(self, tmp_path):
        f = tmp_path / "x.txt"
        tool = FileWriteTool()
        result = tool.execute(path=str(f), content="data")
        assert result.tool_name == "file_write"

    @pytest.mark.spec("REQ-tools.file-write.openai")
    def test_openai_function(self):
        tool = FileWriteTool()
        fn = tool.to_openai_function()
        assert fn["type"] == "function"
        assert fn["function"]["name"] == "file_write"
