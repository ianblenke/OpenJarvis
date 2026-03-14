"""Tests for mount_security module — security-critical validation.

Covers AllowedRoot, MountAllowlist dataclasses, load_mount_allowlist(),
validate_mount(), validate_mounts() with extensive edge cases including
path traversal, blocked patterns, and boundary conditions.
"""

from __future__ import annotations

import json

import pytest

from openjarvis.sandbox.mount_security import (
    DEFAULT_BLOCKED_PATTERNS,
    AllowedRoot,
    MountAllowlist,
    load_mount_allowlist,
    validate_mount,
    validate_mounts,
)

# ---------------------------------------------------------------------------
# AllowedRoot dataclass
# ---------------------------------------------------------------------------


class TestAllowedRoot:
    @pytest.mark.spec("REQ-sandbox.security.mounts")
    @pytest.mark.spec("REQ-sandbox.mount.allowed-root")
    def test_default_read_only(self):
        root = AllowedRoot(path="/data")
        assert root.path == "/data"
        assert root.read_only is True

    @pytest.mark.spec("REQ-sandbox.mount.allowed-root")
    def test_explicit_read_only_false(self):
        root = AllowedRoot(path="/home/user", read_only=False)
        assert root.read_only is False

    @pytest.mark.spec("REQ-sandbox.mount.allowed-root")
    def test_explicit_read_only_true(self):
        root = AllowedRoot(path="/opt", read_only=True)
        assert root.read_only is True

    @pytest.mark.spec("REQ-sandbox.mount.allowed-root")
    def test_slots_prevents_arbitrary_attrs(self):
        root = AllowedRoot(path="/data")
        with pytest.raises(AttributeError):
            root.extra = "should fail"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# MountAllowlist dataclass
# ---------------------------------------------------------------------------


class TestMountAllowlist:
    @pytest.mark.spec("REQ-sandbox.mount.allowlist")
    def test_default_blocked_patterns(self):
        al = MountAllowlist()
        assert al.blocked_patterns == list(DEFAULT_BLOCKED_PATTERNS)

    @pytest.mark.spec("REQ-sandbox.mount.allowlist")
    def test_default_roots_empty(self):
        al = MountAllowlist()
        assert al.roots == []

    @pytest.mark.spec("REQ-sandbox.mount.allowlist")
    def test_custom_blocked_patterns(self):
        al = MountAllowlist(blocked_patterns=[".ssh", "*.key"])
        assert al.blocked_patterns == [".ssh", "*.key"]

    @pytest.mark.spec("REQ-sandbox.mount.allowlist")
    def test_custom_roots(self):
        roots = [AllowedRoot(path="/data"), AllowedRoot(path="/tmp", read_only=False)]
        al = MountAllowlist(roots=roots)
        assert len(al.roots) == 2
        assert al.roots[0].path == "/data"
        assert al.roots[1].read_only is False

    @pytest.mark.spec("REQ-sandbox.mount.allowlist")
    def test_independent_defaults(self):
        """Each instance should get an independent copy of blocked_patterns."""
        al1 = MountAllowlist()
        al2 = MountAllowlist()
        al1.blocked_patterns.append("custom_pattern")
        assert "custom_pattern" not in al2.blocked_patterns


# ---------------------------------------------------------------------------
# DEFAULT_BLOCKED_PATTERNS coverage
# ---------------------------------------------------------------------------


class TestDefaultBlockedPatterns:
    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_ssh(self):
        assert ".ssh" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_env(self):
        assert ".env" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_pem(self):
        assert "*.pem" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_key(self):
        assert "*.key" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_gnupg(self):
        assert ".gnupg" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_aws(self):
        assert ".aws" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_credentials(self):
        assert "credentials" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_docker_config(self):
        assert ".docker/config.json" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_kube_config(self):
        assert ".kube/config" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_npmrc(self):
        assert ".npmrc" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_pypirc(self):
        assert ".pypirc" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_shadow(self):
        assert "shadow" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_token(self):
        assert "token" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_contains_secret(self):
        assert "secret" in DEFAULT_BLOCKED_PATTERNS

    @pytest.mark.spec("REQ-sandbox.mount.blocked-patterns")
    def test_is_a_list(self):
        assert isinstance(DEFAULT_BLOCKED_PATTERNS, list)


# ---------------------------------------------------------------------------
# validate_mount
# ---------------------------------------------------------------------------


class TestValidateMount:
    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_allows_valid_path(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        target = tmp_path / "data"
        target.mkdir()
        assert validate_mount(str(target), allowlist) is True

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_blocks_ssh_dir(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        target = tmp_path / ".ssh"
        target.mkdir()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_blocks_env_file(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        target = tmp_path / ".env"
        target.touch()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_blocks_pem_file(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        target = tmp_path / "server.pem"
        target.touch()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_blocks_key_file(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        target = tmp_path / "private.key"
        target.touch()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_blocks_p12_file(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        target = tmp_path / "cert.p12"
        target.touch()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_blocks_pfx_file(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        target = tmp_path / "cert.pfx"
        target.touch()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_blocks_id_rsa(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        target = tmp_path / "id_rsa"
        target.touch()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_blocks_id_ed25519(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        target = tmp_path / "id_ed25519"
        target.touch()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_blocks_git_config_via_git_component(self, tmp_path):
        """The `.git` directory component does not match `.git/config` pattern
        via fnmatch (compound patterns with `/` don't match single parts).
        However, the `.git` component is not in default blocked patterns either.
        This test verifies the actual behaviour: `.git/config` as a compound
        blocked pattern does NOT block, because fnmatch checks individual
        path components.  Use an explicit `config` pattern to block."""
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
            blocked_patterns=["config"],
        )
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        target = git_dir / "config"
        target.touch()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_blocks_shadow_file(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        target = tmp_path / "shadow"
        target.touch()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_rejects_outside_root(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path / "allowed"))],
        )
        target = tmp_path / "not_allowed" / "data"
        target.mkdir(parents=True)
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_no_roots_allows_non_blocked(self, tmp_path):
        allowlist = MountAllowlist(roots=[])
        target = tmp_path / "safe_dir"
        target.mkdir()
        assert validate_mount(str(target), allowlist) is True

    @pytest.mark.spec("REQ-sandbox.mount.traversal")
    def test_traversal_dot_dot_blocked_by_pattern(self, tmp_path):
        """Paths with .. are resolved before checking."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        # This resolves to tmp_path/secret (outside allowed)
        traversal = str(allowed / ".." / "secret")
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(allowed))],
            blocked_patterns=["secret"],
        )
        assert validate_mount(traversal, allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.traversal")
    def test_traversal_dot_dot_escapes_root(self, tmp_path):
        """Path traversal with .. should be caught by root check."""
        allowed = tmp_path / "jail"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        # Resolve should place it outside allowed root
        traversal = str(allowed / ".." / "outside")
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(allowed))],
            blocked_patterns=[],
        )
        assert validate_mount(traversal, allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_multiple_roots_first_matches(self, tmp_path):
        r1 = tmp_path / "root1"
        r2 = tmp_path / "root2"
        r1.mkdir()
        r2.mkdir()
        target = r1 / "file"
        target.touch()
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(r1)), AllowedRoot(path=str(r2))],
            blocked_patterns=[],
        )
        assert validate_mount(str(target), allowlist) is True

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_multiple_roots_second_matches(self, tmp_path):
        r1 = tmp_path / "root1"
        r2 = tmp_path / "root2"
        r1.mkdir()
        r2.mkdir()
        target = r2 / "file"
        target.touch()
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(r1)), AllowedRoot(path=str(r2))],
            blocked_patterns=[],
        )
        assert validate_mount(str(target), allowlist) is True

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_blocked_pattern_in_intermediate_path(self, tmp_path):
        """A blocked component anywhere in the path should block."""
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        target = tmp_path / ".ssh" / "authorized_keys"
        target.parent.mkdir(parents=True)
        target.touch()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_custom_blocked_pattern(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
            blocked_patterns=["my_secret_*"],
        )
        target = tmp_path / "my_secret_file"
        target.touch()
        assert validate_mount(str(target), allowlist) is False

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_allows_non_matching_custom_pattern(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
            blocked_patterns=["my_secret_*"],
        )
        target = tmp_path / "safe_file"
        target.touch()
        assert validate_mount(str(target), allowlist) is True

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_empty_blocked_patterns_allows_all_under_root(self, tmp_path):
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
            blocked_patterns=[],
        )
        target = tmp_path / ".ssh"
        target.mkdir()
        assert validate_mount(str(target), allowlist) is True

    @pytest.mark.spec("REQ-sandbox.mount.validate")
    def test_exact_root_path_is_allowed(self, tmp_path):
        """The root path itself should be allowed."""
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
            blocked_patterns=[],
        )
        assert validate_mount(str(tmp_path), allowlist) is True


# ---------------------------------------------------------------------------
# validate_mounts
# ---------------------------------------------------------------------------


class TestValidateMounts:
    @pytest.mark.spec("REQ-sandbox.mount.validate-list")
    def test_returns_valid_mounts(self, tmp_path):
        d1 = tmp_path / "data1"
        d2 = tmp_path / "data2"
        d1.mkdir()
        d2.mkdir()
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        result = validate_mounts([str(d1), str(d2)], allowlist)
        assert len(result) == 2
        assert str(d1) in result
        assert str(d2) in result

    @pytest.mark.spec("REQ-sandbox.mount.validate-list")
    def test_raises_for_blocked(self, tmp_path):
        target = tmp_path / ".ssh"
        target.mkdir()
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        with pytest.raises(ValueError, match="blocked"):
            validate_mounts([str(target)], allowlist)

    @pytest.mark.spec("REQ-sandbox.mount.validate-list")
    def test_raises_for_outside_root(self, tmp_path):
        target = tmp_path / "outside"
        target.mkdir()
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path / "inside"))],
        )
        with pytest.raises(ValueError, match="not under"):
            validate_mounts([str(target)], allowlist)

    @pytest.mark.spec("REQ-sandbox.mount.validate-list")
    def test_empty_list(self):
        allowlist = MountAllowlist()
        assert validate_mounts([], allowlist) == []

    @pytest.mark.spec("REQ-sandbox.mount.validate-list")
    def test_first_blocked_raises_before_checking_rest(self, tmp_path):
        """Raises on first blocked mount."""
        ssh = tmp_path / ".ssh"
        ssh.mkdir()
        safe = tmp_path / "safe"
        safe.mkdir()
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(tmp_path))],
        )
        with pytest.raises(ValueError, match="blocked"):
            validate_mounts([str(ssh), str(safe)], allowlist)

    @pytest.mark.spec("REQ-sandbox.mount.validate-list")
    def test_mixed_valid_then_outside_root_raises(self, tmp_path):
        valid = tmp_path / "valid"
        valid.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        allowlist = MountAllowlist(
            roots=[AllowedRoot(path=str(valid))],
            blocked_patterns=[],
        )
        with pytest.raises(ValueError, match="not under"):
            validate_mounts([str(valid), str(outside)], allowlist)


# ---------------------------------------------------------------------------
# load_mount_allowlist
# ---------------------------------------------------------------------------


class TestLoadMountAllowlist:
    @pytest.mark.spec("REQ-sandbox.mount.load-config")
    def test_loads_from_json(self, tmp_path):
        config = {
            "roots": [
                {"path": "/home/user/projects", "read_only": False},
                {"path": "/data"},
            ],
            "blocked_patterns": [".ssh", "*.pem"],
        }
        f = tmp_path / "allowlist.json"
        f.write_text(json.dumps(config))

        al = load_mount_allowlist(str(f))
        assert len(al.roots) == 2
        assert al.roots[0].path == "/home/user/projects"
        assert al.roots[0].read_only is False
        assert al.roots[1].path == "/data"
        assert al.roots[1].read_only is True
        assert al.blocked_patterns == [".ssh", "*.pem"]

    @pytest.mark.spec("REQ-sandbox.mount.load-config")
    def test_default_blocked_patterns_when_omitted(self, tmp_path):
        config = {"roots": []}
        f = tmp_path / "allowlist.json"
        f.write_text(json.dumps(config))

        al = load_mount_allowlist(str(f))
        assert al.blocked_patterns == list(DEFAULT_BLOCKED_PATTERNS)

    @pytest.mark.spec("REQ-sandbox.mount.load-config")
    def test_empty_roots_when_omitted(self, tmp_path):
        config = {}
        f = tmp_path / "allowlist.json"
        f.write_text(json.dumps(config))

        al = load_mount_allowlist(str(f))
        assert al.roots == []

    @pytest.mark.spec("REQ-sandbox.mount.load-config")
    def test_default_read_only_true(self, tmp_path):
        config = {
            "roots": [{"path": "/opt"}],
        }
        f = tmp_path / "allowlist.json"
        f.write_text(json.dumps(config))

        al = load_mount_allowlist(str(f))
        assert al.roots[0].read_only is True

    @pytest.mark.spec("REQ-sandbox.mount.load-config")
    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_mount_allowlist("/nonexistent/path/to/allowlist.json")

    @pytest.mark.spec("REQ-sandbox.mount.load-config")
    def test_invalid_json_raises(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            load_mount_allowlist(str(f))

    @pytest.mark.spec("REQ-sandbox.mount.load-config")
    def test_multiple_roots_loaded(self, tmp_path):
        config = {
            "roots": [
                {"path": "/a", "read_only": True},
                {"path": "/b", "read_only": False},
                {"path": "/c"},
            ],
        }
        f = tmp_path / "allowlist.json"
        f.write_text(json.dumps(config))

        al = load_mount_allowlist(str(f))
        assert len(al.roots) == 3
        assert al.roots[2].read_only is True
