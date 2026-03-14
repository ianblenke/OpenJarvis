#!/usr/bin/env python3
"""Check bidirectional coverage between OpenSpec specs and tests.

Ensures:
1. Every spec requirement has at least one test referencing it.
2. Every test with @pytest.mark.spec references a valid spec requirement.

Exit code 0 = all good, 1 = gaps found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = REPO_ROOT / "openspec" / "specs"
TESTS_DIR = REPO_ROOT / "tests"
FRONTEND_SRC_DIR = REPO_ROOT / "frontend" / "src"

# Pattern to find requirement headings in spec files: ### REQ-<id>: <description>
SPEC_REQ_PATTERN = re.compile(r"^###\s+(REQ-[\w./-]+):", re.MULTILINE)

# Pattern to find @pytest.mark.spec("...") in test files
TEST_SPEC_PATTERN = re.compile(r'@pytest\.mark\.spec\(["\']([^"\']+)["\']\)')

# Pattern to find pytest.mark.spec in parametrize or similar
TEST_SPEC_PARAM_PATTERN = re.compile(r'pytest\.mark\.spec\(["\']([^"\']+)["\']\)')

# Mapping of frontend test file patterns to spec requirements they cover
FRONTEND_TEST_SPEC_MAP: dict[str, list[str]] = {
    "lib/api.test.ts": [
        "REQ-frontend.api.agents",
        "REQ-frontend.api.chat",
        "REQ-frontend.api.models",
        "REQ-frontend.api.speech",
        "REQ-frontend.api.tauri",
    ],
    "lib/store.test.ts": [
        "REQ-frontend.store.agents",
        "REQ-frontend.store.conversations",
        "REQ-frontend.store.messages",
        "REQ-frontend.store.settings",
        "REQ-frontend.store.streaming",
    ],
    "lib/sse.test.ts": ["REQ-frontend.sse.stream"],
    "lib/profanity.test.ts": ["REQ-frontend.profanity"],
    "hooks/useSpeech.test.ts": ["REQ-frontend.hooks.speech"],
    "components/ErrorBoundary.test.tsx": ["REQ-frontend.components.error"],
    "components/Chat/MessageBubble.test.tsx": ["REQ-frontend.components.message"],
    "pages/AgentsPage.test.tsx": ["REQ-frontend.pages.agents"],
    "pages/ChatPage.test.tsx": ["REQ-frontend.pages.chat"],
    "pages/DashboardPage.test.tsx": ["REQ-frontend.pages.dashboard"],
    "pages/SettingsPage.test.tsx": ["REQ-frontend.pages.settings"],
}


def collect_spec_requirements() -> dict[str, Path]:
    """Return {requirement_id: spec_file_path} for all specs."""
    reqs: dict[str, Path] = {}
    if not SPECS_DIR.exists():
        return reqs
    for spec_file in SPECS_DIR.rglob("*.md"):
        content = spec_file.read_text(encoding="utf-8")
        for match in SPEC_REQ_PATTERN.finditer(content):
            req_id = match.group(1)
            reqs[req_id] = spec_file
    return reqs


def collect_test_spec_refs() -> dict[str, list[Path]]:
    """Return {spec_ref: [test_file_paths]} for all tests (Python + frontend)."""
    refs: dict[str, list[Path]] = {}
    # Python tests
    if TESTS_DIR.exists():
        for test_file in TESTS_DIR.rglob("test_*.py"):
            content = test_file.read_text(encoding="utf-8")
            for pattern in (TEST_SPEC_PATTERN, TEST_SPEC_PARAM_PATTERN):
                for match in pattern.finditer(content):
                    ref = match.group(1)
                    refs.setdefault(ref, []).append(test_file)
    # Frontend tests (TypeScript/React)
    if FRONTEND_SRC_DIR.exists():
        for rel_path, spec_ids in FRONTEND_TEST_SPEC_MAP.items():
            test_file = FRONTEND_SRC_DIR / rel_path
            if test_file.exists():
                for spec_id in spec_ids:
                    refs.setdefault(spec_id, []).append(test_file)
    return refs


def main() -> int:
    spec_reqs = collect_spec_requirements()
    test_refs = collect_test_spec_refs()

    errors = 0

    # Check: every spec requirement has at least one test
    untested_reqs = set(spec_reqs.keys()) - set(test_refs.keys())
    if untested_reqs:
        print("Spec requirements with NO test coverage:")
        for req in sorted(untested_reqs):
            print(f"  {req}  (in {spec_reqs[req].relative_to(REPO_ROOT)})")
        errors += len(untested_reqs)

    # Check: every test spec ref points to a valid requirement
    invalid_refs = set(test_refs.keys()) - set(spec_reqs.keys())
    if invalid_refs:
        print("\nTest @pytest.mark.spec refs with NO matching spec requirement:")
        for ref in sorted(invalid_refs):
            files = ", ".join(
                str(f.relative_to(REPO_ROOT)) for f in test_refs[ref]
            )
            print(f"  {ref}  (referenced in {files})")
        errors += len(invalid_refs)

    if errors:
        print(f"\n{errors} spec coverage issue(s) found.")
        return 1

    # Summary
    total_reqs = len(spec_reqs)
    total_refs = len(test_refs)
    if total_reqs == 0 and total_refs == 0:
        print("No spec requirements or test spec refs found yet. OK (bootstrap mode).")
    else:
        covered = len(set(spec_reqs.keys()) & set(test_refs.keys()))
        print(f"Spec coverage: {covered}/{total_reqs} requirements covered by tests.")
        print(f"Test refs: {total_refs} unique spec references across test files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
