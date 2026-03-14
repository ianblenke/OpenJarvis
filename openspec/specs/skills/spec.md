# Skills Module Spec

Composable skill pipelines with step-by-step tool invocation and context chaining.

## Types (`types.py`)

### REQ-skills.types.step: Skill step
`SkillStep` with `tool_name`, `arguments_template` (Jinja2 placeholders), `output_key`.

### REQ-skills.types.manifest: Skill manifest
`SkillManifest` with `name`, `version`, `description`, `author`, `steps: List[SkillStep]`, `required_capabilities`, `signature` (Ed25519).

## Loader (`loader.py`)

### REQ-skills.loader.toml: TOML loading
`load_skill(path, *, verify_signature=False, scan_for_injection=False) -> SkillManifest` loads from TOML files.

### REQ-skills.loader.security: Security checks
Optional signature verification and injection scanning during load.

## Executor (`executor.py`)

### REQ-skills.executor.run: Sequential execution
`SkillExecutor.run(manifest, *, initial_context=None) -> SkillResult` executes steps sequentially with context chaining.

### REQ-skills.executor.context: Context propagation
Each step's output is stored under `output_key` and available to subsequent steps.

## Tool Adapter (`tool_adapter.py`)

### REQ-skills.tool-adapter: Skill as tool
`SkillTool(BaseTool)` wraps a `SkillManifest` as a tool that agents can invoke.

## Tests

- `tests/skills/test_skill_loader.py` - Skill loading tests
- `tests/skills/test_skill_executor.py` - Execution tests
