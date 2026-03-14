# Templates Module Spec

Pre-configured agent manifests for quick agent creation.

## AgentTemplate

### REQ-templates.template: Template definition
`AgentTemplate` with `name`, `description`, `system_prompt`, `agent_type`, `tools`, `max_turns`, `temperature`.

## Loading

### REQ-templates.load: Template loading
`load_template(path) -> AgentTemplate` loads from TOML files.

### REQ-templates.discover: Template discovery
`discover_templates(extra_dirs) -> List[AgentTemplate]` searches built-in, user `~/.openjarvis/templates/agents/`, and extra directories.

## AgentTemplate (detailed)

### REQ-templates.agent.create: Template creation with all fields
AgentTemplate can be created with all fields specified (name, description, system_prompt, agent_type, tools, max_turns, temperature).

### REQ-templates.agent.defaults: Template default values
AgentTemplate provides sensible defaults for optional fields (empty tools, default temperature).

## Loading (detailed)

### REQ-templates.loader.toml: TOML template loading
`load_template()` parses TOML files into AgentTemplate instances, using filename stem as default name, handling missing sections, and raising on missing files.

## Discovery (detailed)

### REQ-templates.discover.extra-dirs: Extra directory discovery
`discover_templates()` scans additional user-specified directories for template files.

### REQ-templates.discover.sort: Discovery sorted by name
`discover_templates()` returns templates sorted alphabetically by name.

### REQ-templates.discover.empty-dir: Discovery with empty directory
`discover_templates()` returns an empty list when the target directory has no template files.

### REQ-templates.discover.override: Discovery name override
Discovered templates respect name overrides specified in the TOML file.

## Tests

- `tests/templates/test_templates.py` - Template loading tests
