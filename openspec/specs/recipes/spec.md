# Recipes Module Spec

Universal composition format for defining complete AI agent configurations.

## Recipe

### REQ-recipes.recipe: Recipe definition
`Recipe` with `name`, `description`, `version`, `kind` ("discrete"|"operator"), intelligence settings, engine settings, agent settings, learning settings, eval settings, schedule settings, channels.

### REQ-recipes.builders: Builder integration
`to_builder_kwargs()` converts to agent builder kwargs. `to_eval_suite()` builds eval harness. `to_operator_manifest()` builds operator.

## Loading

### REQ-recipes.load: Recipe loading
`load_recipe(path) -> Recipe` loads from TOML files.

### REQ-recipes.discover: Recipe discovery
`discover_recipes(extra_dirs, *, kind=None) -> List[Recipe]` searches directories.

### REQ-recipes.resolve: Recipe resolution
`resolve_recipe(name) -> Optional[Recipe]` finds recipe by name.

## Composers

### REQ-recipes.compose-eval: Eval suite composition
`recipe_to_eval_suite()` builds eval harness from recipe.

### REQ-recipes.compose-operator: Operator composition
`recipe_to_operator()` builds operator from recipe.

## Recipe Dataclass

### REQ-recipes.dataclass.defaults: Recipe dataclass defaults
Recipe dataclass provides sensible defaults for all optional fields (kind="discrete", empty lists/dicts).

### REQ-recipes.dataclass.fields: Recipe dataclass fields
Recipe dataclass includes all required fields: name, description, version, kind, intelligence, engine, agent, learning, eval, schedule, and channels.

## Loading (detailed)

### REQ-recipes.loader.toml: TOML recipe loading
`load_recipe()` parses TOML files into Recipe dataclass instances with section-based mapping.

### REQ-recipes.loader.defaults: Loader default values
Recipe loader applies default values for missing optional fields during TOML parsing.

### REQ-recipes.loader.missing-file: Missing file error
Recipe loader raises FileNotFoundError for non-existent recipe files.

### REQ-recipes.loader.name-from-filename: Name from filename
Recipe loader derives the recipe name from the filename stem when no name is specified in the TOML.

### REQ-recipes.loader.operator-kind: Operator kind detection
Recipe loader correctly identifies operator-kind recipes from the TOML `kind` field.

### REQ-recipes.loader.schedule-implies-operator: Schedule implies operator
Recipe loader automatically sets kind to "operator" when a schedule section is present.

### REQ-recipes.loader.external-prompt: External prompt file loading
Recipe loader supports loading system prompts from external files referenced in the TOML.

### REQ-recipes.loader.legacy-operator: Legacy operator format
Recipe loader supports the legacy operator format for backward compatibility.

### REQ-recipes.loader.eval-fields: Eval field loading
Recipe loader correctly parses eval-specific fields (benchmarks, suites, max_samples, judge).

### REQ-recipes.loader.provider-from-engine: Provider inference from engine
Recipe loader infers the provider from the engine setting when not explicitly specified.

## Discovery (detailed)

### REQ-recipes.discover.builtin: Built-in recipe discovery
Recipe discovery scans the built-in recipes directory for available recipes.

### REQ-recipes.discover.extra-dirs: Extra directory discovery
Recipe discovery scans additional user-specified directories for recipes.

### REQ-recipes.discover.kind-filter: Kind-based filtering
Recipe discovery supports filtering discovered recipes by kind (discrete/operator).

### REQ-recipes.discover.name-override: Name override in discovery
Recipe discovery respects name overrides from the TOML file over filename-derived names.

### REQ-recipes.discover.skip-malformed: Skip malformed recipes
Recipe discovery skips malformed TOML files gracefully and continues scanning.

## Resolution

### REQ-recipes.resolve.found: Successful recipe resolution
`resolve_recipe(name)` finds and returns the named recipe from discovered recipes.

### REQ-recipes.resolve.not-found: Recipe not found
`resolve_recipe(name)` returns None when no recipe matches the given name.

## Builder Kwargs

### REQ-recipes.builder-kwargs.full: Full builder kwargs conversion
`to_builder_kwargs()` converts all recipe fields into agent builder keyword arguments.

### REQ-recipes.builder-kwargs.no-schedule-channels: Builder kwargs without schedule/channels
`to_builder_kwargs()` correctly omits schedule and channel fields when not configured.

### REQ-recipes.builder-kwargs.omit-none: Builder kwargs omit None values
`to_builder_kwargs()` omits keys with None values to allow downstream defaults.

### REQ-recipes.builder-kwargs.prompt-from-path: Builder kwargs prompt from path
`to_builder_kwargs()` resolves prompt file paths into prompt content strings.

## Composers (detailed)

### REQ-recipes.composer.eval-suite-basic: Basic eval suite composition
`recipe_to_eval_suite()` composes a basic eval suite from recipe benchmark and model settings.

### REQ-recipes.composer.eval-suite-benchmark-override: Eval suite benchmark override
Eval suite composition supports overriding benchmark-level settings from the recipe.

### REQ-recipes.composer.eval-suite-max-samples-override: Eval suite max samples override
Eval suite composition respects max_samples overrides from the recipe configuration.

### REQ-recipes.composer.eval-suite-judge-override: Eval suite judge override
Eval suite composition supports overriding the LLM judge model from the recipe.

### REQ-recipes.composer.eval-suite-no-model-error: Eval suite requires model
Eval suite composition raises an error when no model is configured in the recipe.

### REQ-recipes.composer.eval-suite-no-benchmarks-error: Eval suite requires benchmarks
Eval suite composition raises an error when no benchmarks are configured.

### REQ-recipes.composer.eval-suite-suites-fallback: Eval suite suites fallback
Eval suite composition falls back to the suites list when no explicit benchmarks are defined.

### REQ-recipes.composer.eval-suite-direct-backend: Eval suite direct backend
Eval suite composition uses a direct inference backend when no agent is configured.

### REQ-recipes.composer.eval-suite-defaults: Eval suite default config
Eval suite composition applies default configuration values for unspecified settings.

### REQ-recipes.composer.eval-suite-model-temperature: Eval suite model temperature
Eval suite composition inherits the model temperature setting from the recipe.

### REQ-recipes.composer.operator-basic: Basic operator composition
`recipe_to_operator()` composes an operator manifest from recipe schedule and agent settings.

### REQ-recipes.composer.operator-no-schedule-error: Operator requires schedule
Operator composition raises an error when no schedule is defined in the recipe.

### REQ-recipes.composer.operator-defaults: Operator default values
Operator composition applies default values for optional operator fields.

### REQ-recipes.composer.operator-required-capabilities: Operator capabilities
Operator composition includes required capabilities from the recipe tools configuration.

### REQ-recipes.composer.operator-prompt-fields: Operator prompt fields
Operator composition correctly maps recipe prompt settings into the operator manifest.

### REQ-recipes.composer.operator-schedule-value-default: Operator schedule value default
Operator composition provides a default schedule value when only schedule type is specified.

## Tests

- `tests/recipes/test_*.py` - Recipe loading and composition tests
