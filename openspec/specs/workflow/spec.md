# Workflow Module Spec

DAG-based orchestration engine for composing agent pipelines.

## Graph (`graph.py`)

### REQ-workflow.graph.nodes: Node management
`WorkflowGraph.add_node(node: WorkflowNode)` and `add_edge(edge: WorkflowEdge)` build the DAG.

### REQ-workflow.graph.validation: Graph validation
`validate() -> Tuple[bool, str]` checks for cycles, disconnected nodes, missing edges.

### REQ-workflow.graph.ordering: Topological sort
`topological_sort() -> List[str]` returns execution order. `execution_stages() -> List[List[str]]` returns parallel-ready groups.

### REQ-workflow.graph.traversal: Graph traversal
`predecessors(node_id)` and `successors(node_id)` for navigation.

### REQ-workflow.graph.edges: Edge validation
`add_edge()` validates that both source and target nodes exist; raises `ValueError` for missing target nodes.

## Node Types (`types.py`)

### REQ-workflow.types.node-types: Node type enum
Node types: AGENT, TOOL, CONDITION, PARALLEL, LOOP, TRANSFORM.

### REQ-workflow.types.workflow-node: Node definition
`WorkflowNode` with `node_id`, `node_type`, `config`, `inputs`, `outputs`.

### REQ-workflow.types.workflow-result: Execution result
`WorkflowResult` with `outputs`, `steps`, `duration`, `success`.

## Engine (`engine.py`)

### REQ-workflow.engine.run: DAG execution
`WorkflowEngine.run(graph, system, *, initial_input, context) -> WorkflowResult` executes the DAG with stage-based parallelization.

### REQ-workflow.engine.node-dispatch: Node type dispatch
Dispatches to `_run_agent_node`, `_run_tool_node`, `_run_condition_node`, `_run_loop_node`, `_run_transform_node`.

### REQ-workflow.engine.events: Workflow events
Publishes `WORKFLOW_START`, `WORKFLOW_NODE_START`, `WORKFLOW_NODE_END`, `WORKFLOW_END`.

## Builder (`builder.py`)

### REQ-workflow.builder.fluent: Fluent API
`WorkflowBuilder` with chainable methods: `add_agent()`, `add_tool()`, `add_condition()`, `add_loop()`, `add_transform()`, `connect()`, `sequential()`, `build() -> WorkflowGraph`.

## Loader (`loader.py`)

### REQ-workflow.loader.toml: TOML loading
`load_workflow(path) -> WorkflowGraph` loads workflow definitions from TOML files.

## Engine (detailed)

### REQ-workflow.engine.duration: Step and workflow duration tracking
WorkflowEngine records `duration_seconds` on each step result and `total_duration_seconds` on the overall workflow result.

### REQ-workflow.engine.error-handling: Node-level exception handling
WorkflowEngine catches exceptions thrown during node execution and reports them as failure step results.

### REQ-workflow.engine.input-routing: Node input routing
WorkflowEngine routes input to nodes: root nodes receive `initial_input`, downstream nodes receive joined predecessor outputs.

### REQ-workflow.engine.parallel: Parallel stage execution
WorkflowEngine executes independent nodes within the same stage in parallel using `max_parallel` concurrency.

### REQ-workflow.engine.sequential: Sequential pipeline execution
WorkflowEngine executes connected nodes in topological order, stopping the pipeline when a node fails.

### REQ-workflow.engine.validation: Graph validation before execution
WorkflowEngine validates the workflow graph before execution and returns a failure result for invalid (e.g., cyclic) graphs.

## Tests

- `tests/workflow/test_workflow.py` - Existing workflow tests
