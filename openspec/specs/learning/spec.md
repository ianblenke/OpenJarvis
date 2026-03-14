# Learning Module Spec

Learning system for model routing, agent optimization, and training with trace data.

## Base Abstractions (`_stubs.py`)

### REQ-learning.router-policy: Router policy protocol
`RouterPolicy` abstract class with `select_model(context: RoutingContext) -> str`.

### REQ-learning.reward-function: Reward computation
`RewardFunction` abstract class with `compute(context, model_key, response, **kwargs) -> float` returning [0, 1].

### REQ-learning.learning-policy: Learning policy protocol
`LearningPolicy` abstract class with `target: str` ("intelligence"|"agent") and `update(trace_store, **kwargs) -> Dict[str, Any]`.

### REQ-learning.query-analyzer: Query analysis
`QueryAnalyzer` abstract class with `analyze(query, **kwargs) -> RoutingContext`.

### REQ-learning.registration: Registry-based registration
Learning policies use `@LearningRegistry.register("name")` decorator.

## Optimization Engine

### REQ-learning.optimizer-engine: Trial-based optimization
Optimizer engine runs trials with configurable search spaces, stores results, supports early stopping and Pareto frontier analysis.

### REQ-learning.trial-runner: Trial execution
Trial runner executes optimization trials with configurable parameters and collects metrics.

### REQ-learning.feedback-collector: Feedback collection
Collects user feedback on traces and stores for training.

### REQ-learning.trace-judge: Automated trace judging
Uses LLM to judge trace quality for reward signal.

## Training

### REQ-learning.sft-trainer: Supervised fine-tuning
SFT trainer for model fine-tuning on collected traces.

### REQ-learning.grpo-trainer: Group relative policy optimization
GRPO trainer for policy optimization.

### REQ-learning.dspy-optimizer: DSPy prompt optimization
DSPy-based agent prompt optimization.

### REQ-learning.gepa-optimizer: GEPA optimization
Genetic evolutionary prompt optimization.

## Routing

### REQ-learning.learned-router: Learned routing policy
Router that learns from trace data to select optimal models.

### REQ-learning.heuristic-router: Heuristic routing
Rule-based routing using query analysis features.

## Tests

- `tests/learning/test_*.py` - Learning subsystem tests
- `tests/learning/intelligence/test_*.py` - Intelligence learning tests
- `tests/learning/routing/test_*.py` - Routing tests
- `tests/learning/agents/test_*.py` - Agent learning tests
