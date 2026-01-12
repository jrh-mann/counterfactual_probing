# Production Readiness Plan

## Current State: 70% Production Ready

### Critical Gaps
1. No dependency pinning (breaking changes risk)
2. Legacy code mixed with new code
3. No checkpointing/recovery for long runs
4. Missing integration tests
5. No structured logging
6. Documentation gaps

---

## Phase 1: Foundation (Priority: Critical)

### 1.1 Dependency Management
- [ ] Pin all dependencies in `pyproject.toml` with exact versions
- [ ] Generate `uv.lock` or `requirements.lock`
- [ ] Add dependency groups: `core`, `dev`, `viz`, `test`
- [ ] Document Python version requirements (3.10+)

**Files to modify:**
- `pyproject.toml`

**Example:**
```toml
[project]
dependencies = [
    "vllm==0.8.5",
    "nnsight==0.3.7",
    "transformers==4.46.0",
    "pydantic>=2.0,<3.0",
    "torch>=2.0,<3.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4", "mypy>=1.10"]
viz = ["plotly>=5.0", "scikit-learn>=1.4"]
```

### 1.2 Legacy Code Cleanup
- [ ] Audit `src/rollouts/` - identify what's used vs dead
- [ ] Move any live code to appropriate modules
- [ ] Delete unused files:
  - `src/rollouts/generate.py` (superseded by `run.py`)
  - `src/utils/api_client.py` (references old models)
- [ ] Update imports across codebase

**Files to audit:**
- `src/rollouts/*.py`
- `src/utils/api_client.py`

### 1.3 Configuration Hardening
- [ ] Add environment variable overrides for sensitive settings
- [ ] Add config schema versioning
- [ ] Validate paths exist before running
- [ ] Add `--dry-run` mode to CLI

**Add to `config.py`:**
```python
import os

class ModelConfig(BaseModel):
    name: str = Field(
        default_factory=lambda: os.getenv("CFPROBE_MODEL", "Qwen/Qwen2.5-0.5B")
    )
```

---

## Phase 2: Reliability (Priority: High)

### 2.1 Checkpointing & Recovery
- [ ] Save progress after each prompt (already partial via `skip_existing`)
- [ ] Add `checkpoint.json` with run metadata
- [ ] Implement `--resume` flag for interrupted runs
- [ ] Add graceful shutdown handler (SIGINT/SIGTERM)

**New file:** `src/counterfactual_probing/checkpoint.py`
```python
@dataclass
class RunCheckpoint:
    run_id: str
    config_hash: str
    completed_prompts: List[str]
    failed_prompts: List[str]
    start_time: datetime
    last_update: datetime
```

### 2.2 Structured Logging
- [ ] Replace `print()` with `logging` module
- [ ] Add log levels: DEBUG, INFO, WARNING, ERROR
- [ ] Add structured JSON logging option
- [ ] Include run_id in all log messages
- [ ] Add `--verbose` / `--quiet` flags to CLI

**New file:** `src/counterfactual_probing/logging_config.py`

### 2.3 Error Handling
- [ ] Define custom exception hierarchy
- [ ] Add retry logic for transient failures (GPU OOM, network)
- [ ] Collect and report errors at end of run
- [ ] Add `--fail-fast` vs `--continue-on-error` modes

**New file:** `src/counterfactual_probing/exceptions.py`
```python
class CounterfactualProbingError(Exception):
    """Base exception for all cfprobe errors."""

class ConfigurationError(CounterfactualProbingError):
    """Invalid configuration."""

class ExtractionError(CounterfactualProbingError):
    """Activation extraction failed."""

class ScorerError(CounterfactualProbingError):
    """Scorer execution failed."""
```

---

## Phase 3: Testing & CI (Priority: High)

### 3.1 Test Coverage
- [ ] Add integration tests for full pipeline (use tiny model)
- [ ] Add tests for model_utils.py
- [ ] Add tests for checkpoint/recovery
- [ ] Add property-based tests for config validation
- [ ] Target: 80% coverage

**New test files:**
- `tests/test_model_utils.py`
- `tests/test_integration.py`
- `tests/test_checkpoint.py`

### 3.2 CI/CD Pipeline
- [ ] Add GitHub Actions workflow
- [ ] Run tests on PR
- [ ] Run linting (ruff)
- [ ] Run type checking (mypy)
- [ ] Build and test Docker image

**New file:** `.github/workflows/ci.yml`
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --all-extras
      - run: uv run pytest
      - run: uv run ruff check
      - run: uv run mypy src/
```

### 3.3 Code Quality
- [ ] Add `ruff.toml` for linting config
- [ ] Add `mypy.ini` for type checking
- [ ] Add pre-commit hooks
- [ ] Fix all existing lint/type errors

**New files:**
- `ruff.toml`
- `mypy.ini`
- `.pre-commit-config.yaml`

---

## Phase 4: Documentation (Priority: Medium)

### 4.1 API Documentation
- [ ] Add docstrings to all public functions
- [ ] Generate API docs with mkdocs/sphinx
- [ ] Document all config options with examples
- [ ] Add architecture diagram

### 4.2 User Guides
- [ ] Quick start guide
- [ ] Custom scorer tutorial
- [ ] Activation extraction deep dive
- [ ] Troubleshooting guide

### 4.3 README Updates
- [ ] Add badges (CI, coverage, version)
- [ ] Add installation instructions
- [ ] Add common use cases
- [ ] Add performance benchmarks

**New structure:**
```
docs/
├── index.md
├── quickstart.md
├── configuration.md
├── custom-scorers.md
├── api/
│   ├── config.md
│   ├── run.md
│   └── activations.md
└── troubleshooting.md
```

---

## Phase 5: Performance & Scalability (Priority: Medium)

### 5.1 Memory Optimization
- [ ] Add activation streaming (don't hold all in memory)
- [ ] Implement batch size auto-tuning
- [ ] Add memory profiling hooks
- [ ] Document memory requirements per model size

### 5.2 Parallelization
- [ ] Support multi-GPU for activation extraction
- [ ] Add `--num-workers` for parallel prompt processing
- [ ] Implement distributed mode (Ray/multiprocessing)

### 5.3 Caching
- [ ] Cache tokenized prompts
- [ ] Cache model config/tokenizer loading
- [ ] Add LRU cache for repeated scorer calls

---

## Phase 6: Deployment (Priority: Low initially)

### 6.1 Containerization
- [ ] Create optimized Dockerfile
- [ ] Add docker-compose for full stack
- [ ] Document GPU passthrough setup

**New file:** `Dockerfile`
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
RUN pip install uv
COPY . /app
WORKDIR /app
RUN uv sync
ENTRYPOINT ["uv", "run", "cfprobe"]
```

### 6.2 Package Distribution
- [ ] Publish to PyPI
- [ ] Add versioning strategy (semver)
- [ ] Create release automation

### 6.3 Monitoring
- [ ] Add Prometheus metrics endpoint
- [ ] Track: prompts/sec, GPU utilization, errors
- [ ] Add health check endpoint

---

## Implementation Order

### Sprint 1 (Week 1-2): Foundation
1. Pin dependencies + lockfile
2. Delete legacy code
3. Add structured logging
4. Add custom exceptions

### Sprint 2 (Week 3-4): Reliability
1. Implement checkpointing
2. Add retry logic
3. Add integration tests
4. Set up CI pipeline

### Sprint 3 (Week 5-6): Quality
1. Fix all lint/type errors
2. Add pre-commit hooks
3. Achieve 80% test coverage
4. Write API documentation

### Sprint 4 (Week 7-8): Polish
1. Write user guides
2. Performance optimization
3. Dockerfile
4. PyPI release

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Test coverage | ~40% | 80% |
| Type coverage | 0% | 90% |
| Lint errors | Unknown | 0 |
| Doc coverage | 30% | 90% |
| CI passing | No CI | 100% |
| Recovery from crash | Manual | Automatic |

---

## Quick Wins (Do Today)

1. **Pin vllm version** - Most likely to break
2. **Delete `src/rollouts/generate.py`** - Clearly superseded
3. **Add `tests/test_model_utils.py`** - New code, easy to test
4. **Add `.gitignore` entries** - `*.pt`, `outputs/`, `activations/`

---

## Files to Create

```
.github/workflows/ci.yml
src/counterfactual_probing/checkpoint.py
src/counterfactual_probing/exceptions.py
src/counterfactual_probing/logging_config.py
tests/test_model_utils.py
tests/test_integration.py
docs/
ruff.toml
mypy.ini
.pre-commit-config.yaml
Dockerfile
```

## Files to Delete

```
src/rollouts/generate.py (after audit)
src/utils/api_client.py (after audit)
```
