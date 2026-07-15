```markdown
# ForgeMind Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns, coding conventions, and workflows used in the ForgeMind Python codebase. ForgeMind is organized for modularity and testability, with a focus on clear commit practices, structured file organization, and robust benchmarking support. The repository emphasizes coordinated changes between implementation and tests, especially when introducing new features, fixing bugs, or updating benchmarking pipelines.

## Coding Conventions

- **Language:** Python
- **Framework:** None detected
- **File Naming:** Use `snake_case` for all file and module names.
  - Example: `my_module.py`, `test_runtime.py`
- **Import Style:** Prefer **relative imports** within the package.
  - Example:
    ```python
    from .utils import helper_function
    ```
- **Export Style:** Use **named exports** (explicitly define what is exported).
  - Example:
    ```python
    __all__ = ["MyClass", "my_function"]
    ```
- **Commit Messages:** Follow [Conventional Commits](https://www.conventionalcommits.org/) with prefixes like `feat`, `fix`, `docs`.
  - Example: `feat: add support for custom context managers`

## Workflows

### Feature or Fix with Implementation and Test
**Trigger:** When adding a new feature or fixing a bug and ensuring it is tested  
**Command:** `/feature-with-tests`

1. Edit or add implementation file(s) in `src/forgemind/` or `benchmarks/`.
2. Edit or add corresponding test file(s) in `tests/`.
3. Commit both implementation and test changes together.

**Example:**
```python
# src/forgemind/new_feature.py
def new_feature():
    return "Hello, ForgeMind!"

# tests/test_new_feature.py
from src.forgemind.new_feature import new_feature

def test_new_feature():
    assert new_feature() == "Hello, ForgeMind!"
```

---

### Multi-Module Feature Expansion
**Trigger:** When implementing a cross-cutting feature affecting multiple modules and their tests  
**Command:** `/multi-module-feature`

1. Edit multiple implementation files in `src/forgemind/` and/or `benchmarks/`.
2. Edit or add multiple test files in `tests/`.
3. Commit all related changes together.

**Example:**
- Update `src/forgemind/module_a.py` and `src/forgemind/module_b.py`
- Add/modify `tests/test_module_a.py` and `tests/test_module_b.py`

---

### Benchmark Pipeline Update
**Trigger:** When adding, modifying, or fixing benchmarking workflows or data pipelines  
**Command:** `/update-benchmark-pipeline`

1. Edit or add files in `benchmarks/` (e.g., `build_forgebench.py`, `import_external.py`).
2. Edit or add related test files in `tests/` (e.g., `test_forgebench_builder.py`, `test_external_import.py`).
3. Optionally update documentation in `benchmarks/README.md`.

**Example:**
```python
# benchmarks/build_forgebench.py
def build_bench():
    # Benchmark logic here

# tests/test_forgebench_builder.py
from benchmarks.build_forgebench import build_bench

def test_build_bench():
    assert build_bench() is not None
```

---

### Config or Context Update with Tests
**Trigger:** When updating configuration logic or context handling and validating with tests  
**Command:** `/update-config-context`

1. Edit `src/forgemind/config.py` and/or `src/forgemind/context.py`.
2. Edit or add related test files in `tests/` (e.g., `test_context.py`, `test_runtime.py`).
3. Commit both implementation and test changes together.

**Example:**
```python
# src/forgemind/config.py
def get_config():
    return {"setting": True}

# tests/test_context.py
from src.forgemind.config import get_config

def test_get_config():
    assert get_config()["setting"] is True
```

## Testing Patterns

- **Test Framework:** Not explicitly detected; tests follow standard Python patterns.
- **Test File Naming:** All test files are named using the pattern `test_*.py` and located in the `tests/` directory.
- **Test Example:**
    ```python
    # tests/test_example.py
    from src.forgemind.example import some_function

    def test_some_function():
        assert some_function() == expected_value
    ```
- **Benchmark Tests:** Specialized tests for benchmarking scripts are named accordingly (e.g., `test_forgebench_builder.py`).

## Commands

| Command                   | Purpose                                                      |
|---------------------------|--------------------------------------------------------------|
| /feature-with-tests       | Add a new feature or fix a bug with corresponding tests      |
| /multi-module-feature     | Expand or introduce a feature across multiple modules/tests  |
| /update-benchmark-pipeline| Update or extend the benchmarking pipeline and its tests     |
| /update-config-context    | Update configuration/context logic and related tests         |
```
