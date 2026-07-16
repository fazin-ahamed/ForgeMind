---
name: multi-module-feature-expansion
description: Workflow command scaffold for multi-module-feature-expansion in ForgeMind.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /multi-module-feature-expansion

Use this workflow when working on **multi-module-feature-expansion** in `ForgeMind`.

## Goal

Expands or introduces a feature that requires coordinated changes across several core modules and their tests.

## Common Files

- `src/forgemind/*.py`
- `benchmarks/*.py`
- `tests/test_*.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit multiple implementation files in src/forgemind/ and/or benchmarks/
- Edit or add multiple test files in tests/
- Commit all related changes together

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.