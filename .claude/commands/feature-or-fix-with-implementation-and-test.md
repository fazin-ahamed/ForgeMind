---
name: feature-or-fix-with-implementation-and-test
description: Workflow command scaffold for feature-or-fix-with-implementation-and-test in ForgeMind.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /feature-or-fix-with-implementation-and-test

Use this workflow when working on **feature-or-fix-with-implementation-and-test** in `ForgeMind`.

## Goal

Implements a new feature or fixes a bug by updating implementation files and corresponding test files together.

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

- Edit or add implementation file(s) in src/forgemind/ or benchmarks/
- Edit or add corresponding test file(s) in tests/
- Commit both implementation and test changes together

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.