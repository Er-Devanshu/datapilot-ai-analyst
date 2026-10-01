# Tests

The `tests` directory contains the automated test suite for DataPilot.

The test suite validates the behavior of the agent pipeline, individual components, integrations, and public runtime interfaces.

## Purpose

The tests are designed to ensure that DataPilot remains:

- Deterministic where deterministic behavior is expected
- Correct across the complete agent pipeline
- Safe when handling invalid inputs and failures
- Stable as individual components evolve
- Compatible with the public runtime API

## Test Structure

```text
tests/
├── test_*.py
└── README.md