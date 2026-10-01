# Utilities

The `utils` package contains shared utility functionality used across DataPilot.

Utilities should contain small, reusable helpers that support the application without owning domain-specific business logic or agent orchestration.

## Responsibilities

The utilities layer is intended for:

- Common helper functionality.
- Shared transformations and convenience functions.
- Reusable functionality needed by multiple DataPilot components.
- Keeping generic helper logic out of domain-specific packages.

## Design Principles

### Reusability

Utilities should solve common problems that are useful across multiple parts of the application.

### Small and Focused

Each utility should have a clear responsibility rather than becoming a collection of unrelated application logic.

### Minimal Coupling

Utility functions should avoid unnecessary dependencies on higher-level components such as the agent graph or runtime.

### No Business Ownership

Business rules should remain in the appropriate domain layer.

For example:

- SQL behavior belongs in `sql`.
- Schema handling belongs in `data`.
- Evidence generation belongs in `analytics`.
- Agent orchestration belongs in `agents`.
- Visualization behavior belongs in `visualization`.

The utilities layer should provide supporting functionality rather than taking ownership of these responsibilities.

## Role in DataPilot

The utilities package sits below the higher-level application components and provides reusable functionality where appropriate.

```text
              DataPilot Components
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      Agents        Analytics        SQL
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
                    Utils