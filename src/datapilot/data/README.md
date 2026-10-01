# Data Layer

The `data` package contains the components responsible for discovering, describing, and accessing the data available to DataPilot.

This layer provides the interface between the agent pipeline and the underlying database/data source.

## Responsibilities

The data layer is responsible for:

- Inspecting available database tables.
- Reading table and column metadata.
- Representing schema information using typed data models.
- Providing access to the underlying DuckDB data source.
- Supplying schema information to the SQL generation and agent layers.
- Validating whether requested tables exist before describing them.

## Components

### `schema.py`

Provides schema inspection and metadata models.

Key components include:

- `ColumnInfo` — describes a database column.
- `TableInfo` — describes a database table and its columns.
- `SchemaInspector` — inspects the available DuckDB schema.

`SchemaInspector` can:

1. List user-visible tables.
2. Describe an individual table.
3. Inspect all available tables.

The inspector reads metadata from DuckDB's `information_schema`.

### `duckdb.py`

Contains the DuckDB data-source implementation used by the application.

It provides the database access layer used by components such as `SchemaInspector`.

## Data Flow

```text
DuckDB Data Source
        │
        ▼
   SchemaInspector
        │
        ├── TableInfo
        │      └── ColumnInfo
        │
        ▼
   Agent / SQL Layer