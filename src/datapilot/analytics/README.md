# DataPilot Analytics

The `analytics` package provides deterministic analysis of executed query results, grounded evidence construction, visualization specifications, and business-facing answer composition. It separates analytical computation from language generation: deterministic components calculate and validate results; language-generation components explain supported findings.

## Responsibilities

1. Summarize query results and compute deterministic analytical outputs.
2. Validate proposed analytical work against available columns, data types, dependencies, and trusted semantic metadata.
3. Build evidence from computed results and visualization specifications.
4. Compose a business-friendly answer from that evidence.

## Main components

| Module | Responsibility |
|---|---|
| `analyzer.py` | General-purpose numeric and categorical summaries of query results. |
| `planning.py` | Immutable contracts for analysis requests, input mappings, plans, validation issues, and validation outcomes. |
| `plan_validator.py` | Deterministic validation, rejection/skipping, dependency propagation, and stable execution ordering. |
| `trend.py` | Trend analysis over a period column and numeric measure. |
| `period_comparison.py` | Scalar comparison of two already-derived period values. |
| `variance.py` | Scalar actual-versus-reference variance calculation. |
| `contribution.py` | Analysis of category-level changes, with trusted semantic meaning required by the planner. |
| `segmentation.py` | Breakdown of a numeric measure by a categorical dimension. |
| `root_cause.py` | Reconciliation of a variance result with a contribution result; this is arithmetic driver reconciliation, not proof of causality. |
| `visualization.py` | Selection/specification of a visualization from available analytical results. |
| `evidence.py` | Construction of grounded evidence from analytical results and visualization specifications. |
| `answer.py` | Business-facing answer composition from evidence. |

## Analytics planning contract

The analytics planner is intended to operate after SQL execution, when a query-result DataFrame is available. The LLM may propose an `AnalyticsPlan`; the deterministic `AnalyticsPlanValidator` decides which requests are safe to accept. A proposal is not trusted merely because it is syntactically valid.

The plan contract uses immutable dataclasses:

- `AnalysisType` identifies the requested deterministic operation.
- `AnalysisInputMapping` maps source columns and period labels.
- `AnalysisRequest` identifies one operation and its dependencies.
- `AnalyticsPlan` groups requests for the user's analytical question.
- `AnalyticsPlanValidation` reports accepted requests, skipped requests, and fatal plan-level errors.

### Aggregation semantics

`AggregationType` supports `SUM`, `MEAN`, `MIN`, `MAX`, and `COUNT`. Period comparison and variance requests must explicitly select one of these aggregations. There is no implicit `SUM` fallback. `COUNT` means counting non-null values in the mapped measure column. The other four operations require a numeric measure.

Aggregation metadata is only valid for period comparison and variance requests. Trend, contribution, segmentation, and root-cause requests do not accept it. The validator checks the requested aggregation and its input compatibility; it does not execute the aggregation. Execution belongs to a later deterministic analysis-execution layer.

### Validation behavior

The validator distinguishes two outcomes:

- **Fatal plan errors** — duplicate request IDs, missing or duplicated dependencies, dependency cycles, or an empty query result. A fatal error prevents execution of the plan.
- **Skipped requests** — an individual request has invalid mappings, unavailable columns or periods, incompatible data types, missing required aggregation, untrusted contribution metadata, or incompatible operation dependencies. Independent valid requests may still be accepted.

Accepted requests are returned in stable topological order so dependencies execute before their consumers. If a dependency is skipped, dependent requests are skipped as well.

Contribution analysis requires trusted metadata identifying columns that represent category-level changes. This metadata must be supplied by the application, not asserted by the LLM plan. Root-cause analysis requires exactly one variance result and one contribution result. The root-cause analyzer checks whether the contribution total reconciles to the overall variance; it does not establish causal relationships.

## Current integration boundary

The planning contract and validator are implemented as standalone analytics components. They do not yet add planning or request-execution nodes to the LangGraph agent workflow. The existing result-analysis, visualization, evidence, and answer stages remain separate until the planning and execution contracts are tested and integrated in a later stage.

## Design principles

- Deterministic calculations are preferred over LLM-generated numerical claims.
- LLM proposals are validated against the actual query result and trusted application metadata.
- Unsupported operations are skipped explicitly; missing aggregation is never silently defaulted.
- Independent valid work is preserved when another request can safely be skipped.
- Dependency ordering and failure propagation are deterministic.
- Evidence and final explanations must remain grounded in computed results.