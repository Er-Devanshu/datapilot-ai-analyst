
from __future__ import annotations

import json

import pandas as pd

from datapilot.agents.graph import build_datapilot_graph
from datapilot.analytics.analyzer import ResultAnalyzer
from datapilot.analytics.executor import AnalyticsExecutor
from datapilot.analytics.planner import AnalyticsPlanner
from datapilot.agents.nodes import evidence_node
from datapilot.agents.state import DataPilotState
from datapilot.analytics.variance import VarianceResult


class FakeLLM:
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def variance_response() -> str:
    return json.dumps(
        {
            "requests": [
                {
                    "request_id": "june_july_variance",
                    "analysis_type": "variance",
                    "inputs": {
                        "period_column": "month",
                        "value_column": "revenue",
                        "reference_period": "June",
                        "actual_period": "July",
                        "aggregation": "sum",
                    },
                    "depends_on": [],
                }
            ]
        }
    )


def test_graph_contains_analytics_pipeline_nodes() -> None:
    graph = build_datapilot_graph(
        llm=FakeLLM(),
        inspector=object(),
        executor=object(),
    )

    assert "analyze_result" in graph.nodes
    assert "plan_analytics" in graph.nodes
    assert "execute_analytics" in graph.nodes
    assert "select_visualization" in graph.nodes


def test_graph_rejects_invalid_trusted_change_column_configuration() -> None:
    try:
        build_datapilot_graph(
            llm=FakeLLM(),
            inspector=object(),
            executor=object(),
            change_value_columns=["revenue_change"],
        )
    except ValueError as exc:
        assert "set or frozenset" in str(exc)
    else:
        raise AssertionError("Invalid trusted-column configuration was accepted.")


def test_evidence_node_includes_all_analytics_executions() -> None:
    dataframe = pd.DataFrame(
        {
            "month": ["June", "June", "July", "July"],
            "revenue": [60.0, 40.0, 50.0, 35.0],
        }
    )

    llm = FakeLLM(variance_response())
    planning = AnalyticsPlanner(llm).plan(
        question="Explain the revenue variance from June to July.",
        dataframe=dataframe,
    )
    execution = AnalyticsExecutor().execute(
        dataframe=dataframe,
        validation=planning.validation,
    )

    state: DataPilotState = {
        "question": "Explain the revenue variance from June to July.",
        "result": dataframe,
        "analysis": ResultAnalyzer().analyze(dataframe),
        "analytics_plan": planning.plan,
        "analytics_plan_validation": planning.validation,
        "analytics_planning_result": planning,
        "analytics_execution": execution,
    }

    result = evidence_node(state)
    evidence = result["evidence"]

    assert len(evidence.analytics_executions) == 1
    assert evidence.analytics_executions[0].request_id == "june_july_variance"
    assert isinstance(evidence.analytics_executions[0].result, VarianceResult)

    prompt_text = evidence.to_prompt_text()

    assert "ANALYTICS EXECUTION RESULTS:" in prompt_text
    assert "june_july_variance" in prompt_text
    assert "variance" in prompt_text
    assert "Result type: VarianceResult" in prompt_text
    assert "Absolute variance: -15.0" in prompt_text
    assert "Percentage variance: -15.0" in prompt_text
    assert "Direction: below_reference" in prompt_text
    assert result["status"] == "evidence_ready"


def test_evidence_node_handles_missing_analytics_execution() -> None:
    dataframe = pd.DataFrame({"revenue": [100.0, 200.0]})

    state: DataPilotState = {
        "question": "Summarize revenue.",
        "result": dataframe,
        "analysis": ResultAnalyzer().analyze(dataframe),
    }

    result = evidence_node(state)

    assert result["evidence"].analytics_executions == ()
