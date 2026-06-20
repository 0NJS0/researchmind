from langgraph.graph import StateGraph, END

from src.graph.state import ResearchState

from src.agents.retriever import retrieve_agent
from src.agents.analyzer import analysis_agent
from src.agents.comparison import comparison_agent
from src.agents.gap_detector import gap_agent
from src.agents.report import report_agent

workflow = StateGraph(ResearchState)

workflow.add_node("retriever", retrieve_agent)
workflow.add_node("analysis", analysis_agent)
workflow.add_node("comparison", comparison_agent)
workflow.add_node("gap", gap_agent)
workflow.add_node("report", report_agent)

workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "analysis")

workflow.add_edge("analysis","comparison")

workflow.add_edge("comparison","gap")

workflow.add_edge("gap","report")

workflow.add_edge("report",END)

research_graph=workflow.compile()
