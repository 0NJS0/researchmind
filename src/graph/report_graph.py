from langgraph.graph import StateGraph, END

from src.graph.state import ResearchState

from src.agents.retriever import retrieve_agent
from src.agents.classifier import classifier_agent
from src.agents.batch_analyzer import batch_analyzer_agent
from src.agents.comparison import comparison_agent
from src.agents.gap_detector import gap_agent
from src.agents.report import report_agent

workflow = StateGraph(ResearchState)

workflow.add_node("retriever", retrieve_agent)
workflow.add_node("classifier", classifier_agent)
workflow.add_node("batch_analyzer", batch_analyzer_agent)
workflow.add_node("comparison", comparison_agent)
workflow.add_node("gap", gap_agent)
workflow.add_node("report", report_agent)

workflow.set_entry_point("retriever")
workflow.add_edge("retriever", "classifier")
workflow.add_edge("classifier", "batch_analyzer")
workflow.add_edge("batch_analyzer", "comparison")
workflow.add_edge("comparison", "gap")
workflow.add_edge("gap", "report")
workflow.add_edge("report", END)

report_graph = workflow.compile()
