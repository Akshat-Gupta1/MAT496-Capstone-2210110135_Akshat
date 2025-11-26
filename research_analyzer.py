#!/usr/bin/env python3
"""
Research Analyzer - A LangGraph-based multi-analyst news research system.

This script creates a team of specialized news analysts who conduct research
on a given topic using web search and synthesize their findings into a
comprehensive report.

Requirements:
    pip install langgraph langchain_openai langchain_community langchain_core tavily-python python-dotenv

Usage:
    python research_analyzer.py [--topic "Your Topic"] [--max-analysts 3]
"""

import os
import operator
import argparse
from typing import List, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# Data Models
# ============================================================================

class NewsAnalyst(BaseModel):
    """Model representing a specialized news analyst."""
    affiliation: str = Field(
        description="Primary affiliation of the analyst (e.g., News Network, Think Tank, Research Institute).",
    )
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst (e.g., Political Correspondent, Tech Reporter, Economic Analyst).",
    )
    description: str = Field(
        description="Description of the analyst's focus, expertise, and analytical approach.",
    )
    
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class AnalystTeam(BaseModel):
    """Model representing a team of news analysts."""
    analysts: List[NewsAnalyst] = Field(
        description="Team of news analysts with diverse specializations.",
    )


class SearchQuery(BaseModel):
    """Model for search query generation."""
    search_query: str = Field(None, description="Search query for news retrieval.")


# ============================================================================
# State Definitions
# ============================================================================

class GenerateAnalystsState(TypedDict):
    """State for the analyst generation subgraph."""
    topic: str  # Topic to analyze
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[NewsAnalyst]  # Generated analysts


class AnalysisState(MessagesState):
    """State for individual analyst research."""
    max_num_turns: int  # Number turns of conversation
    context: Annotated[list, operator.add]  # Source docs
    analyst: NewsAnalyst  # Analyst conducting analysis
    analysis: str  # Analysis transcript
    sections: Annotated[list, operator.add]  # Final sections for report


class ResearchGraphState(TypedDict):
    """State for the main research graph."""
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[NewsAnalyst]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


# ============================================================================
# Instructions Templates
# ============================================================================

ANALYST_INSTRUCTIONS = """You are tasked with creating a team of specialized news analysts. Follow these instructions:

1. Review the topic:
{topic}

2. Examine any editorial feedback for analyst creation:

{human_analyst_feedback}

3. Determine the most important perspectives for comprehensive news analysis of this topic.

4. Pick the top {max_analysts} perspectives.

5. Assign one analyst to each perspective with relevant expertise:
   - Examples: Political Analysis, Economic Impact, Social/Cultural Effects, Scientific/Tech Angle, International Relations, etc."""


QUESTION_INSTRUCTIONS = """You are a news analyst conducting research on {topic}.

Your goal is to gather specific, actionable insights about the topic.

1. Insightful: Find information that provides depth and context.

2. Specific: Include concrete data, quotes, names, and recent developments.

Your analytical focus: {goals}

Begin by introducing yourself, then pose your analytical questions.

Continue to drill down until you have comprehensive insights.

When satisfied with your analysis, conclude with: "Analysis complete!"

Stay in character throughout your response."""


SEARCH_INSTRUCTIONS = """Generate a search query for recent news and information.

Focus on the latest developments, news, and data points relevant to the conversation.

The query should be specific and targeted for news sources."""


ANSWER_INSTRUCTIONS = """You are a news information expert.

Analyst focus: {goals}

Answer the analyst's question using this context:

{context}

Guidelines:

1. Use only information from the provided context.

2. Include specific numbers, dates, and data points.

3. Cite sources using [1], [2], etc.

4. List sources at the end:

[1] Source 1
[2] Source 2"""


SECTION_WRITER_INSTRUCTIONS = """You are a news report writer.

Create a concise, professional section based on analyst research.

1. Analyze the source documents containing analyst research.

2. Use this structure:
## {focus} (section title)

### Key Findings

### Analysis

### Sources

3. Analyst focus: {focus}

4. Key Findings:
- Highlight specific data points and numbers
- Include dates and timeframes
- Note significant trends or changes

5. Analysis:
- Explain implications
- Connect findings to the broader topic
- Maximum 300 words
- Use numbered sources [1], [2]

6. Sources:
- Include all sources used
- Format: [1] URL or source name
- One per line
"""


REPORT_WRITER_INSTRUCTIONS = """You are creating a comprehensive news report on:

{topic}

Your analyst team has completed their research.

Task:

1. Review all analyst sections
2. Identify key insights and themes
3. Synthesize into a cohesive narrative
4. Highlight actionable conclusions or future implications

Format:

1. Use markdown
2. No preamble
3. No sub-headings
4. Start with: ## News Analysis
5. Preserve all citations [1], [2], etc.
6. Create consolidated Sources section:

## Sources
[1] Source 1
[2] Source 2

Analyst sections:

{context}"""


INTRODUCTION_INSTRUCTIONS = """You are writing an investment analysis report on {topic}

You have all report sections available.

Write a compelling introduction section.

No preamble.

Target 100 words.

Use markdown.

Create a compelling title with # header, then ## Introduction section.

Preview the key areas of analysis covered in the report.

Report sections: {formatted_str_sections}"""


CONCLUSION_INSTRUCTIONS = """You are writing an investment analysis report on {topic}

You have all report sections available.

Write a conclusion section.

No preamble.

Target 100 words.

Use markdown.

Use ## Conclusion header.

Summarize the key investment insights and provide an overall perspective.

Report sections: {formatted_str_sections}"""


# ============================================================================
# Initialize LLM and Tools
# ============================================================================

def get_llm():
    """Initialize the language model."""
    return ChatOpenAI(model="gpt-4o", temperature=0)


def get_search_tool():
    """Initialize the Tavily search tool."""
    return TavilySearchResults(max_results=3)


# ============================================================================
# Node Functions - Analyst Generation
# ============================================================================

def create_analysts(state: GenerateAnalystsState):
    """Create news analysts based on the topic."""
    llm = get_llm()
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')
    
    structured_llm = llm.with_structured_output(AnalystTeam)
    
    system_message = ANALYST_INSTRUCTIONS.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts
    )
    
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)] + 
        [HumanMessage(content="Generate the analyst team.")]
    )
    
    return {"analysts": analysts.analysts}


def human_feedback(state: GenerateAnalystsState):
    """No-op node for interruption (human-in-the-loop)."""
    pass


def should_continue(state: GenerateAnalystsState):
    """Return the next node to execute based on feedback."""
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    return END


# ============================================================================
# Node Functions - Analysis
# ============================================================================

def generate_question(state: AnalysisState):
    """Generate analysis question from the analyst."""
    llm = get_llm()
    analyst = state["analyst"]
    messages = state["messages"]
    
    # Extract topic from the first message (HumanMessage contains "Analyze {topic} from your perspective.")
    first_msg = messages[0].content if messages else "the topic"
    # Extract topic from the message format
    topic = first_msg.replace("Analyze ", "").replace(" from your perspective.", "") if "Analyze " in first_msg else "the topic"
    
    system_message = QUESTION_INSTRUCTIONS.format(
        topic=topic,
        goals=analyst.persona
    )
    question = llm.invoke([SystemMessage(content=system_message)] + messages)
    
    return {"messages": [question]}


def search_news(state: AnalysisState):
    """Search for news related to the analysis."""
    llm = get_llm()
    tavily_search = get_search_tool()
    
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke(
        [SystemMessage(content=SEARCH_INSTRUCTIONS)] + state['messages']
    )
    
    try:
        search_docs = tavily_search.invoke(search_query.search_query)
    except Exception:
        search_docs = []
    
    if not search_docs:
        formatted_search_docs = ""
    else:
        formatted_search_docs = "\n\n---\n\n".join([
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ])
    
    return {"context": [formatted_search_docs]}


def generate_answer(state: AnalysisState):
    """Generate answer based on search results."""
    llm = get_llm()
    analyst = state["analyst"]
    messages = state["messages"]
    context = state.get("context", [])
    
    system_message = ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)
    
    answer.name = "expert"
    
    return {"messages": [answer]}


def save_analysis(state: AnalysisState):
    """Save the analysis transcript."""
    messages = state["messages"]
    analysis = get_buffer_string(messages)
    return {"analysis": analysis}


def route_messages(state: AnalysisState, name: str = "expert"):
    """Route between question and answer."""
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)
    
    num_responses = len([
        m for m in messages if isinstance(m, AIMessage) and m.name == name
    ])
    
    if num_responses >= max_num_turns:
        return 'save_analysis'
    
    last_question = messages[-2]
    
    if "Analysis complete" in last_question.content:
        return 'save_analysis'
    return "ask_question"


def write_section(state: AnalysisState):
    """Write report section based on analysis."""
    llm = get_llm()
    context = state.get("context", [])
    analyst = state["analyst"]
    
    system_message = SECTION_WRITER_INSTRUCTIONS.format(focus=analyst.description)
    section = llm.invoke(
        [SystemMessage(content=system_message)] + 
        [HumanMessage(content=f"Use this research: {context}")]
    )
    
    return {"sections": [section.content]}


# ============================================================================
# Node Functions - Report Writing
# ============================================================================

def initiate_all_analyses(state: ResearchGraphState):
    """Map step: run each analysis using Send API."""
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        return "create_analysts"
    
    topic = state["topic"]
    return [
        Send("conduct_analysis", {
            "analyst": analyst,
            "messages": [HumanMessage(
                content=f"Analyze {topic} from your perspective."
            )]
        }) 
        for analyst in state["analysts"]
    ]


def write_report(state: ResearchGraphState):
    """Write the main report body."""
    llm = get_llm()
    sections = state["sections"]
    topic = state["topic"]
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    system_message = REPORT_WRITER_INSTRUCTIONS.format(
        topic=topic, 
        context=formatted_str_sections
    )
    report = llm.invoke(
        [SystemMessage(content=system_message)] + 
        [HumanMessage(content="Write the news analysis report.")]
    )
    return {"content": report.content}


def write_introduction(state: ResearchGraphState):
    """Write the report introduction."""
    llm = get_llm()
    sections = state["sections"]
    topic = state["topic"]
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    instructions = INTRODUCTION_INSTRUCTIONS.format(
        topic=topic,
        formatted_str_sections=formatted_str_sections
    )
    intro = llm.invoke(
        [instructions] + 
        [HumanMessage(content="Write the report introduction")]
    )
    return {"introduction": intro.content}


def write_conclusion(state: ResearchGraphState):
    """Write the report conclusion."""
    llm = get_llm()
    sections = state["sections"]
    topic = state["topic"]
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    instructions = CONCLUSION_INSTRUCTIONS.format(
        topic=topic,
        formatted_str_sections=formatted_str_sections
    )
    conclusion = llm.invoke(
        [instructions] + 
        [HumanMessage(content="Write the report conclusion")]
    )
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """Reduce step: combine all sections into final report."""
    content = state["content"]
    if content.startswith("## News Analysis"):
        content = content.strip("## News Analysis")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None
    
    final_report = (
        state["introduction"] + 
        "\n\n---\n\n" + 
        content + 
        "\n\n---\n\n" + 
        state["conclusion"]
    )
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}


# ============================================================================
# Graph Construction
# ============================================================================

def build_analysis_graph():
    """Build the analysis subgraph for individual analysts."""
    analysis_builder = StateGraph(AnalysisState)
    analysis_builder.add_node("ask_question", generate_question)
    analysis_builder.add_node("search_news", search_news)
    analysis_builder.add_node("answer_question", generate_answer)
    analysis_builder.add_node("save_analysis", save_analysis)
    analysis_builder.add_node("write_section", write_section)
    
    analysis_builder.add_edge(START, "ask_question")
    analysis_builder.add_edge("ask_question", "search_news")
    analysis_builder.add_edge("search_news", "answer_question")
    analysis_builder.add_conditional_edges(
        "answer_question", 
        route_messages, 
        ['ask_question', 'save_analysis']
    )
    analysis_builder.add_edge("save_analysis", "write_section")
    analysis_builder.add_edge("write_section", END)
    
    return analysis_builder


def build_research_graph():
    """Build the main research graph."""
    analysis_builder = build_analysis_graph()
    
    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("conduct_analysis", analysis_builder.compile())
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)
    
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges(
        "human_feedback", 
        initiate_all_analyses, 
        ["create_analysts", "conduct_analysis"]
    )
    builder.add_edge("conduct_analysis", "write_report")
    builder.add_edge("conduct_analysis", "write_introduction")
    builder.add_edge("conduct_analysis", "write_conclusion")
    builder.add_edge(
        ["write_conclusion", "write_report", "write_introduction"], 
        "finalize_report"
    )
    builder.add_edge("finalize_report", END)
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ============================================================================
# Main Execution
# ============================================================================

def run_research(topic: str, max_analysts: int = 3) -> str:
    """
    Run the research analysis on a given topic.
    
    Args:
        topic: The topic to analyze
        max_analysts: Number of analysts to create (default: 3)
        
    Returns:
        The final research report as a string
    """
    graph = build_research_graph()
    thread = {"configurable": {"thread_id": "1"}}
    
    final_report = None
    print(f"\n🔍 Starting research on: {topic}")
    print(f"📊 Using {max_analysts} analysts\n")
    
    for event in graph.stream(
        {"topic": topic, "max_analysts": max_analysts},
        thread,
        stream_mode="updates"
    ):
        step_name = list(event.keys())[0]
        print(f"  ✓ Processing: {step_name}")
        
        if "finalize_report" in event:
            final_report = event["finalize_report"]["final_report"]
    
    print("\n✅ Research complete!\n")
    return final_report


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Research Analyzer - A multi-analyst news research system"
    )
    parser.add_argument(
        "--topic", 
        type=str, 
        default="Artificial Intelligence Safety Regulations",
        help="Topic to research (default: 'Artificial Intelligence Safety Regulations')"
    )
    parser.add_argument(
        "--max-analysts", 
        type=int, 
        default=3,
        help="Number of analysts to use (default: 3)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file path (optional, prints to console if not specified)"
    )
    
    args = parser.parse_args()
    
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("   Please set it in a .env file or export it directly.")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Error: TAVILY_API_KEY not found in environment variables.")
        print("   Please set it in a .env file or export it directly.")
        return
    
    # Run the research
    report = run_research(args.topic, args.max_analysts)
    
    if report:
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to: {args.output}")
        else:
            print("=" * 80)
            print("FINAL REPORT")
            print("=" * 80)
            print(report)


if __name__ == "__main__":
    main()
