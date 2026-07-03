"""
Core multi-agent research system logic.
Supports Groq and Gemini LLM backends.
"""

import os
import asyncio
import operator
from dataclasses import asdict, dataclass
from typing import List, Dict, Union, Any, Annotated, Optional, Literal

from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# ─── Pydantic Schemas ───────────────────────────────────────────────────────

class Section(BaseModel):
    name: str = Field(description="Name for a particular section of the report.")
    description: str = Field(description="Brief overview of the main topics and concepts to be covered.")
    research: bool = Field(description="Whether to perform web search for this section.")
    content: str = Field(description="The content for this section.")


class Sections(BaseModel):
    sections: List[Section] = Field(description="All the Sections of the overall report.")


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")


class Queries(BaseModel):
    queries: List[SearchQuery] = Field(description="List of web search queries.")

    @field_validator("queries", mode="before")
    @classmethod
    def coerce_strings(cls, v):
        if not isinstance(v, list):
            return v
        return [
            SearchQuery(search_query=item) if isinstance(item, str) else item
            for item in v
        ]


class ReportStateInput(TypedDict):
    topic: str


class ReportStateOutput(TypedDict):
    final_report: str


class ReportState(TypedDict):
    topic: str
    sections: list
    completed_sections: Annotated[list, operator.add]
    report_sections_from_research: str
    final_report: str


class SectionState(TypedDict):
    section: Any
    search_queries: list
    source_str: str
    report_sections_from_research: str
    completed_sections: list


class SectionOutputState(TypedDict):
    completed_sections: list


# ─── Prompts ────────────────────────────────────────────────────────────────

DEFAULT_REPORT_STRUCTURE = """The report structure should focus on breaking-down the user-provided topic
and building a comprehensive report in markdown using the following format:

1. Introduction (no web search needed)
      - Brief overview of the topic area

2. Main Body Sections:
      - Each section should focus on a sub-topic of the user-provided topic
      - Include any key concepts and definitions
      - Provide real-world examples or case studies where applicable

3. Conclusion (no web search needed)
      - Aim for 1 structural element (either a list or table) that distills the main body sections
      - Provide a concise summary of the report

When generating the final response in markdown, if there are special characters in the text,
such as the dollar symbol, ensure they are escaped properly for correct rendering e.g $25.5 should become \\$25.5
"""

REPORT_PLAN_QUERY_GENERATOR_PROMPT = """You are an expert technical report writer, helping to plan a report.

The report will be focused on the following topic:
{topic}

The report structure will follow these guidelines:
{report_organization}

Your goal is to generate {number_of_queries} search queries that will help gather comprehensive information for planning the report sections.

The query should:
1. Be related to the topic
2. Help satisfy the requirements specified in the report organization

Make the query specific enough to find high-quality, relevant sources while covering the depth and breadth needed for the report structure.
"""

REPORT_PLAN_SECTION_GENERATOR_PROMPT = """You are an expert technical report writer, helping to plan a report.

Your goal is to generate the outline of the sections of the report.

The overall topic of the report is:
{topic}

The report should follow this organizational structure:
{report_organization}

You should reflect on this additional context information from web searches to plan the main sections of the report:
{search_context}

Now, generate the sections of the report. Each section should have the following fields:
- Name - Name for this section of the report.
- Description - Brief overview of the main topics and concepts to be covered in this section.
- Research - Whether to perform web search for this section of the report or not.
- Content - The content of the section, which you will leave blank for now.

Consider which sections require web search.
For example, introduction and conclusion will not require research because they will distill information from other parts of the report.
"""

REPORT_SECTION_QUERY_GENERATOR_PROMPT = """Your goal is to generate targeted web search queries that will gather comprehensive information for writing a technical report section.

Topic for this section:
{section_topic}

When generating {number_of_queries} search queries, ensure that they:
1. Cover different aspects of the topic (e.g., core features, real-world applications, technical architecture)
2. Include specific technical terms related to the topic
3. Target recent information by including year markers where relevant (e.g., "2024")
4. Look for comparisons or differentiators from similar technologies/approaches
5. Search for both official documentation and practical implementation examples

Your queries should be:
- Specific enough to avoid generic results
- Technical enough to capture detailed implementation information
- Diverse enough to cover all aspects of the section plan
- Focused on authoritative sources (documentation, technical blogs, academic papers)"""

SECTION_WRITER_PROMPT = """You are an expert technical writer crafting one specific section of a technical report.

Title for the section:
{section_title}

Topic for this section:
{section_topic}

Guidelines for writing:

1. Technical Accuracy:
- Include specific version numbers
- Reference concrete metrics/benchmarks
- Cite official documentation
- Use technical terminology precisely

2. Length and Style:
- Strict 150-200 word limit
- No marketing language
- Technical focus
- Write in simple, clear language do not use complex words unnecessarily
- Start with your most important insight in **bold**
- Use short paragraphs (2-3 sentences max)

3. Structure:
- Use ## for section title (Markdown format)
- Only use ONE structural element IF it helps clarify your point:
  * Either a focused table comparing 2-3 key items (using Markdown table syntax)
  * Or a short list (3-5 items) using proper Markdown list syntax:
    - Use `*` or `-` for unordered lists
    - Use `1.` for ordered lists
    - Ensure proper indentation and spacing
- End with ### Sources that references the below source material formatted as:
  * List each source with title, date, and URL
  * Format: `- Title : URL`

3. Writing Approach:
- Include at least one specific example or case study if available
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the section content
- Focus on your single most important point

4. Use this source material obtained from web searches to help write the section:
{context}

5. Quality Checks:
- Format should be Markdown
- Exactly 150-200 words (excluding title and sources)
- Careful use of only ONE structural element (table or bullet list) and only if it helps clarify your point
- One specific example / case study if available
- Starts with bold insight
- No preamble prior to creating the section content
- Sources cited at end
- If there are special characters in the text, such as the dollar symbol,
  ensure they are escaped properly for correct rendering e.g $25.5 should become \\$25.5
"""

FINAL_SECTION_WRITER_PROMPT = """You are an expert technical writer crafting a section that synthesizes information from the rest of the report.

Title for the section:
{section_title}

Topic for this section:
{section_topic}

Available report content of already completed sections:
{context}

1. Section-Specific Approach:

For Introduction:
- Use # for report title (Markdown format)
- 50-100 word limit
- Write in simple and clear language
- Focus on the core motivation for the report in 1-2 paragraphs
- Use a clear narrative arc to introduce the report
- Include NO structural elements (no lists or tables)
- No sources section needed

For Conclusion/Summary:
- Use ## for section title (Markdown format)
- 100-150 word limit
- For comparative reports:
    * Must include a focused comparison table using Markdown table syntax
    * Table should distill insights from the report
    * Keep table entries clear and concise
- For non-comparative reports:
    * Only use ONE structural element IF it helps distill the points made in the report:
    * Either a focused table comparing items present in the report (using Markdown table syntax)
    * Or a short list using proper Markdown list syntax:
      - Use `*` or `-` for unordered lists
      - Use `1.` for ordered lists
      - Ensure proper indentation and spacing
- End with specific next steps or implications
- No sources section needed

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- Focus on your single most important point

4. Quality Checks:
- For introduction: 50-100 word limit, # for report title, no structural elements, no sources section
- For conclusion: 100-150 word limit, ## for section title, only ONE structural element at most, no sources section
- Markdown format
- Do not include word count or any preamble in your response
- If there are special characters in the text, such as the dollar symbol,
  ensure they are escaped properly for correct rendering e.g $25.5 should become \\$25.5"""


# ─── Utility Functions ───────────────────────────────────────────────────────

def get_llm(provider: str = "groq", model: str = None, api_base: str = None):
    """Get LLM based on provider. Supports groq, gemini, openai, and any OpenAI-compatible endpoint."""
    if provider == "groq":
        from langchain_groq import ChatGroq
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not set")
        model_name = model or "llama-3.3-70b-versatile"
        return ChatGroq(model=model_name, temperature=0, api_key=groq_key)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not set")
        model_name = model or "gemini-2.0-flash"
        return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=gemini_key)

    elif provider in ("openai", "openai-compatible"):
        from langchain_openai import ChatOpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not set")
        model_name = model or "gpt-4o-mini"
        kwargs = dict(model=model_name, temperature=0, api_key=openai_key)
        base = api_base or os.getenv("OPENAI_API_BASE")
        if base:
            kwargs["base_url"] = base
        llm_obj = ChatOpenAI(**kwargs)
        # Tag so callers know to use json_mode instead of tool calling
        llm_obj._use_json_mode = (provider == "openai-compatible")
        return llm_obj

    else:
        raise ValueError(f"Unknown provider: {provider}")


class _JsonWrapper:
    """
    Wraps an LLM for providers that don't support tool/function calling.
    Injects a JSON schema hint into the system prompt and parses the raw response.
    """
    def __init__(self, llm, schema):
        self._llm = llm
        self._schema = schema

    def invoke(self, messages):
        import json, re
        # Append JSON schema instructions to the last system message
        schema_hint = (
            f"\n\nRespond ONLY with valid JSON that matches this schema:\n"
            f"{json.dumps(self._schema.model_json_schema(), indent=2)}\n"
            "Do not include any explanation or markdown fences."
        )
        patched = []
        injected = False
        for m in reversed(messages):
            if not injected and hasattr(m, 'content') and m.__class__.__name__ == 'SystemMessage':
                from langchain_core.messages import SystemMessage as SM
                patched.insert(0, SM(content=m.content + schema_hint))
                injected = True
            else:
                patched.insert(0, m)
        if not injected and messages:
            from langchain_core.messages import SystemMessage as SM
            patched.insert(0, SM(content=schema_hint))

        response = self._llm.invoke(patched)
        text = response.content if hasattr(response, 'content') else str(response)
        # Strip markdown fences if present
        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
        return self._schema.model_validate(json.loads(text))


def structured_output(llm, schema):
    """For openai-compatible providers, bypass tool calling and parse JSON manually."""
    if getattr(llm, '_use_json_mode', False):
        return _JsonWrapper(llm, schema)
    return llm.with_structured_output(schema)


tavily_search = None

def get_tavily():
    global tavily_search
    if tavily_search is None:
        tavily_search = TavilySearchAPIWrapper()
    return tavily_search


async def run_search_queries(
    search_queries: List[Union[str, SearchQuery]],
    num_results: int = 4,
    include_raw_content: bool = False
) -> List[Dict]:
    tavily = get_tavily()
    search_tasks = []
    for query in search_queries:
        query_str = query.search_query if isinstance(query, SearchQuery) else str(query)
        try:
            search_tasks.append(
                tavily.raw_results_async(
                    query=query_str,
                    max_results=num_results,
                    search_depth='basic',      # 'advanced' is 2-3x slower, not worth it
                    include_answer=False,
                    include_raw_content=include_raw_content
                )
            )
        except Exception as e:
            continue
    try:
        if not search_tasks:
            return []
        search_docs = await asyncio.gather(*search_tasks, return_exceptions=True)
        return [doc for doc in search_docs if not isinstance(doc, Exception)]
    except Exception:
        return []


def format_search_query_results(
    search_response: Union[Dict, List],
    max_tokens: int = 2000,
    include_raw_content: bool = False
) -> str:
    max_chars = max_tokens * 4
    sources_list = []

    if isinstance(search_response, dict):
        if 'results' in search_response:
            sources_list.extend(search_response['results'])
        else:
            sources_list.append(search_response)
    elif isinstance(search_response, list):
        for response in search_response:
            if isinstance(response, dict):
                if 'results' in response:
                    sources_list.extend(response['results'])
                else:
                    sources_list.append(response)
            elif isinstance(response, list):
                sources_list.extend(response)

    if not sources_list:
        return "No search results found."

    unique_sources = {}
    for source in sources_list:
        if isinstance(source, dict) and 'url' in source:
            if source['url'] not in unique_sources:
                unique_sources[source['url']] = source

    formatted_text = "Content from web search:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source.get('title', 'Untitled')}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source.get('content', 'No content available')}\n===\n"
        if include_raw_content:
            raw_content = source.get("raw_content", "")
            if raw_content:
                formatted_text += f"Raw Content: {raw_content[:max_chars]}\n\n"

    return formatted_text.strip()


def format_sections(sections: list) -> str:
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research:
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str


# ─── Agent Node Functions ────────────────────────────────────────────────────

def _normalize_queries(results) -> List[str]:
    """Extract plain query strings from whatever shape the LLM returns."""
    raw = getattr(results, "queries", results) if not isinstance(results, list) else results
    out = []
    for item in raw:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, SearchQuery):
            out.append(item.search_query)
        elif isinstance(item, dict):
            out.append(item.get("search_query", str(item)))
        else:
            out.append(str(item))
    return out


def _extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "".join([block.get('text', '') if isinstance(block, dict) else str(block) for block in content])
    return str(content)


async def generate_report_plan(state: ReportState, llm):
    topic = state["topic"]
    report_structure = DEFAULT_REPORT_STRUCTURE
    number_of_queries = 4

    structured_llm = structured_output(llm, Queries)
    system_instructions_query = REPORT_PLAN_QUERY_GENERATOR_PROMPT.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries
    )

    results = structured_llm.invoke([
        SystemMessage(content=system_instructions_query),
        HumanMessage(content='Generate search queries that will help with planning the sections of the report.')
    ])

    query_list = _normalize_queries(results)

    search_docs = await run_search_queries(query_list, num_results=4, include_raw_content=False)

    if not search_docs:
        search_context = "No search results available."
    else:
        search_context = format_search_query_results(search_docs, include_raw_content=False)

    system_instructions_sections = REPORT_PLAN_SECTION_GENERATOR_PROMPT.format(
        topic=topic,
        report_organization=report_structure,
        search_context=search_context
    )

    structured_llm2 = structured_output(llm, Sections)
    report_sections = structured_llm2.invoke([
        SystemMessage(content=system_instructions_sections),
        HumanMessage(content="Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. Each section must have: name, description, plan, research, and content fields.")
    ])

    return {"sections": report_sections.sections}


async def generate_queries(state: SectionState, llm):
    section = state["section"]
    number_of_queries = 3          # 3 queries is enough; 4 adds latency with no quality gain
    structured_llm = structured_output(llm, Queries)
    system_instructions = REPORT_SECTION_QUERY_GENERATOR_PROMPT.format(
        section_topic=section.description,
        number_of_queries=number_of_queries
    )
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: structured_llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate search queries on the provided topic.")
    ]))
    return {"search_queries": [SearchQuery(search_query=q) for q in _normalize_queries(results)]}


async def search_web(state: SectionState):
    search_queries = state["search_queries"]
    search_docs = await run_search_queries(search_queries, num_results=3, include_raw_content=True)
    search_context = format_search_query_results(search_docs, max_tokens=2000, include_raw_content=True)
    return {"source_str": search_context}


async def write_section(state: SectionState, llm):
    section = state["section"]
    source_str = state["source_str"]
    system_instructions = SECTION_WRITER_PROMPT.format(
        section_title=section.name,
        section_topic=section.description,
        context=source_str
    )
    loop = asyncio.get_event_loop()
    section_content = await loop.run_in_executor(None, lambda: llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate a report section based on the provided sources.")
    ]))
    section.content = _extract_text(section_content.content)
    return {"completed_sections": [section]}


def format_completed_sections(state: ReportState):
    completed_sections = state["completed_sections"]
    completed_report_sections = format_sections(completed_sections)
    return {"report_sections_from_research": completed_report_sections}


def write_final_sections(state: SectionState, llm):
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    system_instructions = FINAL_SECTION_WRITER_PROMPT.format(
        section_title=section.name,
        section_topic=section.description,
        context=completed_report_sections
    )
    section_content = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Craft a report section based on the provided sources.")
    ])
    section.content = _extract_text(section_content.content)
    return {"completed_sections": [section]}


def compile_final_report(state: ReportState):
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}
    for section in sections:
        section.content = completed_sections.get(section.name, "")
    all_sections = "\n\n".join([
        str(s.content) if not isinstance(s.content, str) else s.content
        for s in sections
    ])
    formatted_sections = all_sections.replace("\\$", "TEMP_PLACEHOLDER")
    formatted_sections = formatted_sections.replace("$", "\\$")
    formatted_sections = formatted_sections.replace("TEMP_PLACEHOLDER", "\\$")
    return {"final_report": formatted_sections}


def parallelize_section_writing(state: ReportState):
    return [
        Send("section_builder_with_web_search", {"section": s})
        for s in state["sections"]
        if s.research
    ]


def parallelize_final_section_writing(state: ReportState):
    return [
        Send("write_final_sections", {"section": s, "report_sections_from_research": state["report_sections_from_research"]})
        for s in state["sections"]
        if not s.research
    ]


# ─── Build Graph ─────────────────────────────────────────────────────────────

def build_reporter_agent(llm):
    """Build and return the compiled LangGraph agent."""

    # Section builder sub-graph
    async def _generate_queries(state):
        return await generate_queries(state, llm)

    async def _write_section(state):
        return await write_section(state, llm)

    def _write_final_sections(state):
        return write_final_sections(state, llm)

    async def _generate_report_plan(state):
        return await generate_report_plan(state, llm)

    section_builder = StateGraph(SectionState, output_schema=SectionOutputState)
    section_builder.add_node("generate_queries", _generate_queries)
    section_builder.add_node("search_web", search_web)
    section_builder.add_node("write_section", _write_section)
    section_builder.add_edge(START, "generate_queries")
    section_builder.add_edge("generate_queries", "search_web")
    section_builder.add_edge("search_web", "write_section")
    section_builder.add_edge("write_section", END)
    section_builder_subagent = section_builder.compile()

    # Main graph
    builder = StateGraph(ReportState, input_schema=ReportStateInput, output_schema=ReportStateOutput)
    builder.add_node("generate_report_plan", _generate_report_plan)
    builder.add_node("section_builder_with_web_search", section_builder_subagent)
    builder.add_node("format_completed_sections", format_completed_sections)
    builder.add_node("write_final_sections", _write_final_sections)
    builder.add_node("compile_final_report", compile_final_report)

    builder.add_edge(START, "generate_report_plan")
    builder.add_conditional_edges("generate_report_plan", parallelize_section_writing, ["section_builder_with_web_search"])
    builder.add_edge("section_builder_with_web_search", "format_completed_sections")
    builder.add_conditional_edges("format_completed_sections", parallelize_final_section_writing, ["write_final_sections"])
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)

    return builder.compile()
