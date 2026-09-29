"""
PatentLens - Explainable Multi-Agent RAG
----------------------------------------

Graph:
    Supervisor
        |
        +--> CHAT ------------------------------> Direct Answer
        |
        +--> SEARCH / ANALYSIS / SIMILARITY / COMPARE
                    |
                    v
              Search Agent
                    |
                    v
             Specialist Agent
          /       |       |       \
     Analysis  Similarity Compare  Research
                    |
                    v
          Evidence Verification
             |             |
          enough         not enough
             |             |
             v             v
         Synthesis     Query Rewriter
             |             |
             v             +----> Search Agent
          Critic
          |    |
        PASS  REVISE
          |    |
         END  Synthesis (one revision)

Public functions:
    build_agent(retriever, llm)
    run_agent(agent, question)
"""

import operator
import re
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph


# ============================================================
# CONFIGURATION
# ============================================================

MAX_SEARCH_ATTEMPTS = 3
MAX_CRITIC_REVISIONS = 1
MAX_DOC_CHARS = 3500
MAX_CONTEXT_CHARS = 18000


# ============================================================
# GRAPH STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    query: str
    intent: str
    patent_ids: List[str]

    documents: List[Any]
    attempts: int

    sufficient: bool
    verification_reason: str

    specialist_output: str

    draft: str
    answer: str

    critic_output: str
    critic_passed: bool
    revision_count: int

    steps: Annotated[List[str], operator.add]
    agent_outputs: Dict[str, str]


# ============================================================
# BASIC HELPERS
# ============================================================

def _ask(llm, prompt: str) -> str:
    """Call the LLM and always return clean text."""
    response = llm.invoke(prompt)

    content = getattr(response, "content", response)

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        content = " ".join(parts)

    return str(content).strip()


def _normalize_patent_id(value: str) -> str:
    match = re.search(r"(\d{1,4})", value)

    if not match:
        return value.strip()

    return f"patent_{int(match.group(1)):03d}"


def extract_patent_ids(text: str) -> List[str]:
    """Extract patent_001, patent-001, patent 001, etc."""
    patterns = [
        r"\bpatent[_\s-]?(\d{1,4})\b",
        r"\bpat[_\s-]?(\d{1,4})\b",
    ]

    found = []

    for pattern in patterns:
        found.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    return sorted({_normalize_patent_id(x) for x in found})


def _patent_id(doc) -> str:
    return (
        str(doc.metadata.get("patent_id", "Unknown"))
        .replace(" (1)", "")
        .strip()
    )


def _page(doc) -> str:
    return str(doc.metadata.get("page", "Unknown"))


# ============================================================
# DOCUMENT HELPERS
# ============================================================

def deduplicate_documents(documents: List[Any]) -> List[Any]:
    output = []
    seen = set()

    for doc in documents:
        key = (
            _patent_id(doc),
            _page(doc),
            str(doc.page_content),
        )

        if key not in seen:
            seen.add(key)
            output.append(doc)

    return output


def format_documents(documents: List[Any]) -> str:
    """Convert retrieved chunks into safe LLM context."""
    if not documents:
        return "No patent evidence was retrieved."

    sections = []
    total_chars = 0

    for doc in documents:
        content = str(doc.page_content).strip()

        if not content:
            continue

        content = content[:MAX_DOC_CHARS]

        section = (
            f"[Patent: {_patent_id(doc)}, Page: {_page(doc)}]\n"
            f"{content}"
        )

        if total_chars + len(section) > MAX_CONTEXT_CHARS:
            break

        sections.append(section)
        total_chars += len(section)

    return "\n\n".join(sections) or "No usable patent evidence was retrieved."


def extract_sources(documents: List[Any]) -> List[Dict[str, str]]:
    sources = []
    seen = set()

    for doc in documents:
        patent_id = _patent_id(doc)
        page = _page(doc)

        key = (patent_id, page)

        if key not in seen:
            seen.add(key)

            sources.append(
                {
                    "patent_id": patent_id,
                    "page": page,
                }
            )

    return sources


def serialize_evidence(documents: List[Any]) -> List[Dict[str, str]]:
    return [
        {
            "patent_id": _patent_id(doc),
            "page": _page(doc),
            "content": str(doc.page_content)[:MAX_DOC_CHARS],
        }
        for doc in documents
    ]


def _merge_documents(
    existing: List[Any],
    new_docs: List[Any],
) -> List[Any]:
    return deduplicate_documents(existing + list(new_docs))


# ============================================================
# VECTOR SEARCH
# ============================================================

def _vector_search(
    retriever,
    query: str,
    k: int = 6,
    patent_id: str = "",
):
    """
    Search the underlying vector store when possible.

    For COMPARE, patent_id restricts retrieval to a particular patent.
    If filtering is unavailable in the installed configuration, the
    function falls back to normal retriever search.
    """

    vectorstore = getattr(retriever, "vectorstore", None)

    if vectorstore is not None and hasattr(
        vectorstore,
        "similarity_search",
    ):

        # Targeted patent search
        if patent_id:
            try:
                return vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={"patent_id": patent_id},
                )
            except Exception:
                pass

        # Normal semantic search
        try:
            return vectorstore.similarity_search(
                query,
                k=k,
            )
        except Exception:
            pass

    # Final fallback
    try:
        return retriever.invoke(query)
    except Exception:
        return []


# ============================================================
# BUILD MULTI-AGENT GRAPH
# ============================================================

def build_agent(retriever, llm):

    # --------------------------------------------------------
    # 1. SUPERVISOR AGENT
    # --------------------------------------------------------

    def supervisor(state: AgentState):

        decision = _ask(
            llm,
            """
Classify the user's request into EXACTLY ONE label.

CHAT:
Greeting, thanks, or small talk.

SEARCH:
The user wants to find patents or information about a patent topic.

ANALYSIS:
The user wants to understand one patent's purpose, problem,
method, technology, components, abstract, claims, or operation.

SIMILARITY:
The user wants patents similar or related to an invention/topic.

COMPARE:
The user wants to compare two or more specific patents.

Return ONLY one label:

CHAT
SEARCH
ANALYSIS
SIMILARITY
COMPARE

USER QUESTION:
"""
            + state["question"],
        ).upper()

        labels = [
            "CHAT",
            "SEARCH",
            "ANALYSIS",
            "SIMILARITY",
            "COMPARE",
        ]

        intent = next(
            (label for label in labels if label in decision),
            "SEARCH",
        )

        patent_ids = extract_patent_ids(
            state["question"]
        )

        targets = (
            ", ".join(patent_ids)
            if patent_ids
            else "topic-based search"
        )

        outputs = dict(state["agent_outputs"])

        outputs["Supervisor Agent"] = (
            f"Intent: {intent}\n"
            f"Patent targets: {targets}\n"
            f"Action: delegate to the appropriate specialist."
        )

        return {
            "intent": intent,
            "patent_ids": patent_ids,
            "query": state["question"],
            "attempts": 0,
            "documents": [],
            "agent_outputs": outputs,
            "steps": [
                f"🧠 Supervisor Agent → {intent}"
            ],
        }

    # --------------------------------------------------------
    # 2. DIRECT CHAT
    # --------------------------------------------------------

    def direct_answer(state: AgentState):

        answer = _ask(
            llm,
            """
You are PatentLens.

The user is making small talk.

Reply briefly and politely.
Mention that PatentLens can analyze the uploaded patent documents.

Do not invent or discuss patent facts.

USER:
"""
            + state["question"],
        )

        outputs = dict(state["agent_outputs"])

        outputs["Synthesis Agent"] = (
            "Direct response generated without patent retrieval."
        )

        return {
            "answer": answer,
            "agent_outputs": outputs,
            "steps": [
                "💬 Direct response → no retrieval required"
            ],
        }

    # --------------------------------------------------------
    # 3. SEARCH AGENT
    # --------------------------------------------------------

    def search_agent(state: AgentState):

        query = state["query"]

        k = (
            8
            if state["intent"] == "SIMILARITY"
            else 6
        )

        docs = _vector_search(
            retriever,
            query,
            k=k,
        )

        merged = _merge_documents(
            state["documents"],
            docs,
        )

        attempt = state["attempts"] + 1

        sources = extract_sources(merged)

        source_text = ", ".join(
            f"{source['patent_id']} p.{source['page']}"
            for source in sources[:8]
        )

        outputs = dict(state["agent_outputs"])

        outputs["Search Agent"] = (
            f"Query: {query}\n"
            f"Attempt: {attempt}\n"
            f"Evidence chunks: {len(merged)}\n"
            f"Sources: {source_text or 'none'}"
        )

        return {
            "documents": merged,
            "attempts": attempt,
            "agent_outputs": outputs,
            "steps": [
                (
                    f"🔎 Search Agent → attempt {attempt}, "
                    f"{len(merged)} evidence chunks"
                )
            ],
        }

    # --------------------------------------------------------
    # 4. SPECIALIST AGENT
    # --------------------------------------------------------

    def specialist_agent(state: AgentState):

        intent = state["intent"]

        documents = list(
            state["documents"]
        )

        # ====================================================
        # COMPARISON
        # ====================================================

        if (
            intent == "COMPARE"
            and len(state["patent_ids"]) >= 2
        ):

            targeted_docs = []

            for patent_id in state["patent_ids"][:3]:

                targeted_docs.extend(
                    _vector_search(
                        retriever,
                        (
                            "purpose problem technology "
                            "method claims components"
                        ),
                        k=6,
                        patent_id=patent_id,
                    )
                )

            documents = _merge_documents(
                documents,
                targeted_docs,
            )

            prompt = f"""
You are the PatentLens Comparison Agent.

Compare these patents:

{", ".join(state["patent_ids"])}

Use ONLY the evidence below.

Create an intermediate comparison covering:

1. Purpose
2. Problem addressed
3. Main technology/method
4. Important components or steps
5. Similarities
6. Differences

Every factual statement must be supported by the supplied evidence.

Do not use outside knowledge.

EVIDENCE:
{format_documents(documents)}
"""

            specialist_name = "Comparison Agent"

        # ====================================================
        # SIMILARITY
        # ====================================================

        elif intent == "SIMILARITY":

            prompt = f"""
You are the PatentLens Similarity Agent.

The user wants patents related or similar to:

{state["question"]}

Use ONLY the supplied patent evidence.

Identify candidate patents and explain briefly why each candidate
is related to the requested topic.

Base the explanation on the retrieved text.

Do not claim that two inventions are legally equivalent or identical.

Do not use outside knowledge.

EVIDENCE:
{format_documents(documents)}
"""

            specialist_name = "Similarity Agent"

        # ====================================================
        # ANALYSIS
        # ====================================================

        elif intent == "ANALYSIS":

            prompt = f"""
You are the PatentLens Analysis Agent.

Analyze the patent information needed to answer:

{state["question"]}

Use ONLY the supplied evidence.

Extract relevant facts such as:

- purpose
- problem addressed
- technology or method
- important components
- important technical details

Attach patent ID and page information to important findings.

Do not use outside knowledge.

EVIDENCE:
{format_documents(documents)}
"""

            specialist_name = "Analysis Agent"

        # ====================================================
        # GENERAL SEARCH
        # ====================================================

        else:

            prompt = f"""
You are the PatentLens Research Analysis Agent.

Answer the user's patent-related request using ONLY the supplied evidence.

USER QUESTION:
{state["question"]}

Identify the most relevant findings and explain them clearly.

Attach patent ID and page information to important findings.

Do not use outside knowledge.

EVIDENCE:
{format_documents(documents)}
"""

            specialist_name = "Research Analysis Agent"

        output = _ask(
            llm,
            prompt,
        )

        outputs = dict(
            state["agent_outputs"]
        )

        outputs[specialist_name] = output

        return {
            "documents": documents,
            "specialist_output": output,
            "agent_outputs": outputs,
            "steps": [
                (
                    f"🧩 {specialist_name} → "
                    "specialist analysis completed"
                )
            ],
        }

    # --------------------------------------------------------
    # 5. EVIDENCE VERIFICATION AGENT
    # --------------------------------------------------------

    def verification_agent(state: AgentState):

        context = format_documents(
            state["documents"]
        )

        # Comparison has an additional deterministic check.
        if (
            state["intent"] == "COMPARE"
            and len(state["patent_ids"]) >= 2
        ):

            required = state["patent_ids"][:2]

            available = {
                _patent_id(doc).lower()
                for doc in state["documents"]
            }

            missing = [
                patent_id
                for patent_id in required
                if patent_id.lower() not in available
            ]

            if missing:

                sufficient = False

                reason = (
                    "Missing evidence for: "
                    + ", ".join(missing)
                )

            else:

                verdict = _ask(
                    llm,
                    f"""
You are the PatentLens Evidence Verification Agent.

Check whether the comparison can be answered from the evidence.

Required patents:
{", ".join(required)}

QUESTION:
{state["question"]}

SPECIALIST ANALYSIS:
{state["specialist_output"]}

EVIDENCE:
{context}

Reply exactly:

YES
Reason: <short reason>

or

NO
Reason: <short reason>

Do not use outside knowledge.
""",
                )

                sufficient = verdict.upper().startswith(
                    "YES"
                )

                reason = verdict

        else:

            verdict = _ask(
                llm,
                f"""
You are the PatentLens Evidence Verification Agent.

Determine whether the supplied evidence is sufficient to answer
the user's question accurately.

QUESTION:
{state["question"]}

SPECIALIST ANALYSIS:
{state["specialist_output"]}

EVIDENCE:
{context}

Reply exactly:

YES
Reason: <short reason>

or

NO
Reason: <short reason>

Do not use outside knowledge.
""",
            )

            sufficient = verdict.upper().startswith(
                "YES"
            )

            reason = verdict

        outputs = dict(
            state["agent_outputs"]
        )

        outputs["Evidence Verification Agent"] = (
            f"Result: "
            f"{'SUFFICIENT' if sufficient else 'INSUFFICIENT'}\n"
            f"{reason}"
        )

        return {
            "sufficient": sufficient,
            "verification_reason": reason,
            "agent_outputs": outputs,
            "steps": [
                (
                    "✅ Evidence Verification Agent → "
                    + (
                        "sufficient"
                        if sufficient
                        else "insufficient"
                    )
                )
            ],
        }

    # --------------------------------------------------------
    # 6. QUERY REWRITER AGENT
    # --------------------------------------------------------

    def rewrite_agent(state: AgentState):

        patent_text = (
            ", ".join(state["patent_ids"])
            or "none"
        )

        new_query = _ask(
            llm,
            f"""
You are the PatentLens Query Rewriter Agent.

Rewrite the user's question into a stronger semantic-search query.

Original question:
{state["question"]}

Current query:
{state["query"]}

Intent:
{state["intent"]}

Patent IDs:
{patent_text}

Rules:
- Keep important patent IDs.
- Add specific technical keywords when appropriate.
- Do not invent facts.
- Return ONLY the rewritten query.
""",
        ).strip(" '\"")

        outputs = dict(
            state["agent_outputs"]
        )

        outputs["Query Rewriter Agent"] = (
            f"New search query: {new_query}"
        )

        return {
            "query": new_query,
            "agent_outputs": outputs,
            "steps": [
                (
                    f"🔄 Query Rewriter Agent → "
                    f"{new_query}"
                )
            ],
        }

    # --------------------------------------------------------
    # 7. SYNTHESIS AGENT
    # --------------------------------------------------------

    def synthesis_agent(state: AgentState):

        previous_draft = state["draft"]

        if state["sufficient"]:

            evidence_instruction = """
The evidence is sufficient.

Answer the question directly and cite relevant
patent IDs and pages.
"""

        else:

            evidence_instruction = """
The evidence is NOT sufficient.

Do not guess.

Clearly say:
"I could not find sufficient information in the provided patent documents."
"""

        revision_instruction = ""

        if previous_draft:

            revision_instruction = f"""
A previous draft was reviewed by the Critic Agent.

PREVIOUS DRAFT:
{previous_draft}

CRITIC FEEDBACK:
{state["critic_output"]}

Revise the answer using the critic feedback.
"""

        prompt = f"""
You are the PatentLens Synthesis Agent.

Create the final answer to the user's question.

USER QUESTION:
{state["question"]}

INTENT:
{state["intent"]}

SPECIALIST ANALYSIS:
{state["specialist_output"]}

EVIDENCE:
{format_documents(state["documents"])}

VERIFICATION:
{state["verification_reason"]}

{evidence_instruction}

{revision_instruction}

STRICT RULES:
- Use ONLY the supplied patent evidence.
- Never invent patent facts.
- Do not use outside knowledge.
- Keep the answer clear and professional.
- Include patent ID and page references for important claims.
- If evidence is insufficient, say:
  "I could not find sufficient information in the provided patent documents."

For COMPARE requests, use a compact comparison table
when enough evidence is available.

Return:

ANSWER:
<direct answer>

KEY DETAILS:
- <supporting detail with patent/page>
- <supporting detail with patent/page>

SOURCES:
- <patent ID, page>
"""

        answer = _ask(
            llm,
            prompt,
        )

        revision_count = (
            state["revision_count"]
            + (1 if previous_draft else 0)
        )

        outputs = dict(
            state["agent_outputs"]
        )

        outputs["Synthesis Agent"] = answer

        revision_text = (
            f" (revision {revision_count})"
            if previous_draft
            else ""
        )

        return {
            "draft": answer,
            "answer": answer,
            "revision_count": revision_count,
            "agent_outputs": outputs,
            "steps": [
                (
                    "✨ Synthesis Agent → "
                    f"final draft generated{revision_text}"
                )
            ],
        }

    # --------------------------------------------------------
    # 8. CRITIC AGENT
    # --------------------------------------------------------

    def critic_agent(state: AgentState):

        verdict = _ask(
            llm,
            f"""
You are the PatentLens Critic Agent.

Review the proposed final answer against the supplied evidence.

USER QUESTION:
{state["question"]}

FINAL DRAFT:
{state["draft"]}

EVIDENCE:
{format_documents(state["documents"])}

Check:

1. Are factual claims supported by the evidence?
2. Are patent IDs/pages used correctly?
3. Did the answer introduce outside information?
4. Did it answer the actual question?
5. If evidence was insufficient, did it avoid guessing?

Reply exactly:

PASS
Reason: <short reason>

or

REVISE
Reason: <short reason>
""",
        )

        passed = verdict.upper().startswith(
            "PASS"
        )

        outputs = dict(
            state["agent_outputs"]
        )

        outputs["Critic Agent"] = (
            f"Result: "
            f"{'PASS' if passed else 'REVISE'}\n"
            f"{verdict}"
        )

        return {
            "critic_passed": passed,
            "critic_output": verdict,
            "agent_outputs": outputs,
            "steps": [
                (
                    "🧐 Critic Agent → "
                    + (
                        "PASS"
                        if passed
                        else "REVISION REQUESTED"
                    )
                )
            ],
        }

    # ========================================================
    # ROUTING FUNCTIONS
    # ========================================================

    def after_supervisor(state: AgentState):

        if state["intent"] == "CHAT":
            return "direct_answer"

        return "search_agent"

    def after_verification(state: AgentState):

        if state["sufficient"]:
            return "synthesis_agent"

        if state["attempts"] < MAX_SEARCH_ATTEMPTS:
            return "rewrite_agent"

        return "synthesis_agent"

    def after_critic(state: AgentState):

        if state["critic_passed"]:
            return END

        if state["revision_count"] < MAX_CRITIC_REVISIONS:
            return "synthesis_agent"

        return END

    # ========================================================
    # CREATE LANGGRAPH
    # ========================================================

    workflow = StateGraph(
        AgentState
    )

    workflow.add_node(
        "supervisor",
        supervisor,
    )

    workflow.add_node(
        "direct_answer",
        direct_answer,
    )

    workflow.add_node(
        "search_agent",
        search_agent,
    )

    workflow.add_node(
        "specialist_agent",
        specialist_agent,
    )

    workflow.add_node(
        "verification_agent",
        verification_agent,
    )

    workflow.add_node(
        "rewrite_agent",
        rewrite_agent,
    )

    workflow.add_node(
        "synthesis_agent",
        synthesis_agent,
    )

    workflow.add_node(
        "critic_agent",
        critic_agent,
    )

    # START
    workflow.add_edge(
        START,
        "supervisor",
    )

    # Supervisor routing
    workflow.add_conditional_edges(
        "supervisor",
        after_supervisor,
        [
            "direct_answer",
            "search_agent",
        ],
    )

    # Chat path
    workflow.add_edge(
        "direct_answer",
        END,
    )

    # RAG path
    workflow.add_edge(
        "search_agent",
        "specialist_agent",
    )

    workflow.add_edge(
        "specialist_agent",
        "verification_agent",
    )

    # Verification routing
    workflow.add_conditional_edges(
        "verification_agent",
        after_verification,
        [
            "synthesis_agent",
            "rewrite_agent",
        ],
    )

    # Retry path
    workflow.add_edge(
        "rewrite_agent",
        "search_agent",
    )

    # Final answer
    workflow.add_edge(
        "synthesis_agent",
        "critic_agent",
    )

    # Critic routing
    workflow.add_conditional_edges(
        "critic_agent",
        after_critic,
        [
            END,
            "synthesis_agent",
        ],
    )

    return workflow.compile()


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(agent, question: str) -> dict:

    initial_state: AgentState = {
        "question": question,
        "query": question,
        "intent": "",
        "patent_ids": [],

        "documents": [],
        "attempts": 0,

        "sufficient": False,
        "verification_reason": "",

        "specialist_output": "",

        "draft": "",
        "answer": "",

        "critic_output": "",
        "critic_passed": False,
        "revision_count": 0,

        "steps": [],
        "agent_outputs": {},
    }

    result = agent.invoke(
        initial_state
    )

    if result.get("intent") == "CHAT":
        sources = []
    else:
        sources = extract_sources(
            result.get("documents", [])
        )

    return {
        "answer": result.get(
            "answer",
            "",
        ),

        "sources": sources,

        "steps": result.get(
            "steps",
            [],
        ),

        "intent": result.get(
            "intent",
            "CHAT",
        ),

        "attempts": result.get(
            "attempts",
            0,
        ),

        "patent_ids": result.get(
            "patent_ids",
            [],
        ),

        "agent_outputs": result.get(
            "agent_outputs",
            {},
        ),

        "retrieved_evidence": serialize_evidence(
            result.get(
                "documents",
                [],
            )
        ),

        "verification": {
            "sufficient": result.get(
                "sufficient",
                False,
            ),

            "reason": result.get(
                "verification_reason",
                "",
            ),
        },
    }

