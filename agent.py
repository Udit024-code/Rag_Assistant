"""Research agent: a LangGraph state machine around hybrid retrieval.

    plan ──► retrieve ──► check ──► answer
      │          ▲          │
      │          └──────────┘   evidence missing (at most one extra round)
      └──────────────────────────► answer   (greetings, no search needed)

plan     rewrites follow-ups into a standalone question ("and BM25?" ->
         "How does BM25 compare to DPR on NQ?") and splits comparisons or
         multi-part questions into one search per part, so "ColBERT vs DPR"
         retrieves evidence from both papers instead of whichever matches best.
retrieve runs hybrid search + reranking for each new query and merges results.
check    asks whether the evidence covers every part of the question, and
         if not, which searches would fill the gap.
answer   streams a grounded answer citing the numbered sources as [1], [2].

Progress ("Searching: ...") and answer tokens are emitted through LangGraph's
custom stream, so the UI can show them live.
"""
import re
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from llm import complete_json, stream_text

PER_QUERY_K = 4      # reranked chunks kept per search
MAX_SOURCES = 10     # sources passed to the answer step (Groq free tier: 8K tokens/min)
MAX_ROUNDS = 2       # retrieval rounds, including the first
HISTORY_TURNS = 6    # previous messages the planner and answerer see


class AgentState(TypedDict, total=False):
    question: str
    history: list        # [{'role': 'user' | 'assistant', 'content': str}]
    standalone: str
    needs_search: bool
    queries: list        # searches for the next retrieval round
    searched: list       # every search run so far
    sources: list        # merged chunks, numbered [1].. in this order
    rounds: int
    answer: str


def _history_text(history, max_chars=600):
    lines = []
    for message in history[-HISTORY_TURNS:]:
        content = message['content']
        if len(content) > max_chars:
            content = content[:max_chars] + ' ...'
        lines.append(f"{message['role'].upper()}: {content}")
    return '\n'.join(lines) or '(none)'


# gpt-oss was trained to cite browsing results as 【3†L1-L3】 and sometimes
# does so despite the prompt; rewrite those to [3].
FULLWIDTH_CITATION = re.compile(r'【\s*(\d+)[^】]*】')
CITATION = re.compile(r'\[(\d+)\]')


def normalize_citations(text):
    return FULLWIDTH_CITATION.sub(r'[\1]', text)


def cited_numbers(text):
    """Source numbers the answer cites, in order of first appearance."""
    return list(dict.fromkeys(int(n) for n in CITATION.findall(normalize_citations(text))))


def _source_label(chunk):
    section = f" > {chunk['section']}" if chunk.get('section') else ''
    return f"{chunk.get('paper') or chunk['source']}, page {chunk['page']}{section}"


PLAN_PROMPT = """You plan searches for a research assistant that answers ONLY from a library of documents.

Library: {titles}

Conversation so far:
{history}

Latest user message: {question}

Return JSON:
{{"standalone_question": "...", "needs_search": true, "queries": ["..."]}}

- standalone_question: the latest message rewritten so it makes sense without the conversation (resolve "it", "that paper", "and BM25?").
- needs_search: false only for greetings, thanks, or questions about the assistant itself.
- queries: 1-3 search queries. Most questions need exactly ONE query; never add rephrasings of the same query. Only for comparisons or multi-part questions use one query per item or part (e.g. "ColBERT late interaction scoring" and "DPR dual encoder passage retrieval"). Include the specific method, model, dataset or paper names, and paraphrase into the words a paper would use."""

CHECK_PROMPT = """Question: {question}

Evidence found so far (one line per source):
{evidence}

Does this evidence cover EVERY part of the question? Return JSON:
{{"sufficient": true, "missing_queries": []}}
If something is missing, set sufficient to false and give 1-2 search queries that would find it. Don't ask for information the question doesn't need.
For a comparison, evidence about each item is enough: drawing the comparison is the answerer's job, so never search for documents that compare them directly."""

ANSWER_SYSTEM = """You are a research assistant. Answer using ONLY the numbered sources below.
- After each claim, cite the source(s) it comes from in square brackets, exactly like [1] or [2][4]. No other citation style.
- If the sources cover only part of the question, answer that part and say plainly what the sources don't cover. Never fill gaps from your own knowledge.
- If no sources are given, reply briefly and conversationally, without factual claims about the documents.
- Use markdown; tables are welcome for comparisons. Be concise."""


def build_agent(retriever, filters=None, library_titles=(), with_answer=True):
    """Compile the agent for one request. filters (owners/doc_ids) limit what retrieve can see.

    with_answer=False stops after retrieval, for evaluating what evidence the
    agent gathers without spending tokens on answers.
    """
    filters = filters or {}
    titles = '; '.join(library_titles) or '(unknown)'

    def plan(state):
        get_stream_writer()({'status': 'Planning the search'})
        result = complete_json([{'role': 'user', 'content': PLAN_PROMPT.format(
            titles=titles, history=_history_text(state.get('history', [])), question=state['question'])}])
        queries = [q for q in result.get('queries', []) if isinstance(q, str) and q.strip()][:3]
        standalone = result.get('standalone_question') or state['question']
        return {'standalone': standalone, 'needs_search': bool(result.get('needs_search', True)),
                'queries': queries or [standalone], 'searched': [], 'sources': [], 'rounds': 0}

    def retrieve(state):
        write = get_stream_writer()
        sources = list(state.get('sources', []))
        seen = {(c['doc_id'], c['chunk_id']) for c in sources}
        per_query = []
        for query in state['queries']:
            write({'status': f'Searching: {query}'})
            per_query.append(retriever(query, PER_QUERY_K, **filters))
        # Interleave results so every sub-question gets sources before any
        # single query fills the budget.
        limit = MAX_SOURCES + 2 * state.get('rounds', 0)
        for rank in range(PER_QUERY_K):
            for results in per_query:
                if rank < len(results) and len(sources) < limit:
                    chunk = results[rank]
                    if (chunk['doc_id'], chunk['chunk_id']) not in seen:
                        seen.add((chunk['doc_id'], chunk['chunk_id']))
                        sources.append(chunk)
        return {'sources': sources, 'searched': state.get('searched', []) + state['queries'],
                'rounds': state.get('rounds', 0) + 1}

    def check(state):
        if state['rounds'] >= MAX_ROUNDS or not state['sources']:
            return {'queries': []}
        get_stream_writer()({'status': 'Checking the evidence covers the question'})
        # Full chunks (they're ~192 tokens): a truncated snippet can cut off the
        # table row that holds the answer and trigger a pointless extra search.
        evidence = '\n'.join(f"- {_source_label(c)}: {c['text']}" for c in state['sources'])
        result = complete_json([{'role': 'user', 'content': CHECK_PROMPT.format(
            question=state['standalone'], evidence=evidence)}], max_tokens=512)
        missing = [q for q in result.get('missing_queries', []) if isinstance(q, str) and q.strip()]
        if result.get('sufficient', True) or not missing:
            return {'queries': []}
        # Don't repeat a search that already ran.
        return {'queries': [q for q in missing if q not in state['searched']][:2]}

    def answer(state):
        write = get_stream_writer()
        write({'status': 'Writing the answer'})
        sources = state.get('sources', [])
        context = '\n\n'.join(f'[{i}] {_source_label(c)}\n{c["text"]}' for i, c in enumerate(sources, start=1))
        messages = [{'role': 'system', 'content': ANSWER_SYSTEM}]
        messages += [{'role': m['role'], 'content': m['content']} for m in state.get('history', [])[-HISTORY_TURNS:]]
        messages.append({'role': 'user', 'content':
                         f"SOURCES:\n{context or '(none)'}\n\nQUESTION: {state['standalone']}"})
        text = ''
        for token in stream_text(messages):
            text += token
            write({'token': token})
        return {'answer': normalize_citations(text)}

    graph = StateGraph(AgentState)
    graph.add_node('plan', plan)
    graph.add_node('retrieve', retrieve)
    graph.add_node('check', check)
    graph.add_node('answer', answer)
    graph.add_edge(START, 'plan')
    finish = 'answer' if with_answer else END
    graph.add_conditional_edges('plan', lambda s: 'retrieve' if s['needs_search'] else finish)
    graph.add_edge('retrieve', 'check')
    graph.add_conditional_edges('check', lambda s: 'retrieve' if s['queries'] else finish)
    graph.add_edge('answer', END)
    return graph.compile()


def run(agent, question, history=()):
    """Stream ('status', text) and ('token', text) events, then ('done', final_state)."""
    final = {}
    for mode, chunk in agent.stream({'question': question, 'history': list(history)},
                                    stream_mode=['custom', 'updates']):
        if mode == 'custom':
            for kind in ('status', 'token'):
                if kind in chunk:
                    yield kind, chunk[kind]
        else:
            for update in chunk.values():
                final.update(update or {})
    yield 'done', final
