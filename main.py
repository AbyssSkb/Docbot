import json
import logging
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from create_index import sha256_file
from pipeline import (
    ATTEMPT_STATUS_LABELS,
    CITATION_RE,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_REVISION,
    load_index_bundle,
    load_models,
    run_agent,
)


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docbot")

INDEX_DIR = Path("index")
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
RETRIEVAL_CONFIG_LABELS = {
    "hybrid": "Hybrid（默认，快速）",
    "hybrid_rerank": "Hybrid + Qwen3 Reranker（精排，较慢）",
}


@st.cache_resource(show_spinner=False)
def load_resources(manifest_checksum, base_url):
    del manifest_checksum  # It exists to invalidate Streamlit's resource cache.
    manifest, chunks = load_index_bundle(INDEX_DIR)
    configured_manifest = {
        **manifest,
        "embedding_models": [
            entry
            for entry in manifest["embedding_models"]
            if Path(entry["index"]).stem in {"embed1", "embed2"}
        ],
    }
    retrievers, _ = load_models(
        configured_manifest,
        chunks,
        INDEX_DIR,
    )
    llm = OpenAI(base_url=base_url or None, timeout=60, max_retries=2)
    return manifest, chunks, retrievers, llm


@st.cache_resource(show_spinner=False)
def load_reranker():
    from model import RerankingModel

    return RerankingModel(
        DEFAULT_RERANKER_MODEL,
        revision=DEFAULT_RERANKER_REVISION,
    )


def render_sources(citation_ids, chunks_by_id):
    if not citation_ids:
        return
    with st.expander(f"查看引用原文（{len(citation_ids)} 条）"):
        for number, chunk_id in enumerate(citation_ids, start=1):
            chunk = chunks_by_id[chunk_id]
            location = chunk["source"]
            if chunk["section"]:
                location += f" · {chunk['section']}"
            with st.container(border=True):
                st.caption(f"[{number}] {location}")
                st.write(chunk["text"])


def format_answer_citations(answer, citation_ids):
    labels = {chunk_id: str(number) for number, chunk_id in enumerate(citation_ids, 1)}
    return CITATION_RE.sub(
        lambda match: f"[{labels.get(match.group(1), match.group(1))}]", answer
    )


def format_attempt_counts(attempt):
    searches = attempt.get("search_count", attempt["attempt"])
    retries = attempt.get("retry_count", 0)
    return (f"检索 {searches} 次" if searches else "未检索") + (
        f" · 重试 {retries} 次" if retries else ""
    )


def render_attempt(attempt):
    st.markdown(
        f"**第 {attempt['attempt']}/{attempt['max_attempts']} 步 · {format_attempt_counts(attempt)}**"
        if attempt["attempt"] else "**先尝试回答（未检索）**"
    )
    if attempt.get("retrieval_query"):
        st.caption("检索文本")
        st.code(attempt["retrieval_query"], language=None, wrap_lines=True)
    if attempt["attempt"] and "context_chunk_ids" in attempt:
        st.caption(
            f"本次召回：{len(attempt['retrieved_chunk_ids'])} 个 chunk · "
            f"新增 {attempt.get('new_chunk_count', len(attempt['retrieved_chunk_ids']))} 个 · "
            f"累计上下文（含历史引用）：{len(attempt.get('context_chunk_ids', attempt['retrieved_chunk_ids']))} 个 · "
            f"检索耗时：{attempt['retrieval_ms'] / 1000:.2f} 秒"
        )
    st.caption(ATTEMPT_STATUS_LABELS[attempt["status"]])
    if attempt.get("error"):
        st.caption(attempt["error"])
    if "answer" in attempt:
        st.caption(
            f"模型决策 {attempt['generation_ms'] / 1000:.2f} 秒"
        )


def render_attempts(attempts):
    if not attempts:
        return
    with st.expander(f"处理过程（{format_attempt_counts(attempts[-1])}）"):
        for attempt in attempts:
            render_attempt(attempt)


st.title("Docbot")
retrieval_config = st.sidebar.radio(
    "检索策略",
    options=tuple(RETRIEVAL_CONFIG_LABELS),
    format_func=RETRIEVAL_CONFIG_LABELS.get,
)
max_attempts = st.sidebar.number_input(
    "最大搜索与重试次数",
    min_value=1,
    value=5,
    step=1,
    help="首次回答不占次数；每次搜索或修正模型输出错误均消耗 1 次，共用此上限。重试不重复检索。",
)

try:
    manifest_path = INDEX_DIR / "manifest.json"
    manifest_checksum = sha256_file(manifest_path)
    with st.status("正在加载索引与模型...", expanded=False) as status:
        manifest, chunks, retrievers, llm = load_resources(
            manifest_checksum,
            os.getenv("OPENAI_BASE_URL", ""),
        )
        ranker = load_reranker() if retrieval_config == "hybrid_rerank" else None
        status.update(label="初始化完毕", state="complete")
except Exception as error:
    st.error(str(error))
    st.stop()

chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
if st.session_state.get("index_checksum") != manifest_checksum:
    st.session_state.messages = []
    st.session_state.index_checksum = manifest_checksum
st.caption(
    f"{manifest['counts']['documents']} 份文档 · {manifest['counts']['chunks']} 个片段 · "
    f"{RETRIEVAL_CONFIG_LABELS[retrieval_config]}"
)
if not st.session_state.messages:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "你好，我会基于已索引文档回答，并给出可核验的原文引用。",
            "citations": [],
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_attempts(message.get("attempts", []))
        st.write(
            format_answer_citations(message["content"], message.get("citations", []))
        )
        render_sources(message.get("citations", []), chunks_by_id)

if query := st.chat_input("询问已索引文档"):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    history = [
        {
            "role": item["role"],
            "content": item["content"],
            "cited_chunks": item.get("cited_chunks", [
                chunks_by_id[chunk_id] for chunk_id in item.get("citations", [])
            ]),
        }
        for item in st.session_state.messages[:-1]
        if item.get("role") in {"user", "assistant"}
    ]
    trace_id = uuid.uuid4().hex[:12]
    attempts = []
    with st.chat_message("assistant"):
        with st.status("正在尝试回答...", expanded=True) as status:
            active_attempt = None
            try:
                for event in run_agent(
                    query,
                    chunks,
                    retrievers,
                    llm,
                    LLM_MODEL,
                    ranker=ranker,
                    history=history,
                    max_attempts=max_attempts,
                ):
                    if event["attempt"] != active_attempt:
                        attempt_view = st.empty()
                        active_attempt = event["attempt"]
                    with attempt_view.container():
                        render_attempt(event)
                    label = (
                        (f"第 {active_attempt}/{max_attempts} 步 · " if active_attempt else "未检索 · ")
                        + ATTEMPT_STATUS_LABELS[event["status"]]
                    )
                    if "answer" in event:
                        attempts.append(event)
                        if (
                            event["status"] != "answered"
                            and active_attempt < max_attempts
                        ):
                            label += "，即将修正重试" if event["status"] == "llm_invalid" else "，即将执行检索"
                    status.update(label=label)
                result = attempts[-1]
                answered = result["status"] == "answered"
                status.update(
                    label=(
                        f"已完成 · {format_attempt_counts(result)}"
                        if answered
                        else f"已达到 {max_attempts} 次上限，未生成可用答案"
                    ),
                    state="complete" if answered else "error",
                )
            except Exception as error:
                status.update(label="处理失败", state="error")
                logger.exception("agent_failed trace_id=%s", trace_id)
                st.error(f"生成失败：{error}")
                st.stop()
        answer = result["answer"]
        citation_ids = result["cited_chunk_ids"]
        st.write(format_answer_citations(answer, citation_ids))
        render_sources(citation_ids, chunks_by_id)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citation_ids,
            "cited_chunks": result["cited_chunks"],
            "attempts": attempts,
        }
    )
    logger.info(
        "query_trace %s",
        json.dumps(
            {
                "trace_id": trace_id,
                "index": manifest_checksum,
                "retrieval_config": retrieval_config,
                "answer_status": result["status"],
                "query": query,
                "retrieval_query": result["retrieval_query"],
                "attempt_count": result["attempt"],
                "search_count": result["search_count"],
                "retry_count": result["retry_count"],
                "max_attempts": max_attempts,
                "attempts": attempts,
                "retrieval_ms": round(
                    sum(item["retrieval_ms"] for item in attempts), 2
                ),
                "generation_ms": round(
                    sum(item["generation_ms"] for item in attempts), 2
                ),
                "retrieved_chunk_ids": result["retrieved_chunk_ids"],
                "cited_chunk_ids": citation_ids,
                "invalid_citation_ids": result["invalid_citation_ids"],
                "citation_validation_failed": result["citation_validation_failed"],
            },
            ensure_ascii=False,
        ),
    )
