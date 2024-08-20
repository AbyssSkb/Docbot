import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from model import RerankingModel, EmbeddingModel, BM25Model
import streamlit as st
import time

api_key = "YOUR_OPENAI_API_KEY"
base_url = None
llm_model = "gpt-4o"

@st.cache_data(show_spinner=False)
def load_texts():
    start_time = time.time()
    print("开始加载文本")
    with open('index/texts.json', 'r', encoding='utf-8') as f:
        texts = json.load(f)
    
    end_time = time.time()
    print(f"文本加载完毕, 花费 {(end_time - start_time):.2f} 秒")
    return texts

@st.cache_resource(show_spinner=False)
def load_model():
    start_time = time.time()
    print("开始加载模型")
    embed1 = EmbeddingModel('thenlper/gte-large-zh', 'index/embed1.index')
    embed2 = EmbeddingModel('BAAI/bge-large-zh-v1.5', 'index/embed2.index')
    bm25 = BM25Model(st.session_state.texts) 
    models = [embed1, embed2, bm25]
    
    ranker = RerankingModel('BAAI/bge-reranker-large')

    llm = ChatOpenAI(model=llm_model, temperature=0.7, api_key=api_key, base_url=base_url)
    end_time = time.time()
    print(f"模型加载完毕, 花费 {(end_time - start_time):.2f} 秒")
    return models, ranker, llm

@st.cache_data(show_spinner="正在搜索上下文...")
def find_context(query):
    start_time = time.time()
    print("开始搜索上下文")
    context = []
    for model in st.session_state.models:
        indices = model.query(query)
        context += [st.session_state.texts[index] for index in indices]

    context = st.session_state.ranker.select(query, context)
    end_time = time.time()
    print(f"上下文搜索完毕，花费 {(end_time - start_time):.2f} 秒")
    return context
    
def process_query(query, context):
    start_time = time.time()
    print("开始询问")
    content = "上下文：" + "\n\n".join(context) + "\n\n问题：" + query
    messages = [
        SystemMessage(content="请你基于以下材料回答用户问题。回答要清晰准确，包含正确关键词。不要胡编乱造。如果所给材料与用户问题无关，只输出：无答案。"),
        HumanMessage(content=content),
    ]
    with st.chat_message("assistant"):
        response = st.write_stream(st.session_state.llm.stream(messages))

    st.session_state.messages.append({"role": "assistant", "content": response})
    end_time = time.time()
    print(f"询问完毕，花费 {(end_time - start_time):.2f} 秒")

st.title("Docbot")

if "initialized" not in st.session_state:
    with st.status("正在初始化...") as status:
        st.write("加载文本...")
        st.session_state.texts = load_texts()
        st.write("加载模型...")
        st.session_state.models, st.session_state.ranker, st.session_state.llm = load_model()
        status.update(label="初始化完毕！", state="complete")
    st.session_state.initialized = True
    time.sleep(1)
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好，我是 AI 文档机器人，请问你有什么问题吗，我很乐意帮你解答。"}]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    context = find_context(user_input)
    process_query(user_input, context)