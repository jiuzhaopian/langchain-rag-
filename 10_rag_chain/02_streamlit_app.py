"""
02_streamlit_app.py - RAG 对话应用（Streamlit 界面）

两个页面：
  📄 文档管理：上传文件、查看已上传文档、调整切分参数、清空向量库
  💬 RAG 对话：多轮对话、侧边栏调优（k/search_type/score_threshold）

启动方式：
  streamlit run 02_streamlit_app.py

依赖：
  pip install streamlit langchain-chroma langchain-community langchain-text-splitters langchain-zhipu pypdf
"""

import importlib
import sys
from pathlib import Path

import streamlit as st

# 动态导入带数字前缀的模块（Python 不允许 from 00_xxx import）
_module_dir = Path(__file__).parent
sys.path.insert(0, str(_module_dir))
rag_config = importlib.import_module("00_rag_config")
rag_engine = importlib.import_module("01_rag_engine")

CHROMA_PERSIST_DIR = rag_config.CHROMA_PERSIST_DIR
DEFAULT_CHUNK_SIZE = rag_config.DEFAULT_CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = rag_config.DEFAULT_CHUNK_OVERLAP
DEFAULT_K = rag_config.DEFAULT_K
DEFAULT_SEARCH_TYPE = rag_config.DEFAULT_SEARCH_TYPE
DEFAULT_SCORE_THRESHOLD = rag_config.DEFAULT_SCORE_THRESHOLD
DEFAULT_MMR_LAMBDA = rag_config.DEFAULT_MMR_LAMBDA
get_llm = rag_config.get_llm
get_embeddings = rag_config.get_embeddings
DocumentManager = rag_engine.DocumentManager
RAGEngine = rag_engine.RAGEngine

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="RAG 对话系统",
    page_icon="📖",
    layout="wide",
)

# ============================================================
# 初始化（全局只执行一次）
# ============================================================

@st.cache_resource
def init_engine():
    """初始化文档管理器和 RAG 引擎（Streamlit 缓存，全局单例）"""
    embeddings = get_embeddings()
    llm = get_llm()
    doc_manager = DocumentManager(
        persist_dir=CHROMA_PERSIST_DIR,
        embeddings=embeddings,
    )
    rag_engine = RAGEngine(doc_manager=doc_manager, llm=llm)
    return doc_manager, rag_engine


doc_manager, rag_engine = init_engine()

# ============================================================
# 侧边栏导航 + 全局调优参数
# ============================================================

with st.sidebar:
    st.title("📖 RAG 对话系统")

    page = st.radio("选择页面", ["📄 文档管理", "💬 RAG 对话"])

    st.divider()
    st.subheader("🔧 检索调优")

    k = st.slider("返回文档数 (k)", min_value=1, max_value=10, value=DEFAULT_K)
    search_type = st.selectbox(
        "检索策略",
        options=["similarity", "mmr", "similarity_score_threshold"],
        index=0,
        format_func=lambda x: {
            "similarity": "相似度（默认）",
            "mmr": "MMR（平衡相关性与多样性）",
            "similarity_score_threshold": "相似度阈值过滤",
        }[x],
    )
    score_threshold = st.slider(
        "相似度阈值",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_SCORE_THRESHOLD,
        step=0.05,
        disabled=(search_type != "similarity_score_threshold"),
    )
    mmr_lambda = st.slider(
        "MMR 多样性 (lambda)",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_MMR_LAMBDA,
        step=0.05,
        disabled=(search_type != "mmr"),
    )

    st.divider()
    st.subheader("💬 对话设置")
    stream_mode = st.checkbox("流式输出", value=True)

# ============================================================
# 页面 1：文档管理
# ============================================================

if page == "📄 文档管理":
    st.header("📄 文档管理")

    # 上传区域
    with st.expander("上传文档", expanded=True):
        chunk_size = st.number_input("chunk_size", value=DEFAULT_CHUNK_SIZE, min_value=100, max_value=2000, step=50)
        chunk_overlap = st.number_input("chunk_overlap", value=DEFAULT_CHUNK_OVERLAP, min_value=0, max_value=500, step=10)

        uploaded_files = st.file_uploader(
            "选择文件（支持 txt / md / csv / pdf）",
            type=["txt", "md", "csv", "pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("处理文件", type="primary"):
            # 临时保存上传的文件，处理后删除
            import tempfile
            import os

            total_chunks = 0
            for file in uploaded_files:
                suffix = Path(file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                try:
                    # 临时覆盖切分参数
                    original_size = doc_manager.chunk_size
                    original_overlap = doc_manager.chunk_overlap
                    doc_manager.chunk_size = chunk_size
                    doc_manager.chunk_overlap = chunk_overlap
                    doc_manager._splitter = __import__(
                        "langchain_text_splitters", fromlist=["RecursiveCharacterTextSplitter"]
                    ).RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    count = doc_manager.process_file(tmp_path)
                    total_chunks += count
                    st.success(f"✅ {file.name}: 存入 {count} 个文档块")

                    # 恢复默认参数
                    doc_manager.chunk_size = original_size
                    doc_manager.chunk_overlap = original_overlap
                    doc_manager._splitter = __import__(
                        "langchain_text_splitters", fromlist=["RecursiveCharacterTextSplitter"]
                    ).RecursiveCharacterTextSplitter(
                        chunk_size=original_size,
                        chunk_overlap=original_overlap,
                    )
                except Exception as e:
                    st.error(f"❌ {file.name}: {e}")
                finally:
                    os.unlink(tmp_path)

            if total_chunks > 0:
                st.info(f"共处理 {total_chunks} 个文档块")

    # 已上传文档列表
    st.divider()
    st.subheader("已上传的文档")

    sources = doc_manager.list_sources()
    if sources:
        for src in sources:
            st.write(f"  📎 {src}")
        if st.button("🗑️ 清空所有文档（删除向量库）", type="secondary"):
            doc_manager.delete_collection()
            st.cache_resource.clear()
            st.rerun()
    else:
        st.info("暂无文档，请先上传。")

# ============================================================
# 页面 2：RAG 对话
# ============================================================

elif page == "💬 RAG 对话":
    st.header("💬 RAG 对话")

    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 清空对话按钮
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("清空对话"):
            st.session_state.messages = []
            st.rerun()

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 用户输入
    if prompt := st.chat_input("请输入问题..."):
        # 显示用户消息
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 调用 RAG 引擎
        chat_kwargs = dict(
            query=prompt,
            history=st.session_state.messages[:-1],  # 不含当前消息
            k=k,
            search_type=search_type,
            score_threshold=score_threshold,
            mmr_lambda=mmr_lambda,
        )

        with st.chat_message("assistant"):
            try:
                if stream_mode:
                    # 流式输出：逐 token 显示
                    response = st.write_stream(rag_engine.chat_stream(**chat_kwargs))
                else:
                    # 非流式输出：等完整回复后显示
                    with st.spinner("检索中..."):
                        response = rag_engine.chat(**chat_kwargs)
                    st.markdown(response)
            except Exception as e:
                response = f"❌ 出错了: {e}"
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
