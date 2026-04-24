"""
01_rag_engine.py - RAG 核心引擎

引擎与 UI 分离：可以独立使用，不依赖 Streamlit。
每个方法对应前面模块学过的知识点：
  - DocumentManager.process_file(): 07_document_loaders + 08_text_splitters + 09_vectorstore_retriever
  - RAGEngine.chat(): 05_output_parsers + 06_chains + 09_retriever

参考文档：
  - ChatPromptTemplate: https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate
  - MessagesPlaceholder: https://reference.langchain.com/python/langchain-core/prompts/chat/MessagesPlaceholder
  - RunnableParallel: https://reference.langchain.com/python/langchain-core/runnables/base/RunnableParallel
  - RunnablePassthrough: https://reference.langchain.com/python/langchain-core/runnables/passthrough/RunnablePassthrough
"""

from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)

# 动态导入配置（数字前缀的模块无法直接 from import）
# 00_rag_config.py: 全局配置（模型、向量库、切分、检索参数）
# 01_rag_engine.py: 核心引擎（DocumentManager + RAGEngine）
# 02_streamlit_app.py: Streamlit UI（文档管理 + RAG 对话）
import sys as _sys  # 别名 _sys 避免污染模块命名空间
_module_dir = Path(__file__).parent  # 当前文件所在目录
_sys.path.insert(0, str(_module_dir))  # 将当前目录加入搜索路径，确保能找到 00_rag_config
import importlib as _importlib  # 动态导入模块的标准库
_config = _importlib.import_module("00_rag_config")  # 导入配置模块（数字前缀无法用 from ... import）


# ============================================================
# 文档管理器
# ============================================================

class DocumentManager:
    """
    负责文档的加载、切分、存储。
    对应知识点：07 Document Loaders → 08 Text Splitters → 09 VectorStore
    """

    # 文件扩展名 → Loader 映射
    LOADER_MAP = {
        ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
        ".md": lambda p: UnstructuredMarkdownLoader(str(p)),
        ".csv": lambda p: CSVLoader(str(p)),
        ".pdf": lambda p: PyPDFLoader(str(p)),
    }

    def __init__(
        self,
        persist_dir: Path,
        embeddings,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        pass

    def process_file(self, file_path: str) -> int:
        """
        处理单个文件：加载 → 切分 → 存储（增量，不删旧数据）

        Args:
            file_path: 文件路径

        Returns:
            存入的文档块数
        """
        pass

    def get_retriever(self,
                      k: int = 3,
                      search_type: str = "similarity",
                      score_threshold: float = 0.5,
                      mmr_lambda: float = 0.5):
        """
        获取检索器，支持动态调优参数（09 Retriever）

        Args:
            k: 返回文档数
            search_type: similarity | mmr | similarity_score_threshold
            score_threshold: 相似度阈值（search_type=similarity_score_threshold 时生效）
            mmr_lambda: MMR 多样性参数（search_type=mmr 时生效）
        """
        pass

    def delete_collection(self):
        """删除整个向量库（重新开始时用）"""
        self.vectorstore.delete_collection()
        # delete_collection 后需要重建实例
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
            collection_name=_config.CHROMA_COLLECTION_NAME,
            collection_metadata=_config.CHROMA_COLLECTION_METADATA,
        )

    def list_sources(self) -> list[str]:
        """列出所有已上传的文档来源"""
        # 从 Chroma 元数据中提取去重的 source_file
        results = self.vectorstore.get(include=["metadatas"])
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source_file" in meta:
                sources.add(meta["source_file"])
        return sorted(sources)


# ============================================================
# RAG 对话引擎
# ============================================================

class RAGEngine:
    """
    RAG 对话引擎：检索 + 生成 + 多轮对话历史。
    对应知识点：05 Output Parsers + 06 Chains + 09 Retriever
    """

    def __init__(self, doc_manager: DocumentManager, llm,max_history_rounds: int = None):
        pass

    def _print_retrieved_docs(self, docs: list):
        """
        打印检索到的文档（用于调试和教学观察）

        Args:
            docs: retriever 返回的 Document 列表
        """
        print(f"\n🔍 检索到 {len(docs)} 条文档（按相似度排序，不一定都相关）：")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] 来源: {doc.metadata.get('source_file', '未知')}")
            print(f"     内容: {doc.page_content[:100]}...\n")

    def _build_history_messages(self, history: list[dict]) -> list:
        """
        将对话历史构建为 LangChain Message 对象列表

        Args:
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]

        Returns:
            [HumanMessage, AIMessage, ...] 列表（只保留最近 N 轮）
        """
        if not history:
            return []
        # 只保留最近 N 轮（1 轮 = 1 条 user + 1 条 assistant）
        recent = history[-self.max_history_rounds * 2:]
        messages = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def _build_chain(self,
                     history: list[dict],
                     k: int,
                     search_type: str,
                     score_threshold: float,
                     mmr_lambda: float):
        """
        构建完整的 RAG Chain：retriever → 格式化 context → prompt → LLM → parser

        对应知识点：06 Chains（RunnablePassthrough + RunnableParallel）+ 09 Retriever

        Returns:
            构建好的 chain（未执行）
        """
        pass

    def chat(
        self,
        query: str,
        history: list[dict],
        k: int = 3,
        search_type: str = "similarity",
        score_threshold: float = 0.5,
        mmr_lambda: float = 0.5,
    ) -> str:
        """
        执行一次 RAG 对话（非流式，手动控制每一步）

        人工调用 retriever、手动拼装 context 和历史，
        chain 中只有 3 个节点：prompt → llm → parser。

        流程：
          1. 人工调用 retriever 检索相关文档
          2. 人工将检索结果格式化为 context 文本
          3. 人工将历史对话构建为 Message 列表
          4. 组装 chain（prompt | llm | parser）并执行

        Args:
            query: 用户问题
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            k: 返回文档数
            search_type: similarity | mmr | similarity_score_threshold
            score_threshold: 相似度阈值
            mmr_lambda: MMR 多样性参数

        Returns:
            LLM 生成的回答（完整字符串）
        """
        # 1. 人工调用 retriever 检索（09 Retriever）
        retriever = self.doc_manager.get_retriever(
            k=k,
            search_type=search_type,
            score_threshold=score_threshold,
            mmr_lambda=mmr_lambda,
        )
        docs = retriever.invoke(query)

        # 打印检索结果
        self._print_retrieved_docs(docs)

        # 2. 人工格式化检索结果
        context = "\n\n".join(
            f"[来源: {d.metadata.get('source_file', '未知')}]\n{d.page_content}"
            for d in docs
        )

        # 3. 人工构建历史消息（02 Messages）
        history_messages = self._build_history_messages(history)

        # 4. chain 只有 3 个节点：prompt → llm → parser（05 + 06）
        chain = self._prompt | self.llm | self._parser
        response = chain.invoke({
            "context": context,
            "history": history_messages,
            "question": query,
        })

        return response

    def chat_stream(
        self,
        query: str,
        history: list[dict],
        k: int = 3,
        search_type: str = "similarity",
        score_threshold: float = 0.5,
        mmr_lambda: float = 0.5,
    ):
        """
        执行一次 RAG 对话（流式输出，全流程 Chain）

        与 chat() 不同，这里把 retriever、history 构建等都放进 chain，
        形成完整的 pipeline：retriever → format_docs → prompt → llm → parser。
        对应知识点：06 Chains（RunnablePassthrough + RunnableParallel）

        适合实际开发：一个 chain 搞定所有步骤，用 .stream() 逐 token 输出。

        注意：本方法不打印检索结果（chain 是黑盒），如需观察检索结果，
        请使用 chat() 方法（手动控制每一步）。

        Yields:
            str: LLM 生成的 token 片段
        """
        pass
