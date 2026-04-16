"""
01_rag_engine.py - RAG 核心引擎

引擎与 UI 分离：可以独立使用，不依赖 Streamlit。
每个方法对应前面模块学过的知识点：
  - DocumentManager.process_file(): 07_document_loaders + 08_text_splitters + 09_vectorstore_retriever
  - RAGEngine.chat(): 05_output_parsers + 06_chains + 09_retriever

参考文档：
  - Chroma: https://reference.langchain.com/python/langchain-chroma/vectorstores/Chroma
  - ChatPromptTemplate: https://reference.langchain.com/python/langchain-core/prompts/ChatPromptTemplate
"""

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
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

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatZhipuAI


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
        self.persist_dir = Path(persist_dir)
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 初始化 Chroma（已有数据自动加载）
        # 注意：bge-m3 输出归一化向量，必须用 cosine 距离，不能用默认的 l2
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
            collection_name="rag_docs",
            collection_metadata={"hnsw:space": "cosine"},
        )

    def process_file(self, file_path: str) -> int:
        """
        处理单个文件：加载 → 切分 → 存储（增量，不删旧数据）

        Args:
            file_path: 文件路径

        Returns:
            存入的文档块数
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.LOADER_MAP:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {list(self.LOADER_MAP.keys())}")

        # 1. 加载文档（07 Document Loaders）
        loader = self.LOADER_MAP[ext](path)
        docs = loader.load()

        # 为每个文档附加来源信息到 metadata
        for doc in docs:
            doc.metadata["source_file"] = path.name

        # 2. 切分（08 Text Splitters）
        splits = self._splitter.split_documents(docs)

        # 3. 存入向量库（09 VectorStore）
        if splits:
            self.vectorstore.add_documents(splits)

        return len(splits)

    def get_retriever(self, k: int = 3, search_type: str = "similarity",
                      score_threshold: float = 0.5, mmr_lambda: float = 0.5):
        """
        获取检索器，支持动态调优参数（09 Retriever）

        Args:
            k: 返回文档数
            search_type: similarity | mmr | similarity_score_threshold
            score_threshold: 相似度阈值（search_type=similarity_score_threshold 时生效）
            mmr_lambda: MMR 多样性参数（search_type=mmr 时生效）
        """
        search_kwargs = {"k": k}
        if search_type == "mmr":
            search_kwargs["fetch_k"] = k * 3
            search_kwargs["lambda_mult"] = mmr_lambda
        elif search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = score_threshold

        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def delete_collection(self):
        """删除整个向量库（重新开始时用）"""
        self.vectorstore.delete_collection()
        # delete_collection 后需要重建实例
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
            collection_name="rag_docs",
            collection_metadata={"hnsw:space": "cosine"},
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

    def __init__(self, doc_manager: DocumentManager, llm):
        self.doc_manager = doc_manager
        self.llm = llm
        self._prompt = ChatPromptTemplate.from_template(
            "基于以下检索到的文档内容回答用户的问题。\n"
            "如果文档中没有相关信息，请明确告知「检索到的文档中未找到相关信息」，不要编造。\n\n"
            "检索到的文档：\n{context}\n\n"
            "{history_block}"
            "用户问题：{question}\n\n"
            "请用中文回答："
        )
        self._parser = StrOutputParser()

    def _format_history(self, history: list[dict]) -> str:
        """将对话历史格式化为 prompt 中的文本"""
        if not history:
            return ""
        lines = []
        for msg in history[-6:]:  # 只保留最近 6 轮，控制 prompt 长度
            role = msg.get("role", "user")
            content = msg.get("content", "")
            label = "用户" if role == "user" else "助手"
            lines.append(f"{label}：{content}")
        return "历史对话：\n" + "\n".join(lines) + "\n\n"

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
        执行一次 RAG 对话

        流程：
          1. 用 retriever 检索相关文档
          2. 将检索结果 + 历史对话 + 用户问题组装成 prompt
          3. 调用 LLM 生成回答

        Args:
            query: 用户问题
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            k: 返回文档数
            search_type: similarity | mmr | similarity_score_threshold
            score_threshold: 相似度阈值
            mmr_lambda: MMR 多样性参数

        Returns:
            LLM 生成的回答
        """
        # 获取检索器（参数可动态调整）
        retriever = self.doc_manager.get_retriever(
            k=k,
            search_type=search_type,
            score_threshold=score_threshold,
            mmr_lambda=mmr_lambda,
        )

        # 检索相关文档
        docs = retriever.invoke(query)
        context = "\n\n".join(
            f"[来源: {d.metadata.get('source_file', '未知')}]\n{d.page_content}"
            for d in docs
        )

        # 格式化历史
        history_block = self._format_history(history)

        # 组装 chain 并执行（06 Chains + 05 Output Parsers）
        chain = self._prompt | self.llm | self._parser
        response = chain.invoke({
            "context": context,
            "history_block": history_block,
            "question": query,
        })

        return response
