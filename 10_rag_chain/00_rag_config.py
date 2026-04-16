"""
00_rag_config.py - RAG 全局配置

集中管理模型、Embedding、向量库、切分参数等配置。
修改默认值即可全局生效，无需改动业务代码。
"""

from pathlib import Path
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatZhipuAI

# ============================================================
# 路径配置
# ============================================================

# Chroma 持久化目录（重启不丢数据）
CHROMA_PERSIST_DIR = Path(__file__).parent / "data" / "chroma_db"

# ============================================================
# 模型配置
# ============================================================

# LLM（复用智谱 glm-4.7）
LLM_MODEL = "glm-4.7"
LLM_TEMPERATURE = 0.3  # RAG 场景用低温度，减少幻觉

# Embedding（Ollama 本地 bge-m3）
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_BASE_URL = "http://localhost:11434"

# ============================================================
# 文档处理默认参数
# ============================================================

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# ============================================================
# 检索默认参数
# ============================================================

DEFAULT_K = 3
DEFAULT_SEARCH_TYPE = "similarity"  # similarity | mmr | similarity_score_threshold
DEFAULT_SCORE_THRESHOLD = 0.5  # search_type=similarity_score_threshold 时生效
DEFAULT_MMR_LAMBDA = 0.5  # search_type=mmr 时生效

# ============================================================
# 对话配置
# ============================================================

DEFAULT_MAX_HISTORY_ROUNDS = 50  # 保留最近 N 轮对话历史

# ============================================================
# Chroma 向量库配置
# ============================================================

CHROMA_COLLECTION_NAME = "rag_docs"

# bge-m3（以及 OpenAI、DashScope 等主流模型）输出的向量是归一化的（模长=1）。
# 归一化向量下 L2 和 cosine 排序完全等价（L2² = 2(1-cos)），TopK 结果一模一样，
# 但分数值不同：
#   cosine 下：分数 = 余弦相似度（如 0.6），直观易懂
#   L2 下：经公式转换后分数被压缩（如 0.37），"相关文档才 0.37" 容易困惑
# 因此显式指定 cosine，让分数即可解释、又好看。
CHROMA_COLLECTION_METADATA = {"hnsw:space": "cosine"}


# ============================================================
# 工厂函数
# ============================================================

def get_llm():
    """获取 LLM 实例（智谱 glm-4.7）"""
    return ChatZhipuAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )


def get_embeddings():
    """获取 Embedding 实例（Ollama bge-m3）"""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=EMBEDDING_BASE_URL,
    )
