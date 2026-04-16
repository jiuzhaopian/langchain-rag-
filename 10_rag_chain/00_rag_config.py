"""
00_rag_config.py - RAG 全局配置

集中管理模型、Embedding、向量库、切分参数等配置。
修改默认值即可全局生效，无需改动业务代码。
"""

from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
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
# Prompt 模板
# ============================================================

RAG_PROMPT_TEMPLATE = """基于以下检索到的文档内容回答用户的问题。
如果文档中没有相关信息，请明确告知「检索到的文档中未找到相关信息」，不要编造。

检索到的文档：
{context}

历史对话：
{history}

用户问题：{question}

请用中文回答："""


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
