"""
04_builtin_tools.py
LangChain Tools - 常见内置工具

除了自定义 @tool 之外，LangChain 社区提供了大量开箱即用的内置工具。
本文件演示 5 类最常见的内置工具，全部不需要 API Key。

参考文档：
  - Tools 概述: https://docs.langchain.com/oss/python/langchain/tools

安装：
  pip install langchain-community langchain-experimental ddgs wikipedia
"""

import warnings
warnings.filterwarnings("ignore")  # ShellTool 有安全警告，教学演示忽略


# ============================================================
# 演示 1：文件操作工具
# ============================================================

def demo_file_tools():
    """
    LangChain 提供了完整的文件操作工具集：
      - ListDirectoryTool：列出目录内容
      - ReadFileTool：读取文件内容
      - WriteFileTool：写入文件
      - CopyFileTool / MoveFileTool / DeleteFileTool：复制、移动、删除

    参数格式：{"file_path": "/path/to/file", "text": "内容"}
    """
    print(">>> 演示 1：文件操作工具")
    print("-" * 50)
    from langchain_community.tools.file_management import (
        ReadFileTool, WriteFileTool, ListDirectoryTool,
    )

    # ListDirectoryTool
    list_dir = ListDirectoryTool()
    print(f"ListDirectoryTool('.'):【 {list_dir.invoke({'dir_path': '.'})[:100]}】")

    # WriteFileTool
    write_file = WriteFileTool()
    write_file.invoke({"file_path": "agent_test.txt", "text": "Hello from LangChain!"})
    print("WriteFileTool: 已写入当前目录 agent_test.txt")

    # ReadFileTool
    read_file = ReadFileTool()
    content = read_file.invoke({"file_path": "agent_test.txt"})
    print(f"ReadFileTool: {content}")
    print()


# ============================================================
# 演示 2：Shell 执行工具
# ============================================================

def demo_shell_tool():
    """
    ShellTool 允许 Agent 执行任意 Shell 命令。
    参数格式：{"commands": ["ls", "date"]}
    
    ⚠️ ShellTool 没有安全沙箱，生产环境慎用！
    生产环境建议通过 Docker 容器或沙箱执行。
    """
    print(">>> 演示 2：Shell 执行工具")
    print("-" * 50)

    from langchain_community.tools.shell import ShellTool

    shell = ShellTool()
    result = shell.invoke({"commands": ["echo hello", "date", "uname -s"]})
    print(f"ShellTool:【 {result}】")
    print()


# ============================================================
# 演示 3：HTTP 请求工具
# ============================================================

def demo_requests_tool():
    """
    RequestsGetTool 发送 HTTP GET 请求，获取网页内容。
    适合让 Agent 访问 API、抓取网页。

    参数格式：{"url": "https://example.com"}

    ⚠️ 两个必传参数：
      - requests_wrapper: GenericRequestsWrapper的实例,如：TextRequestsWrapper（Pydantic required field，不传会报 ValidationError）
      - allow_dangerous_requests: True（安全检查，不传会报 ValueError）
    生产环境建议通过代理服务器使用，防止 SSRF 攻击。
    """
    print(">>> 演示 3：HTTP 请求工具")
    print("-" * 50)

    from langchain_community.tools.requests.tool import RequestsGetTool
    from langchain_community.utilities.requests import TextRequestsWrapper

    http_get = RequestsGetTool(
        requests_wrapper=TextRequestsWrapper(),
        allow_dangerous_requests=True,
    )
    result = http_get.invoke({"url": "https://www.boxuegu.com"})
    print(f"RequestsGetTool: {result[:150]}...")
    print()


# ============================================================
# 演示 4：搜索引擎工具
# ============================================================

def demo_search_tool():
    """
    DuckDuckGoSearchRun 使用 DuckDuckGo 搜索引擎。
    不需要 API Key，但需要安装 ddgs 包。

    参数格式：{"query": "搜索内容"}

    其他搜索工具：
      - TavilySearchResults（需要 API Key，质量更高）
      - BraveSearchRun（需要 API Key）
    """
    print(">>> 演示 4：搜索引擎工具")
    print("-" * 50)

    from langchain_community.tools import DuckDuckGoSearchRun

    search = DuckDuckGoSearchRun()
    result = search.invoke("langchain 最新版本是什么")
    print(f"DuckDuckGoSearchRun: {result[:200]}...")
    print()


# ============================================================
# 演示 5：知识查询工具
# ============================================================

def demo_wikipedia_tool():
    """
    WikipediaQueryRun 查询维基百科，获取结构化的知识内容。
    适合需要事实性知识的场景。

    参数格式：直接传字符串（查询内容）

    可配置项：
      - doc_content_chars_max：返回内容最大字符数
      - top_k_results：返回条目数量
    """
    print(">>> 演示 5：知识查询工具")
    print("-" * 50)

    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(doc_content_chars_max=300)
    )
    result = wiki.invoke("Python (programming language)")
    print(f"WikipediaQueryRun: {result[:300]}...")
    print()


# ============================================================
# 内置工具速查表
# ============================================================

BUILTIN_TOOLS_REFERENCE = """
部分内置工具速查（全部来自 langchain-community）：

┌─────────────────────────┬─────────────────┬──────────────────────────────────┐
│ 工具                     │ 类别             │ 额外依赖                         │
├─────────────────────────┼─────────────────┼──────────────────────────────────┤
│ DuckDuckGoSearchRun     │ 搜索引擎         │ pip install ddgs                │
│ TavilySearchResults      │ 搜索引擎         │ 需要 API Key                    │
│ WikipediaQueryRun       │ 知识查询         │ pip install wikipedia           │
│ ArxivQueryRun           │ 论文搜索         │ pip install arxiv               │
│ ReadFileTool            │ 文件操作         │ 无                              │
│ WriteFileTool           │ 文件操作         │ 无                              │
│ ListDirectoryTool       │ 文件操作         │ 无                              │
│ ShellTool               │ Shell 执行       │ pip install langchain-experimental│
│ RequestsGetTool         │ HTTP 请求        │ 无（需 allow_dangerous_requests）│
│ JsonListKeysTool        │ JSON 解析        │ 无                              │
│ JsonGetValueTool        │ JSON 解析        │ 无                              │
│ QuerySQLDatabaseTool    │ SQL 查询         │ 需要数据库连接                   │
│ GitHubTool              │ GitHub 操作      │ 需要 GitHub Token               │
│ SlackSendMessage        │ Slack 消息        │ 需要 Slack Token                │
└─────────────────────────┴─────────────────┴──────────────────────────────────┘

参考文档：
  - Tools 概述: https://docs.langchain.com/oss/python/langchain/tools
  - 安全注意事项: https://docs.langchain.com/oss/python/security-policy
  - 所有内置工具列表（学会从这找）: https://pypi.org/search/?q=langchain-community+tools
"""


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    demo_file_tools()
    demo_shell_tool()
    demo_requests_tool()
    demo_search_tool()
    demo_wikipedia_tool()

    print(BUILTIN_TOOLS_REFERENCE)
