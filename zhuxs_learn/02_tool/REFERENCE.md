# 02_tools Reference

## 环境核查

1. 根目录 `.env` 中已经存在 DeepSeek 相关变量，当前核查结果为：`DEEPSEEK_API_KEY=True`、`DEEPSEEK_BASE_URL=True`、`OPENAI_API_KEY=False`。
   核查命令：`set -a && source /workspace/langgraph-dev/.env && python -c "import os; keys=['DEEPSEEK_API_KEY','DEEPSEEK_BASE_URL','OPENAI_API_KEY']; print({k: bool(os.getenv(k)) for k in keys})"`
2. `langchain-deepseek==1.0.1` 已安装。
   核查命令：`source /workspace/langgraph-dev/.venv/bin/activate && uv pip show langchain-deepseek`
3. `langchain-mcp-adapters==0.2.1` 已安装。
   核查命令：`source /workspace/langgraph-dev/.venv/bin/activate && uv pip show langchain-mcp-adapters`
4. `fastmcp==3.1.1` 已安装。
   核查命令：`source /workspace/langgraph-dev/.venv/bin/activate && uv pip show fastmcp`
5. `ChatDeepSeek` 源码位于 `/workspace/langgraph-dev/.venv/lib/python3.13/site-packages/langchain_deepseek/chat_models.py`。

## 关键本地资料

1. LangChain tools 文档
   `/workspace/langgraph-dev/langchain_docs/src/oss/langchain/tools.mdx`
2. LangChain models 文档
   `/workspace/langgraph-dev/langchain_docs/src/oss/langchain/models.mdx`
3. LangChain MCP 文档
   `/workspace/langgraph-dev/langchain_docs/src/oss/langchain/mcp.mdx`
4. ChatDeepSeek 集成文档
   `/workspace/langgraph-dev/langchain_docs/src/oss/python/integrations/chat/deepseek.mdx`
5. ChatDeepSeek 源码
   `/workspace/langgraph-dev/.venv/lib/python3.13/site-packages/langchain_deepseek/chat_models.py`
6. `@tool` 转换实现
   `/workspace/langgraph-dev/.venv/lib/python3.13/site-packages/langchain_core/tools/convert.py`
7. tool 基类与 injected arg 实现
   `/workspace/langgraph-dev/.venv/lib/python3.13/site-packages/langchain_core/tools/base.py`
8. `StructuredTool` 实现
   `/workspace/langgraph-dev/.venv/lib/python3.13/site-packages/langchain_core/tools/structured.py`
9. `ToolNode` 实现
   `/workspace/langgraph-dev/libs/prebuilt/langgraph/prebuilt/tool_node.py`
10. `create_react_agent` 实现
    `/workspace/langgraph-dev/libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py`
11. `ToolRetryMiddleware` 实现
    `/workspace/langgraph-dev/.venv/lib/python3.13/site-packages/langchain/agents/middleware/tool_retry.py`
12. graph retry 实现
    `/workspace/langgraph-dev/libs/langgraph/langgraph/pregel/_retry.py`
13. `ToolNode` 注入测试
    `/workspace/langgraph-dev/libs/prebuilt/tests/test_tool_node.py`

## 当前已确认事实

1. `deepseek-chat` 支持 tool calling、structured output、native async；`deepseek-reasoner` 不支持 tool calling。
   来源：`langchain_docs/src/oss/python/integrations/chat/deepseek.mdx:28-33,72-83`
2. `ChatDeepSeek` 是 `BaseChatOpenAI` 子类，`bind_tools()` 额外暴露了 `parallel_tool_calls` 参数；`strict=True` 时会切到 beta endpoint，其余情况走父类实现。
   来源：`.venv/lib/python3.13/site-packages/langchain_deepseek/chat_models.py:31-70,373-415`
3. `@tool` 是最基础的 tool 定义方式，类型标注决定输入 schema，docstring 默认作为 description。
   来源：`langchain_docs/src/oss/langchain/tools.mdx:17-35`
4. `@tool` 在 `infer_schema=True` 或显式提供 `args_schema` 时，底层会转成 `StructuredTool.from_function(...)`；只有不推断 schema 且无 `args_schema` 时才退回简单 `Tool`。
   来源：`.venv/lib/python3.13/site-packages/langchain_core/tools/convert.py:303-325`
5. `StructuredTool` 是 `BaseTool` 的具体实现，能显式包装 sync / async 函数，并自动把 `RunnableConfig` 注入到带对应类型标注的参数中。
   来源：`.venv/lib/python3.13/site-packages/langchain_core/tools/structured.py:40-70,74-130,132-180`
6. `ToolRuntime` 是直接注入参数，属于 injected arg；注入参数不会出现在模型可见 schema 中。
   来源：`langchain_docs/src/oss/langchain/tools.mdx:163-173,217-254`；`.venv/lib/python3.13/site-packages/langchain_core/tools/base.py:587-609,1367-1468`
7. Pydantic `args_schema` 会在 Python 侧做实际校验并补默认值；JSON schema dict 路径则不会做同等级运行时校验，而且不允许字符串输入。
   来源：`.venv/lib/python3.13/site-packages/langchain_core/tools/base.py:676-757`
8. `bind_tools` 路径可以直接接受 `Pydantic class`、dict schema、LangChain tool、普通函数，本质上都是转成模型侧可见的 tool schema，并在每次模型调用时传入。
   来源：`langchain_docs/src/oss/python/integrations/chat/openai.mdx:243-267`；`.venv/lib/python3.13/site-packages/langchain_deepseek/chat_models.py:373-415`
9. `ToolNode` 文档明确负责 parallel tool execution、error handling、state injection。
   来源：`langchain_docs/src/oss/langchain/tools.mdx:564-567`
10. `ToolNode` 同步路径使用 `executor.map(...)`，异步路径使用 `asyncio.gather(...)`，并且会为每个 tool call 先构造独立的 `ToolRuntime`。
    来源：`libs/prebuilt/langgraph/prebuilt/tool_node.py:793-846`
11. `ToolNode` 同步线程池的并发度来自 `config["max_concurrency"]`。
    来源：`.venv/lib/python3.13/site-packages/langchain_core/runnables/config.py:588-603`
12. `create_react_agent(version="v1")` 会在一个 `ToolNode` 内并发执行当前消息中的多个 tool call；`version="v2"` 会把每个 tool call 拆成独立 `Send(...)` 分发。
    来源：`libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py:450-459,825-853`
13. `ToolNode` 默认错误处理并不是“所有异常都自动转成 `ToolMessage(status="error")`”。默认 handler 只把 `ToolInvocationError` 兜底成消息，其它异常继续抛出；`handle_tool_errors=True/str/callable/tuple` 才会扩大兜底范围。
    来源：`langchain_docs/src/oss/langchain/tools.mdx:637-654`；`libs/prebuilt/langgraph/prebuilt/tool_node.py:379-387,951-979`
14. tool retry 不是某个定义方式自带的统一能力，而是分层存在：`ToolRetryMiddleware` 包装单次 tool execution；Pregel `retry_policy` 包装 graph task。
    来源：`.venv/lib/python3.13/site-packages/langchain/agents/middleware/tool_retry.py:288-399`；`libs/langgraph/langgraph/pregel/_retry.py:26-217`
15. 仓库测试已经证明：plain function、`@tool`、显式 `args_schema` 三条路径都能与 `InjectedState`、`InjectedStore`、`ToolRuntime` 配合，前提是执行宿主能做注入。
    来源：`libs/prebuilt/tests/test_tool_node.py:1337-1502,1689-1904`
16. MCP 在 LangChain 中通过 `langchain-mcp-adapters` 接入；`MultiServerMCPClient` 默认是 stateless，每次调用都会创建新的 `ClientSession`。
    来源：`langchain_docs/src/oss/langchain/mcp.mdx:27-31,512-528`
17. MCP 支持 `streamable_http`；官方文档示例给出了 `FastMCP` 形式的 `streamable-http` server。
    来源：`langchain_docs/src/oss/langchain/mcp.mdx:200-212,399-430`
18. 远程 MCP tool 不能直接访问本地 graph runtime；若要读取 `state` / `context` / `store` 或做 retry / headers，需要通过 interceptor 在调用边界处理。
    来源：`langchain_docs/src/oss/langchain/mcp.mdx:745-759,765-925,1068-1097`

## 实测观察

1. `01_tool_decorator.py`、`02_structured_tool.py`、`03_basetool_subclass.py`、`04_pydantic_args_schema.py` 的并发实验耗时都在单次 `0.3s` 延迟附近，说明两次 `delay_http` 调用在 `ToolNode` 下确实并发执行，而不是串行两次 `0.3s + 0.3s`。
2. `01_tool_decorator.py` 实测表明：`ToolRuntime` 和 `InjectedState` 都不会出现在模型可见 schema 中，但在 `ToolNode` 执行时可被成功注入。
3. `03_basetool_subclass.py` 实测表明：`BaseTool` 子类可以工作，但如果 injected arg 不进入 `args_schema`，真实执行时会出现 `_run()` 缺参；另外 Pydantic v2 下类和输入模型都需要 `model_rebuild()` 才更稳。
4. `04_pydantic_args_schema.py` 实测表明：`bad_type`、`missing_field`、`extra_field` 都会在 Python 侧被 `DelayInput` 拦住，且 `Field(ge=0, le=3)` 会直接体现在模型侧 schema 上。
5. `04_pydantic_args_schema.py` 还暴露了一个当前版本细节：如果文件启用 `from __future__ import annotations`，`@tool(args_schema=...)` 路线下的 injected arg 识别会失效，导致 `runtime/topic` 没有被注入；去掉 future import 后恢复正常。
6. `05_bind_tools_schema_only.py` 实测表明：Pydantic class 和 JSON schema dict 都能让 DeepSeek 返回标准 `tool_calls`，但这条路径没有本地执行体，因此本地执行、错误处理、retry、state injection 都应记为 `N/A`。
7. `06_toolnode_host.py` 实测表明：
   - `handle_tool_errors=True` 会把异常包成默认错误消息
   - `handle_tool_errors="固定字符串"` 会直接返回固定内容
   - `handle_tool_errors=callable` 可以返回自定义错误消息
8. `06_toolnode_host.py` 还实测验证了 `InjectedState`、`ToolRuntime`、`InjectedStore` 三条注入链都由 `ToolNode` 在执行前完成。
9. `07_mcp_tool.py` 实测表明：`MultiServerMCPClient` 返回的 MCP tools 是可绑定到 DeepSeek 的 LangChain tools，但执行体在远端 server；本地 `state/store/context` 只能在 interceptor 边界读取和改写。
10. `07_mcp_tool.py` 还暴露了一个 MCP 特有细节：远端失败不一定抛异常，也可能返回 `CallToolResult(isError=True)`；因此 MCP retry interceptor 不能只 catch exception，还要检查 `isError`。

## 设计含义

1. “并发 / 顺序”必须同时记录定义方式与执行宿主，否则会把 `ToolNode` 的能力误记到 `@tool` 或 `StructuredTool` 头上。
2. “错误处理 / 重试”需要拆成三层观察：tool 自身、middleware / wrapper、graph runtime。
3. “State injection” 需要分别验证：
   - 是否能声明 injected arg
   - 是否对模型隐藏
   - 是否只有在特定宿主下才会真正注入成功
4. `bind_tools(Pydantic class)` 必须作为独立对照组，因为它更像 schema 暴露路径，不等价于一个本地可执行 tool。
5. MCP 必须单独看，因为它把“工具函数体”移到了远端 server，State injection 只能在客户端 interceptor 边界处理。
