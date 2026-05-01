# LangChain + vLLM 工具调用最佳实践指南

本文基于 2026-05-01 当前 `langchain_vllm_tool_call_experiments.py` 的实际 pytest 结果整理，目标是给后续使用本地 vLLM 作为 OpenAI-compatible 后端时，提供一套可复现的工具调用实践。

相关文件：

- 实验脚本：`zhuxs_learn/03_tool/langchain_vllm_tool_call_experiments.py`
- 异步 MCP server：`zhuxs_learn/03_tool/mcp_async_tool_server.py`
- streamable HTTP MCP 示例 server：`zhuxs_learn/03_tool/http_mcp_server.py`
- streamable HTTP MCP 调用示例：`zhuxs_learn/03_tool/http_mcp_call_example.py`
- 抓包与汇总结果：`zhuxs_learn/03_tool/captures_tool_call/`
- vLLM 官方工具调用资料：`zhuxs_learn/03_tool/vllm_tool_calling.md`
- LangChain 工具调用资料：`zhuxs_learn/03_tool/langchain_tool_call.md`
- LangChain MCP 文档：`/data/zhuxs/llm_agent/langchain_docs/build/oss/python/langchain/mcp.mdx`
- LangChain Tools 文档：`/data/zhuxs/llm_agent/langchain_docs/build/oss/python/langchain/tools.mdx`

当前 pytest 实验清单：

| 编号 | 测试函数 | 当前状态 | 主要产物 |
| --- | --- | --- | --- |
| 01 | `test_01_openai_sdk_and_langchain_request_body_alignment` | 实测运行 | SDK/LangChain 四种 tool_choice 抓包 |
| 02 | `test_02_langchain_auto_can_use_tool_description` | 实测运行 | description auto 路由抓包 |
| 03 | `test_03_langchain_tool_parameter_description_visible_to_llm` | 实测运行 | 参数 description 抓包 |
| 04 | `test_04_langchain_vllm_tool_choice_strong_modes` | 实测运行 | required/none/named 强模式抓包 |
| 05 | `test_05_lightweight_llm_tool_external_context_injection` | 实测运行 | `InjectedToolArg` 隐藏参数抓包 |
| 06 | `test_06_traditional_toolnode_runtime_error_and_parallel` | 实测运行 | 汇总 JSON，验证 ToolNode runtime/Command/error/parallel |
| 07 | `test_07_custom_tool_execution_node_runtime_command_error_order` | 实测运行 | 汇总 JSON，验证 custom node runtime/Command/error/order |
| 08 | `test_08_runtime_hidden_minimal_graph_with_toolnode_and_custom_node` | 实测运行 | ToolNode/custom node 两条最小图抓包 |
| 09 | `test_09_async_mcp_toolnode_compatibility` | 实测运行 | MCP schema、runtime 隐藏注入抓包、错误/并行；HTTP 示例拆到独立文件 |
| 10 | `test_10_langgraph_tool_loop_reserved` | 占位 | 仅写入 reserved 汇总，不运行 agent loop |
| 11 | `test_11_langchain_args_schema_required_parameter_over_prompt` | 实测运行 | required 参数压过 prompt 的抓包 |

## 一、最终结论

1. OpenAI SDK 与 LangChain 可以向 vLLM 发出一致的工具调用请求。关键是复用同一份工具 schema，并显式传入 `tool_choice`、`stream=False`、采样参数和 vLLM `extra_body`。
2. vLLM 支持 `tool_choice="auto"`、`"required"`、`"none"` 和命名函数调用。`auto` 依赖 parser 从模型文本中提取工具调用，不做 schema constrained decoding；需要参数结构强约束时，优先使用 `required` 或指定函数。
3. `tools[].function.description` 和 `tools[].function.parameters.properties.<param>.description` 都会进入 LangChain 发给 vLLM 的请求体。`auto` 模式下模型可以利用工具 description 做工具选择；参数 description 也能影响模型填参。
4. 工具参数 schema 里的 `required` 来自 Pydantic/函数签名的必填字段：无默认值就是必填，有默认值就是可选。它不是 `tool_choice="required"`，但在 vLLM 的命名函数/required 工具调用模式下会参与结构化参数生成约束。
5. `required`、`none`、命名函数在 vLLM 中是明确模式，不是建议。正常完成时，`required` 会强制返回一个或多个 tool calls，命名函数会强制返回指定函数，`none` 不会在 API 结果里产生 `tool_calls`。
6. LangChain 的 `ChatOpenAI.bind_tools(..., tool_choice="any")` 会归一化为 OpenAI 的 `required`。为了抓包和 SDK 对齐，建议直接写 `required`。
7. ToolNode 是当前自定义工具执行流程的推荐基础组件。它负责并行执行、runtime 注入、错误处理和 `Command` 状态更新校验。
8. ToolNode 的 wrapper 钩子是单个 tool call 级别；如果要控制多个工具调用的全局顺序、条件跳过、批量事务或严格串行，建议写逻辑更直接的 custom tool execution node。
9. `ToolRuntime` 是隐藏注入参数，不会出现在 LLM 可见的 tool schema 中。实验已用 ToolNode 和 custom node 两条最小 LLM -> 工具 -> LLM 图抓包证明。
10. `ToolRuntime` 的自动构造和注入是 LangGraph runtime 能力；不用 LangGraph 时，可以用 `InjectedToolArg` 隐藏外部变量参数，并在应用层执行工具前手动注入。
11. MCP 工具可以兼容 ToolNode，但 MCP server 本身拿不到 LangGraph runtime。需要通过 `langchain-mcp-adapters` 的 async interceptor 从 `request.runtime` 读取 state/context/store，再注入到 MCP 工具参数或 HTTP headers。若这些注入参数只应由后端填充，发给 LLM 前要隐藏对应 schema 字段。
12. MCP 最大价值之一是异步和进程隔离。需要 runtime 注入和并行性能验证时使用持久 `ClientSession`；无状态场景可以使用 streamable HTTP，每次调用不依赖 LangGraph state/context。

## 二、请求体对齐实践

实验覆盖四组请求：

| 模式 | SDK 写法 | LangChain 写法 | 实测结果 |
| --- | --- | --- | --- |
| 自动选择 | `tool_choice="auto"` | `tool_choice="auto"` | 请求体一致，一次返回 `get_weather` 和 `get_population` 两个 tool calls |
| 必选工具 | `tool_choice="required"` | `tool_choice="required"` | 请求体一致，返回 `get_weather` |
| 禁用工具 | `tool_choice="none"` | `tool_choice="none"` | 请求体一致，未返回 tool calls |
| 命名函数 | `{"type": "function", "function": {"name": "get_weather"}}` | `tool_choice="get_weather"` | 请求体一致，LangChain 自动转成 OpenAI dict |

最佳实践：

```python
from langchain_core.utils.function_calling import convert_to_openai_tool

formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
```

OpenAI SDK 侧直接传 `formatted_tools`。LangChain 侧传同一组 `@tool` 对象给 `bind_tools`。抓包时比较这些字段：

- `model`
- `messages`
- `tools`
- `tool_choice`
- `stream`
- `temperature`
- `top_p`
- `repetition_penalty`
- `presence_penalty`
- `top_k`
- `min_p`
- `chat_template_kwargs`

实测抓包均已保存：

- `captures_tool_call/01_auto_openai_sdk_capture.json`
- `captures_tool_call/01_auto_langchain_capture.json`
- `captures_tool_call/01_required_openai_sdk_capture.json`
- `captures_tool_call/01_required_langchain_capture.json`
- `captures_tool_call/01_none_openai_sdk_capture.json`
- `captures_tool_call/01_none_langchain_capture.json`
- `captures_tool_call/01_named_openai_sdk_capture.json`
- `captures_tool_call/01_named_langchain_capture.json`

汇总结果见 `captures_tool_call/00_tool_call_experiment_results.json`。

## 三、vLLM 注意事项

vLLM 侧需要启动时启用自动工具选择和匹配模型的 parser/template。例如 Qwen 系列通常参考 vLLM 文档使用 Hermes 风格 parser。

核心差异：

- `auto`：模型自由生成，vLLM parser 解析工具调用，不保证参数满足 JSON Schema。
- `required`：保证至少生成一个工具调用，并通过结构化输出后端约束参数对象。
- 命名函数：强制某个函数，参数通过结构化输出后端约束。
- `none`：禁止返回工具调用，但 vLLM 默认仍可能把 tools 注入 prompt。若需要彻底排除 tools，需要服务端启动参数 `--exclude-tools-when-tool-choice-none`。
- `strict=True`：vLLM 当前接受字段，但不要按 OpenAI strict semantic 理解；尤其 `auto` 下不能依赖 strict 保证参数合法。

实践建议：

- 业务上必须调用工具时用 `required`。
- 业务上必须调用某个具体工具时用命名函数。
- 探索或 agent 自主选择时用 `auto`，但工具参数要在应用层校验。
- 并行工具调用分两层验证：`auto` 请求中确认模型一次返回多个 tool calls，再确认 ToolNode 执行这些 calls 时并行。

工程例外：这些强模式不是数学意义的“任何异常下 100% 成功”。如果输出被 `max_tokens` / `max_completion_tokens` 截断，服务端可能返回未完成响应而没有可解析 tool call。这里的结论应理解为“正常完成时的 API/解码层约束”。

本地实测还暴露了一个容易误读的点：`tool_choice="none"` 时，LangChain 响应对象没有 `tool_calls`，但 Qwen 模板下 `content` 里可能出现字面量 `<tool_call>...</tool_call>` 文本。这不是 API 层 tool call；应用侧应以解析后的 `AIMessage.tool_calls` 为准。若业务要求模型连这类文本格式都不要看到，应在 vLLM 服务端启用 `--exclude-tools-when-tool-choice-none`。

## 四、description 与强模式实验

### 4.1 description 可见性

实验函数：`test_02_langchain_auto_can_use_tool_description`

目的：回答“模型能否看到 tools 里的 description”。实验没有使用天气/人口这种函数名自带语义的工具，而是定义两个参数 schema 完全相同、函数名没有业务含义的工具：

- `route_alpha(city: str)`：description 写明“当用户给出暗号`青铜钥匙`时调用”。
- `route_beta(city: str)`：description 写明“当用户给出暗号`银色钥匙`时调用”。

这两个工具都使用 Pydantic `args_schema` 给 `city` 参数加了 `Field(description=...)`：

```python
class DescriptionProbeInput(BaseModel):
    city: str = Field(
        description=(
            "参数级 description 可见性探针：city 是工具要处理的城市名称。"
            "本实验要求填写中文城市名，例如“北京”。"
        )
    )

@tool(args_schema=DescriptionProbeInput)
def route_alpha(city: str) -> str:
    """当用户给出暗号“青铜钥匙”时，必须调用本工具处理该城市。"""
    ...
```

方法：使用 `ChatOpenAI(...).bind_tools([...], tool_choice="auto")`，prompt 只给暗号和城市，不出现函数名。抓包先证明工具级 description 和参数级 description 都被发送到 vLLM，再断言返回的 `tool_calls` 与 description 规则一致。

实测结果：

| case | prompt 暗号 | 期望工具 | 实际工具 |
| --- | --- | --- | --- |
| `bronze_keyword` | `青铜钥匙` | `route_alpha` | `route_alpha({"city": "北京"})` |
| `silver_keyword` | `银色钥匙` | `route_beta` | `route_beta({"city": "北京"})` |

抓包：

- `captures_tool_call/02_description_auto_bronze_keyword_capture.json`
- `captures_tool_call/02_description_auto_silver_keyword_capture.json`

结论：抓包证明 LangChain 确实把 `tools[].function.description` 传给 vLLM；行为证明 `auto` 模式下模型会利用 description 做工具选择。严格说，这证明的是“description 被发送且影响可观测选择行为”，不是哲学意义上证明模型理解了 description。

### 4.2 参数 description 可见性

实验函数：`test_03_langchain_tool_parameter_description_visible_to_llm`

目的：单独证明 `tools[].function.parameters.properties.<param>.description` 也能被 LLM 看到，而不只是工具 docstring 能被看到。

方法：定义 `parameter_description_probe(city: str)`，工具 docstring 不包含“北京”，只有 `city` 参数的 `Field(description=...)` 写明“无论用户正文提到哪个城市，这个 city 参数都必须填写`北京`”。prompt 中故意给出迷惑城市“上海”，并要求模型按参数 schema 的 description 填参。

实测抓包：

```json
{
  "city": {
    "description": "参数级 description 行为探针：无论用户正文提到哪个城市，这个 city 参数都必须填写“北京”。",
    "type": "string"
  }
}
```

实测 tool call：

```json
{
  "name": "parameter_description_probe",
  "arguments": {"city": "北京"}
}
```

抓包：`captures_tool_call/03_parameter_description_probe_capture.json`

结论：参数 description 确实进入 vLLM 请求体，且本实验中影响了模型对工具参数的填写。生产上仍建议对参数做应用层校验，因为 `auto` 模式不提供 schema constrained decoding。

### 4.3 required / none / named 是强模式

实验函数：`test_04_langchain_vllm_tool_choice_strong_modes`

目的：把 prompt 和 `tool_choice` 故意设成冲突，证明强模式不是建议。

| case | prompt 意图 | `tool_choice` | 实测结果 |
| --- | --- | --- | --- |
| `required_against_prompt` | 明确要求不要用工具 | `required` | 仍返回 `get_weather` 和 `get_population` 两个 tool calls |
| `none_against_prompt` | 明确要求调用 `get_weather` | `none` | `AIMessage.tool_calls == []` |
| `named_against_prompt` | 明确要求调用 `get_weather` | `get_population` | 返回 `get_population({"city": "北京"})` |

抓包：

- `captures_tool_call/04_strong_required_against_prompt_langchain_capture.json`
- `captures_tool_call/04_strong_none_against_prompt_langchain_capture.json`
- `captures_tool_call/04_strong_named_against_prompt_langchain_capture.json`

工程解释：

- `required`：正常完成时强制至少一个工具调用，即使 prompt 语义上不需要工具。
- `none`：正常完成时 API 结果不产生 tool calls，即使 prompt 要求用工具。
- named：强制指定函数名，prompt 不能改掉它；但参数质量仍取决于模型和 schema 约束，不应过度断言语义正确性。
- `auto`：唯一真正让模型自己决定是否调用、调用哪个工具的模式。

### 4.4 args_schema 的 required 是否压过 prompt

实验函数：`test_11_langchain_args_schema_required_parameter_over_prompt`

目的：解释抓包中 `tools[].function.parameters.required` 的来源，以及它在 vLLM 命名函数调用中到底是硬约束还是软提示。

这个 `required` 是 JSON Schema 的参数必填列表，不是 `tool_choice="required"`。LangChain 会根据 Pydantic `args_schema` 生成它：

- 字段没有 default：进入 `required`。
- 字段有 default：不进入 `required`，模型可以省略。
- `str | None` / `Optional[str]` 只表示允许值为 null；在 Pydantic v2 中，如果没有 `default=None`，它仍然是必填字段。

实验写法：

```python
class RequiredParameterProbeInput(BaseModel):
    city: str = Field(
        description="必填参数示例：没有 default，因此会进入 required。"
    )
    note: str = Field(
        description=(
            "必填冲突参数示例：没有 default，因此会进入 required。"
            "本实验会在 prompt 中要求模型不要填写 note。"
        )
    )
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="可选参数示例：有 default，因此不会进入 required。"
    )

@tool(args_schema=RequiredParameterProbeInput)
def required_parameter_probe(
    city: str,
    note: str,
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> str:
    ...
```

prompt 故意要求：“请不要填写 note 参数”。命名函数调用仍返回了 `note`。

实测抓包中的参数 schema：

```json
{
  "required": ["city", "note"],
  "properties": {
    "city": {"type": "string"},
    "note": {"type": "string"},
    "unit": {"default": "celsius", "enum": ["celsius", "fahrenheit"]}
  }
}
```

抓包：`captures_tool_call/11_required_parameter_over_prompt_capture.json`

工程解释：

- 在 vLLM 的命名函数调用正常完成时，参数 schema 的 `required` 会参与结构化参数生成约束；本实验中它压过了“不要填写 note”的 prompt。
- 这个结论针对“工具已经被强制调用指定函数”的参数生成阶段，不等同于 `tool_choice="required"` 的“必须调用某个工具”。
- 在 `tool_choice="auto"` 下，vLLM 主要依赖模型文本和 parser 抽取工具调用，schema 会作为提示和解析参考，但不应当作严格 schema constrained decoding。
- optional 的含义是“允许省略”，不是“禁止输出”。模型仍可能输出 `unit`，应用层应接受并校验。

## 五、ToolNode 最佳实践

推荐导入：

```python
from langchain.tools import ToolRuntime, tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
```

工具访问运行时上下文：

```python
@tool
def inspect_runtime(query: str, runtime: ToolRuntime[ContextT, StateT]) -> str:
    state = runtime.state
    user_id = runtime.context.user_id
    tool_call_id = runtime.tool_call_id
    return f"{query}: {user_id}: {tool_call_id}: {len(state['messages'])}"
```

工具更新状态：

```python
@tool
def record_event(event: str, runtime: ToolRuntime[ContextT, StateT]) -> Command:
    return Command(
        update={
            "messages": [
                ToolMessage(content="ok", tool_call_id=runtime.tool_call_id or "")
            ],
            "tool_events": [event],
        }
    )
```

注意：工具返回当前图的 `Command(update=...)` 时，`messages` 更新里应包含匹配当前 `tool_call_id` 的 `ToolMessage`。否则后续模型调用会遇到“assistant tool_calls 没有对应 tool message”的历史消息错误。

确定性测试方式：

```python
state = {
    "messages": [
        HumanMessage(content="fixture"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "inspect_runtime", "args": {"query": "x"}, "id": "call-1", "type": "tool_call"},
                {"name": "record_event", "args": {"event": "ok"}, "id": "call-2", "type": "tool_call"},
            ],
        ),
    ],
    "tool_events": [],
}
```

再将 `ToolNode` 放入 `StateGraph`，用 `graph.ainvoke(..., context=...)` 执行。这能验证真实 runtime 注入，而不是只调用 Python 函数。

已实现测试函数：

- `test_06_traditional_toolnode_runtime_error_and_parallel`
  - 目的：验证传统 `@tool` 与 ToolNode 的 runtime 注入、`Command` 状态写回、错误处理和异步并行执行。
  - 方法：手工构造 `AIMessage.tool_calls`；两个 `slow_echo(1.2s)` 同时放入一条 AIMessage。
  - 结果：runtime/context/tool_call_id 正常读取；`tool_events` 写回 `runtime_checked`；错误转成 `ToolMessage(status="error")`；两个 1.2s 慢工具总耗时约 1.204s。

## 六、create_agent 与 ToolRuntime 隐藏参数

LangChain 的 `create_agent` 本质上返回一个已编译的 LangGraph 图。根据当前安装源码 `/usr/local/lib/python3.12/site-packages/langchain/agents/factory.py`，它的核心循环可以简化为：

```text
START
  -> before_agent? -> before_model? -> model
  -> after_model?
  -> 如果最后一条 AIMessage 有 pending tool_calls: tools
  -> tools 追加 ToolMessage
  -> before_model? / model
  -> 重复，直到没有 pending tool_calls
  -> after_agent? / END
```

更接近最小手写图的形式：

```text
START -> llm
llm --tools_condition==tools--> tools
tools -> llm
llm --tools_condition==__end__--> END
```

`ToolRuntime` 的隐藏发生在 schema 生成阶段：

- `ToolRuntime` 继承 LangChain Core 的 directly injected argument 类型。
- `BaseTool.tool_call_schema` 会排除 injected 参数，因此 `convert_to_openai_tool(inspect_runtime)` 只包含 `query`，不包含 `runtime`。
- `ToolNode` 执行工具前构造 `ToolRuntime(state, context, config, stream_writer, tool_call_id, store, ...)`，再把它注入工具参数。
- `ToolNode` 会先剥离 LLM 传入的 injected 参数，再注入可信值，防止模型伪造 hidden 参数。

实验函数：`test_08_runtime_hidden_minimal_graph_with_toolnode_and_custom_node`

方法：同一个最小图分别接入 ToolNode 和 custom node。第一轮 LLM 使用命名函数强制调用 `inspect_runtime`；抓包检查请求体工具 schema；工具执行后返回包含 `user_id`、`session_label`、`tool_call_id` 的 JSON；第二轮 LLM 用 `tool_choice="none"` 基于 ToolMessage 输出最终回答。

实测抓包：

- `captures_tool_call/08_runtime_hidden_toolnode_graph_capture.json`
- `captures_tool_call/08_runtime_hidden_custom_graph_capture.json`

两条路径第一轮请求里的 `inspect_runtime` schema 都是：

```json
{
  "name": "inspect_runtime",
  "parameters": {
    "properties": {
      "query": {"type": "string"}
    },
    "required": ["query"],
    "type": "object"
  }
}
```

结论：LLM 只能看到 `query`，看不到也不能填写 `runtime`；但工具执行时仍能读取 `runtime.state` 和 `runtime.context`。

## 七、不用 LangGraph 的轻量上下文注入

`ToolRuntime` 这条自动注入路径是 LangGraph runtime 的能力。换句话说：如果你只是 `llm.bind_tools(...)`，没有 `StateGraph` / `ToolNode` / `create_agent`，LangChain 不会自动给工具构造 `ToolRuntime`。

但这不代表必须上 LangGraph。普通 `llm with tools` 有两种轻量做法：

1. 外部变量固定于本次请求：用闭包或工具工厂创建工具，工具函数内部直接读取外部变量。
2. 外部变量需要显式注入但不暴露给 LLM：用 `InjectedToolArg` 标记隐藏参数，LLM 只生成可见参数，应用层执行工具前把隐藏参数合并进去。

本仓库验证了第二种方式。

实验函数：`test_05_lightweight_llm_tool_external_context_injection`

工具定义：

```python
from typing import Annotated
from langchain_core.tools import InjectedToolArg

@tool
def lightweight_account_lookup(
    query: Annotated[str, "用户想查询的当前账户资料主题，例如 language 或 plan。"],
    account_id: Annotated[str, InjectedToolArg],
) -> str:
    """Look up deterministic private account data for the current account."""
    ...
```

LLM 看到的 schema 只有：

```json
{
  "properties": {
    "query": {
      "description": "用户想查询的当前账户资料主题，例如 language 或 plan。",
      "type": "string"
    }
  },
  "required": ["query"],
  "type": "object"
}
```

应用层执行工具时手动注入：

```python
first_ai = await llm_with_tool.ainvoke(messages)
call = first_ai.tool_calls[0]

injected_call = {
    **call,
    "args": {
        **call["args"],
        "account_id": external_context["account_id"],
    },
    "type": "tool_call",
}

tool_message = await lightweight_account_lookup.ainvoke(injected_call)
```

实测结果：

- 抓包里没有 `account_id` 参数。
- LLM 生成的 tool call 是 `lightweight_account_lookup({"query": "language"})`。
- 工具返回值中包含应用层注入的 `account_id="account-hidden-42"`。

抓包：`captures_tool_call/05_lightweight_external_context_injection_capture.json`

工程建议：如果只是少量外部变量，`InjectedToolArg + 手写执行器` 足够轻；如果你还需要多轮循环、错误反馈、并行工具调度、状态 reducer、checkpoint、store 或可恢复执行，再使用 LangGraph。

## 八、自定义工具执行 node

ToolNode 的默认行为适合大多数场景：它并发执行同一条 `AIMessage` 中的 tool calls，并提供 runtime 注入、错误处理和 Command 校验。它也支持 `wrap_tool_call` / `awrap_tool_call`，可用于单个工具调用的鉴权、缓存、参数改写、重试或短路。

边界在于：wrapper 拿到的是单个 `ToolCallRequest`，不是整个 tool call 批次。它不能可靠表达“先执行 A，再根据 A 的结果决定是否执行 B/C”这类全局计划，因为其他 tool call 可能已经被并发调度。

需要 custom node 的场景：

- 多个工具调用有依赖关系，必须严格排序。
- 需要根据一批 tool calls 做全局过滤、条件跳过或优先级排序。
- 需要批量事务、统一审计、统一限流或批量前后处理。
- 需要明确控制顺序、是否并行、是否跳过某些工具，而不是接受 ToolNode 的默认并发策略。

本仓库实现的 custom node 不追求兼容 ToolNode 的所有输入输出格式，而是保持执行逻辑清晰：

- 输入固定为本实验图的 `ExperimentState`。
- 从最后一条 `AIMessage.tool_calls` 读取整批工具调用。
- 在执行前做全局排序，例如先执行 `inspect_runtime`，再执行 `record_runtime_event`。
- 手动剥离 LLM 可能伪造的 hidden args，再注入可信 `ToolRuntime`。
- 普通工具结果写回 `messages`；工具返回 `Command(update=...)` 时转成普通 state update 合并。
- 工具异常转成 `ToolMessage(status="error")`，除非显式关闭错误处理。

实验函数：`test_07_custom_tool_execution_node_runtime_command_error_order`

实测结果：

- `inspect_runtime` 可读取 `user_id="custom-user"`。
- `record_runtime_event` 可通过 `Command(update=...)` 写回 `tool_events=["custom_runtime_checked"]`。
- `fail_on_purpose` 被转成 `ToolMessage(status="error")`。
- 输入中故意把 `record_runtime_event` 放在 `inspect_runtime` 前面，custom node 仍按 `["inspect_runtime", "record_runtime_event"]` 顺序执行。

## 九、错误处理实践

ToolNode 的 `handle_tool_errors` 支持：

- `True`：捕获所有普通异常并返回错误 `ToolMessage`。
- 固定字符串：用统一错误文案返回给模型。
- 异常类型：只捕获指定异常。
- tuple：捕获多个指定异常。
- callable：自定义异常到字符串的转换。
- `False`：不捕获，直接抛出。

建议分层处理：

- 参数缺失、类型错误：保留默认或 `True`，让模型有机会修正参数。
- 业务执行错误：生产环境建议返回简洁、可恢复的错误信息，不要把内部堆栈暴露给模型。
- 不可恢复错误：让异常抛出，由外层监控/重试系统处理。

实验中：

- 传统工具 `fail_on_purpose` 抛出 `ValueError`。
- MCP 工具 `always_fail` 抛出 `ValueError`，adapter 转成 `ToolException`，再由 `ToolNode(handle_tool_errors=True)` 转成 `ToolMessage(status="error")`。

## 十、MCP 工具实践

MCP server 示例见 `mcp_async_tool_server.py`，工具均为 async：

- `get_weather`
- `get_population`
- `slow_echo`
- `context_probe`
- `always_fail`

加载 MCP 工具：

```python
client = MultiServerMCPClient(
    {
        "fixture": {
            "transport": "stdio",
            "command": "python",
            "args": [str(MCP_SERVER_PATH)],
        }
    },
    tool_interceptors=[inject_runtime_context],
)
```

runtime 桥接通过 interceptor 完成：

```python
async def inject_runtime_context(request: MCPToolCallRequest, handler):
    runtime = request.runtime
    if request.name == "context_probe" and runtime is not None:
        request = request.override(
            args={
                **request.args,
                "user_id": runtime.context.user_id,
                "session_label": runtime.state.get("session_label", ""),
            }
        )
    return await handler(request)
```

注意：如果 MCP server 的函数签名显式声明 `user_id` / `session_label` 来接收后端注入，那么原始 MCP schema 会包含这些字段。当前实验在发送给 LLM 前复制 MCP tool 并删除这些字段，只保留 `query`，工具执行时仍由 interceptor 注入真实值。

LLM 可见的 `context_probe` schema 抓包结果：

```json
{
  "properties": {
    "query": {
      "description": "要读取的运行时上下文主题。",
      "type": "string"
    }
  },
  "required": ["query"],
  "type": "object"
}
```

抓包：`captures_tool_call/09_mcp_runtime_hidden_llm_graph_capture.json`

工具执行结果仍能拿到后端注入值：

```json
{
  "query": "隐藏参数验证",
  "session_label": "mcp-llm-hidden-session",
  "user_id": "mcp-llm-user"
}
```

注意：上面这个工具结果回显 `user_id/session_label` 只是为了测试证明 interceptor 注入成功。生产工具不应该把这类后端注入的隐藏参数写进 `ToolMessage` 再交给 LLM；生产返回值应只包含业务上允许模型看到的信息。

并行性能和 runtime 注入测试使用持久 session：

```python
async with client.session("fixture") as session:
    tools = await load_mcp_tools(
        session,
        tool_interceptors=[inject_runtime_context],
        server_name="fixture",
    )
```

不要用默认无状态 `client.get_tools()` 来判断 stdio MCP 工具并行耗时，因为每次工具调用会新建 session 和子进程，启动成本会污染结果。

MCP 参数 schema 也可以表达 required、default 和 description。server 端使用 `Annotated[..., Field(description=...)]`，并通过默认值区分必填/可选：

```python
@mcp.tool()
async def get_weather(
    city: Annotated[str, Field(description="城市名称，例如“北京”或“上海”。")],
    unit: Annotated[
        Literal["celsius", "fahrenheit"],
        Field(description="温度单位。默认 celsius，可选 fahrenheit。"),
    ] = "celsius",
) -> str:
    ...
```

实测加载到 LangChain 后的 OpenAI tool schema：

```json
{
  "required": ["city"],
  "properties": {
    "city": {"description": "城市名称，例如“北京”或“上海”。", "type": "string"},
    "unit": {
      "default": "celsius",
      "description": "温度单位。默认 celsius，可选 fahrenheit。",
      "enum": ["celsius", "fahrenheit"],
      "type": "string"
    }
  }
}
```

最小 streamable HTTP 无状态示例已从主 pytest 文件拆出，避免端口、
子进程和轮询逻辑干扰 `test_09` 的 MCP runtime/ToolNode 主线。独立验证方式：

```bash
# 终端 A
python zhuxs_learn/03_tool/http_mcp_server.py

# 终端 B
python zhuxs_learn/03_tool/http_mcp_call_example.py
```

调用示例里的连接方式：

```python
client = MultiServerMCPClient(
    {
        "http_fixture": {
            "transport": "http",
            "url": "http://127.0.0.1:8000/mcp",
        }
    }
)
```

`http_mcp_server.py` 使用 `mcp.run(transport="streamable-http", stateless_http=True)`，
只暴露一个最小 `get_weather` 工具；`http_mcp_call_example.py` 只负责加载工具、
打印 schema 和一次确定性调用结果。这个示例不使用 LangGraph runtime，也不使用
interceptor，只作为和持久 stdio runtime 注入路径的对照。

已实现测试函数：

- `test_09_async_mcp_toolnode_compatibility`
  - 目的：验证异步 MCP 工具和传统工具在 ToolNode 下能力一致。
  - 方法：测试函数内直接构造 `MultiServerMCPClient`；持久 stdio session 加载工具；检查参数 schema 的 required/default/description；跑 start -> llm -> ToolNode -> llm -> end 最小图并抓包证明隐藏参数不发给 LLM；对比 `get_weather`/`get_population` 的稳定 JSON；验证错误处理和两个 async 慢工具并行；在汇总结果中标注 streamable HTTP 独立示例的验证命令。
  - 结果：MCP 加载工具 `always_fail`、`context_probe`、`get_population`、`get_weather`、`slow_echo`；参数 schema 中 required/default/description 完整；LLM 请求中的 `context_probe` 只包含 `query`，不包含 `user_id/session_label`；工具执行时 runtime 注入得到 `user_id="mcp-llm-user"` 和 `session_label="mcp-llm-hidden-session"`。这两个字段被回显只是测试探针行为，生产工具不要把隐藏注入参数返回给 LLM；传统/MCP 结果一致；两个 1.2s MCP 慢工具总耗时约 1.217s；streamable HTTP stateless 示例已拆到 `http_mcp_server.py` / `http_mcp_call_example.py`。

MCP 返回值注意点：

- 传统工具的 `ToolMessage.content` 通常是字符串。
- MCP adapter 会返回 LangChain 标准 content blocks，例如 `[{"type": "text", "text": "..."}]`。
- 如果要统一解析结果，需要兼容两种格式。
- MCP structured content 会放在 `ToolMessage.artifact["structured_content"]` 中。

## 十一、预留 LangGraph 工具循环实验

`test_10_langgraph_tool_loop_reserved` 只保留编号和汇总占位。完整 vLLM + LangGraph 的 `llm -> tools -> llm` agent loop 暂时不作为本阶段工具调用实验内容，后续系统学习 LangGraph 后再补。

## 十二、如何复现实验

运行完整实验：

```bash
pytest zhuxs_learn/03_tool/langchain_vllm_tool_call_experiments.py -s
```

输出汇总：

```text
zhuxs_learn/03_tool/captures_tool_call/00_tool_call_experiment_results.json
```

注意：只有实际发起 vLLM HTTP 请求的实验会生成独立编号抓包文件；`test_09` 现在包含 vLLM 最小图，因此会生成 `09_mcp_runtime_hidden_llm_graph_capture.json`。纯 ToolNode 确定性实验和 `test_10` 占位只写入 `00_tool_call_experiment_results.json`。streamable HTTP stateless 示例不属于 pytest 主流程，按本节上面的两条 `python` 命令单独验证。

本实验只保留 pytest 入口，并默认 vLLM OpenAI-compatible 服务已经启动且可用。
没有额外的环境变量开关；如果服务不可用，相关测试应直接失败，便于暴露真实集成问题。

当前环境实测版本：

- `langchain==1.2.15`
- `langgraph==1.1.6`
- `langgraph-prebuilt==1.0.9`
- `langchain-openai==1.1.12`
- `openai==2.30.0`
- `langchain-mcp-adapters==0.2.2`
- `mcp==1.27.0`
- `fastmcp==3.2.3`

本地 Python 环境提示 `PyTorch was not found`，但实验连接的是远端已启动的 vLLM OpenAI-compatible HTTP 服务，因此不影响这些客户端侧实验。

## 十三、推荐落地模式

普通业务工具：

1. 用 `@tool` 或 Pydantic schema 定义工具。
2. 用 `ChatOpenAI(...).bind_tools(tools, tool_choice="auto" 或 "required")` 绑定模型。
3. 用 `StateGraph + ToolNode + tools_condition` 实现工具循环。
4. 工具需要上下文时使用 `runtime: ToolRuntime`。
5. 工具需要写状态时返回 `Command`，并包含匹配的 `ToolMessage`。

需要隔离、跨进程或多语言工具：

1. 用 FastMCP 实现 async MCP server。
2. 用 `MultiServerMCPClient` 加载工具。
3. 需要 runtime/state/context/store 时，用 MCP interceptor 桥接。
4. 性能敏感或需要验证并行时，使用持久 `ClientSession` 和 `load_mcp_tools(session, ...)`。
5. 统一处理 MCP content blocks 和 artifacts。

需要强 schema 约束：

1. vLLM 上优先选择 `tool_choice="required"` 或命名函数调用。
2. `auto` 适合 agent 自主选择，但不要把它当成 schema 强约束。
3. 应用层仍要校验工具参数，尤其是 enum、必填字段、范围和业务权限。

需要全局控制工具执行：

1. 单个工具调用的鉴权、缓存、参数修正、重试，优先用 `wrap_tool_call` / `awrap_tool_call`。
2. 多个工具调用之间存在依赖、排序、批量跳过或事务语义时，写 custom tool execution node。
3. custom node 不必兼容 ToolNode 的所有输入输出格式，但如果后续还要把结果交回 LLM，每个 LLM tool call 仍必须有对应 `ToolMessage`，否则消息历史非法。

只需要注入少量外部变量：

1. 不必为了一个 `user_id` / `account_id` 上 LangGraph。
2. 使用 `InjectedToolArg` 把隐藏参数排除在 LLM schema 外。
3. LLM 返回 tool call 后，应用层在调用 `tool.invoke` / `tool.ainvoke` 前把隐藏参数合并进去。

## 十四、已验证结论清单

- OpenAI SDK 与 LangChain 四组工具调用请求体关键字段一致。
- `tool_choice="none"` 时两边都没有 tool calls。
- 命名函数调用时 LangChain 字符串工具名会转成 OpenAI dict。
- vLLM `auto` 能一次返回两个 tool calls。
- `tools[].function.description` 会出现在 LangChain 发给 vLLM 的请求体中。
- `tools[].function.parameters.properties.<param>.description` 会出现在 LangChain 发给 vLLM 的请求体中。
- `@tool(args_schema=...)` 中无默认值字段会进入 JSON Schema `required`；在命名函数调用中，required 字段会压过“不要填写该参数”的 prompt。
- `tool_choice="auto"` 下，模型能根据 description 中的互斥暗号选择 `route_alpha` 或 `route_beta`。
- 参数 description 实验中，prompt 给出迷惑城市“上海”，模型仍按 `city.description` 填写“北京”。
- LangChain + vLLM 的 `required` 在 prompt 要求不用工具时仍返回 tool calls。
- LangChain + vLLM 的 `none` 在 prompt 要求用工具时不返回 API tool calls。
- LangChain + vLLM 的命名函数在 prompt 要求别的工具时仍返回指定工具。
- `convert_to_openai_tool(inspect_runtime)` 和实际 vLLM 抓包都不暴露 `runtime` 参数。
- ToolNode 和 custom node 两条最小 LLM -> 工具 -> LLM 图都能跑通，并在工具执行阶段读到 `ToolRuntime`。
- 普通 `llm.bind_tools` 不使用 LangGraph 时，可以用 `InjectedToolArg` 隐藏 `account_id`，再由应用层手动注入执行。
- ToolNode 能读取 `ToolRuntime.state/context/tool_call_id`。
- ToolNode 工具返回 `Command` 能写回图状态。
- ToolNode `handle_tool_errors=True` 能把传统工具异常转成错误 ToolMessage。
- ToolNode 对两个 async 慢工具并行执行。
- custom tool execution node 能读取 `ToolRuntime.state/context/tool_call_id`。
- custom tool execution node 能处理 `Command`、错误 ToolMessage，并按自定义顺序执行整批 tool calls。
- MCP 工具与传统工具返回的业务结果一致。
- MCP 工具参数 schema 能表达 required、默认值和 description。
- MCP LLM 最小图抓包证明 `user_id/session_label` 不会出现在发给 vLLM 的 `context_probe` schema 中。
- MCP interceptor 能把 LangGraph runtime 注入 MCP 工具参数；当前回显这些注入参数只是测试探针，生产工具不应把隐藏注入值返回给 LLM。
- MCP 错误能经 adapter 和 ToolNode 变成错误 ToolMessage。
- 持久 MCP session 下两个 1.2s MCP async 工具总耗时约 1.217s。
- streamable HTTP stateless 示例已独立为 `http_mcp_server.py` 和 `http_mcp_call_example.py`，不再进入主 pytest 实验。
- `test_10` 当前仅保留 LangGraph 工具循环占位，暂不运行端到端 agent loop。
