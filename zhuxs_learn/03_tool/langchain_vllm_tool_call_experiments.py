#!/usr/bin/env python3
"""LangChain + vLLM 工具调用最佳实践实验脚本。

实验速查：
01. `test_01_openai_sdk_and_langchain_request_body_alignment`
    对比 OpenAI SDK 与 LangChain 发送给 vLLM 的 tool calling 请求体。
02. `test_02_langchain_auto_can_use_tool_description`
    验证工具级 description 和参数级 description 会进入请求体，且 auto 能按工具描述选工具。
03. `test_03_langchain_tool_parameter_description_visible_to_llm`
    单独验证参数 `Field(description=...)` 会进入请求体并影响模型填参。
04. `test_04_langchain_vllm_tool_choice_strong_modes`
    验证 `required`、`none`、命名函数调用是 vLLM 强模式而不是建议。
05. `test_05_lightweight_llm_tool_external_context_injection`
    不使用 LangGraph，用 `InjectedToolArg` 给普通 `llm.bind_tools` 手动注入隐藏外部变量。
06. `test_06_traditional_toolnode_runtime_error_and_parallel`
    验证传统 `@tool` + ToolNode 的 runtime 注入、Command 写回、错误处理和并行执行。
07. `test_07_custom_tool_execution_node_runtime_command_error_order`
    验证逻辑清晰的自定义工具执行 node 支持 runtime 注入、Command 写回、错误处理和自定义执行顺序。
08. `test_08_runtime_hidden_minimal_graph_with_toolnode_and_custom_node`
    用 ToolNode/custom node 两条最小 LLM -> 工具 -> LLM 图证明 `ToolRuntime` 对 LLM 不可见。
09. `test_09_async_mcp_toolnode_compatibility`
    验证异步 MCP 工具接入 ToolNode 后与传统工具在结果、runtime、错误和并行上兼容。
    streamable HTTP MCP 已拆到同目录 `http_mcp_server.py` 和
    `http_mcp_call_example.py`，不放在本 pytest 主流程中。
10. `test_10_langgraph_tool_loop_reserved`
    预留给后续学习 LangGraph 后再补完整 vLLM 工具循环；当前不执行端到端逻辑。
11. `test_11_langchain_args_schema_required_parameter_over_prompt`
    验证 `@tool(args_schema=...)` 中 JSON Schema required 会强制生成参数，压过“不要填写”的 prompt。

唯一运行方式：
    pytest zhuxs_learn/03_tool/langchain_vllm_tool_call_experiments.py -s

前提：
    本脚本不负责启动或探测 vLLM；默认本地/远端 vLLM OpenAI-compatible
    服务已经启动且可用。所有测试都直接访问真实 vLLM，不再通过环境变量跳过。

注意：
    部分没有涉及到request的没有抓包到本地的0x_xxxxx.json，但是结果还是被保存到了全局记录： 00_tool_call_experiment_results.json
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import operator
import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import httpx
import openai
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

# =============================================================================
# 一、vLLM 与输出路径配置
# =============================================================================

OPENAI_BASE_URL = "http://192.168.10.99:33627/v1"
MODEL = "qwen36-27b"
API_KEY = "fake-vllm"

VLLM_EXTRA_BODY: dict[str, Any] = {
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    "top_k": 20,
    "min_p": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}

HERE = Path(__file__).resolve().parent
CAPTURE_DIR = HERE / "captures_tool_call"
RESULTS_PATH = CAPTURE_DIR / "00_tool_call_experiment_results.json"
MCP_SERVER_PATH = HERE / "mcp_async_tool_server.py"
EXPERIMENT_RESULTS: dict[str, Any] = {}


# =============================================================================
# 二、确定性业务数据与工具定义
# =============================================================================

CITY_DATA: dict[str, dict[str, Any]] = {
    "北京": {
        "weather": "sunny",
        "temperature_c": 26,
        "population_millions": 21.9,
    },
    "上海": {
        "weather": "cloudy",
        "temperature_c": 24,
        "population_millions": 24.9,
    },
    "Tokyo": {
        "weather": "rainy",
        "temperature_c": 19,
        "population_millions": 14.2,
    },
}


def stable_json(data: dict[str, Any]) -> str:
    """返回稳定 JSON 字符串，便于传统工具与 MCP 工具做结果一致性比较。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


# 实验工具：确定性天气查询（传统 @tool 版本）。
# 目的：为 ToolNode 实验和 MCP 结果一致性校验提供基准天气数据。
# 手段：从预设 CITY_DATA 直接读取，不发起网络请求，结果完全确定。
# 结果：返回含城市、温度（支持摄氏/华氏切换）、天气状况的稳定 JSON 字符串。
@tool
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get deterministic weather for a city."""
    data = CITY_DATA.get(city, CITY_DATA["北京"])
    temperature_c = int(data["temperature_c"])
    temperature = (
        temperature_c if unit == "celsius" else round(temperature_c * 9 / 5 + 32)
    )
    return stable_json(
        {
            "city": city,
            "source": "deterministic_fixture",
            "temperature": temperature,
            "unit": unit,
            "weather": data["weather"],
        }
    )


# 实验工具：确定性人口查询（传统 @tool 版本）。
# 目的：搭配 get_weather 验证并行工具调用能力，同时作为 MCP 结果比对的基准。
# 手段：从预设 CITY_DATA 直接读取，不发起网络请求，结果完全确定。
# 结果：返回含城市、人口（百万）的稳定 JSON 字符串。
@tool
def get_population(city: str) -> str:
    """Get deterministic population for a city."""
    data = CITY_DATA.get(city, CITY_DATA["北京"])
    return stable_json(
        {
            "city": city,
            "population_millions": data["population_millions"],
            "source": "deterministic_fixture",
        }
    )


# 实验工具：可控延迟回显，专门用于 ToolNode 并行执行耗时验证。
# 目的：在同一 AIMessage 的多个 tool_call 中并发调用，通过总耗时判断是否真正并行。
# 手段：asyncio.sleep 模拟 IO 阻塞，精确记录实际耗时写回结果。
# 结果：返回含标签、预期延迟、实际耗时的稳定 JSON 字符串。
@tool
async def slow_echo(label: str, delay_seconds: float = 1.2) -> str:
    """Return a label after an async sleep."""
    start = time.perf_counter()
    await asyncio.sleep(delay_seconds)
    return stable_json(
        {
            "delay_seconds": delay_seconds,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "label": label,
            "source": "deterministic_fixture",
        }
    )


@dataclass(frozen=True)
class ExperimentContext:
    """图运行时传入的不可变上下文。"""

    user_id: str


class ExperimentState(TypedDict, total=False):
    """ToolNode 实验用状态。

    `messages` 使用 LangGraph 官方推荐的 `add_messages` reducer；
    `tool_events` 使用列表追加 reducer，便于验证工具通过 `Command` 写回状态。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    session_label: str
    user_preferences: dict[str, str]
    tool_events: Annotated[list[str], operator.add]
    custom_execution_order: Annotated[list[str], operator.add]


# 实验工具：ToolRuntime 注入验证探针。
# 目的：验证 ToolNode 能将 LangGraph state 和 context 注入工具的 runtime 参数。
# 手段：直接读取 runtime.state 和 runtime.context，将关键字段（user_id、session_label、消息数等）序列化返回。
# 结果：返回含 user_id、session_label、消息数、工具调用 ID 的稳定 JSON 字符串。
@tool
def inspect_runtime(
    query: str, runtime: ToolRuntime[ExperimentContext, ExperimentState]
) -> str:
    """Inspect ToolRuntime state and context."""
    state = runtime.state
    context = runtime.context
    preferences = state.get("user_preferences", {})
    return stable_json(
        {
            "message_count": len(state.get("messages", [])),
            "query": query,
            "session_label": state.get("session_label", ""),
            "tool_call_id": runtime.tool_call_id,
            "user_id": context.user_id,
            "user_language": preferences.get("language", ""),
        }
    )


# 实验工具：Command 写回状态验证探针。
# 目的：验证工具可通过 LangGraph Command 向图状态追加数据（tool_events 列表）。
# 手段：构造 Command(update=...) 同时写入 ToolMessage 和 tool_events 字段，由 ToolNode 合并进状态。
# 结果：返回 Command 对象；执行后图状态的 tool_events 列表中将出现新事件字符串。
@tool
def record_runtime_event(
    event: str,
    runtime: ToolRuntime[ExperimentContext, ExperimentState],
) -> Command:
    """Write a tool event back to graph state with `Command`."""
    content = stable_json(
        {
            "event": event,
            "recorded_by": runtime.context.user_id,
            "tool_call_id": runtime.tool_call_id,
        }
    )
    return Command(
        update={
            "messages": [
                ToolMessage(content=content, tool_call_id=runtime.tool_call_id or "")
            ],
            "tool_events": [event],
        }
    )


# 实验工具：故意抛异常，验证 ToolNode 错误处理机制。
# 目的：确认 ToolNode(handle_tool_errors=True) 能捕获工具异常并返回带 error 状态的 ToolMessage。
# 手段：直接 raise ValueError，不做任何处理。
# 结果：ToolNode 返回 ToolMessage(status='error')，content 中包含异常信息。
@tool
def fail_on_purpose(reason: str = "intentional failure") -> str:
    """Raise an error so ToolNode error handling can be verified."""
    raise ValueError(f"traditional tool failed intentionally: {reason}")


MODEL_TOOLS: list[BaseTool] = [get_weather, get_population]
TRADITIONAL_TOOLNODE_TOOLS: list[BaseTool] = [
    get_weather,
    get_population,
    slow_echo,
    inspect_runtime,
    record_runtime_event,
    fail_on_purpose,
]


# =============================================================================
# 三、抓包与 JSON 序列化辅助函数
# =============================================================================


class HttpCapture:
    """记录 httpx 请求/响应，保存为便于人工审核的 JSON 文件。"""

    def __init__(self, save_path: Path) -> None:
        self.save_path = save_path
        self.records: list[dict[str, Any]] = []

    def _request_payload(self, request: httpx.Request) -> dict[str, Any]:
        """解析 httpx.Request，提取方法、URL、请求头和 body（尝试 JSON 解码，失败则保留原始字符串）。"""
        body: object = None
        if request.content:
            try:
                body = json.loads(request.content.decode("utf-8"))
            except Exception:
                body = request.content.decode("utf-8", errors="replace")
        return {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": body,
        }

    def _response_payload(self, response: httpx.Response) -> dict[str, Any]:
        """解析 httpx.Response，提取状态码、响应头和 body（尝试 JSON 解码，失败则保留原始文本）。"""
        body: object
        try:
            body = json.loads(response.text)
        except Exception:
            body = response.text
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": body,
        }

    def on_request(self, request: httpx.Request) -> None:
        """同步请求钩子：将请求信息追加到 records 并预留 response 槽位，由 on_response 回填。"""
        self.records.append(
            {"request": self._request_payload(request), "response": None}
        )

    def on_response(self, response: httpx.Response) -> None:
        """同步响应钩子：强制读取完整响应体后，将响应数据回填到最近的未完成记录。"""
        response.read()
        self._attach_response(self._response_payload(response))

    async def aon_request(self, request: httpx.Request) -> None:
        """异步请求钩子：同 on_request，供异步 httpx.AsyncClient 的 event_hooks 使用。"""
        self.records.append(
            {"request": self._request_payload(request), "response": None}
        )

    async def aon_response(self, response: httpx.Response) -> None:
        """异步响应钩子：异步读取完整响应体后，将响应数据回填到最近的未完成记录。"""
        await response.aread()
        self._attach_response(self._response_payload(response))

    def sync_hooks(self) -> dict[str, list[Any]]:
        """返回供同步 httpx.Client event_hooks 参数使用的请求/响应钩子字典。"""
        return {"request": [self.on_request], "response": [self.on_response]}

    def async_hooks(self) -> dict[str, list[Any]]:
        """返回供异步 httpx.AsyncClient event_hooks 参数使用的请求/响应钩子字典。"""
        return {"request": [self.aon_request], "response": [self.aon_response]}

    def save(self) -> None:
        """将所有抓包记录序列化为带缩进的 JSON 文件，便于人工审核每次 LLM HTTP 请求/响应体。"""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path.write_text(
            json.dumps(
                self.records, ensure_ascii=False, indent=2, default=json_default
            ),
            encoding="utf-8",
        )

    def _attach_response(self, response_payload: dict[str, Any]) -> None:
        """将响应数据回填到 records 中最近一条 response 为 None 的记录；若找不到则追加独立记录。"""
        for record in reversed(self.records):
            if record["response"] is None:
                record["response"] = response_payload
                return
        self.records.append({"request": None, "response": response_payload})


def json_default(obj: Any) -> Any:
    """把 LangChain/OpenAI/Pydantic 对象转为可保存 JSON。"""
    if isinstance(obj, BaseMessage):
        return message_to_dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return str(obj)


def tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    """构造 LangChain 标准 `ToolCall` 字典。"""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def initial_tool_state(tool_calls: list[dict[str, Any]]) -> ExperimentState:
    """为不依赖 LLM 输出的 ToolNode 实验构造状态。"""
    return {
        "messages": [
            HumanMessage(content="请执行预置工具调用。"),
            AIMessage(content="", tool_calls=tool_calls),
        ],
        "session_label": "toolnode-fixture-session",
        "user_preferences": {"language": "zh"},
        "tool_events": [],
    }


def compile_tool_graph(tools: list[BaseTool], *, handle_tool_errors: Any = True):
    """把 `ToolNode` 放进 `StateGraph`，确保 runtime/context/store 注入路径真实。"""
    builder = StateGraph(ExperimentState, context_schema=ExperimentContext)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=handle_tool_errors))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile()


def latest_tool_messages(result: dict[str, Any]) -> list[ToolMessage]:
    """从图结果中提取所有 ToolMessage。"""
    return [m for m in result.get("messages", []) if isinstance(m, ToolMessage)]


def tool_message_text(message: ToolMessage) -> str:
    """兼容传统字符串内容和 MCP content blocks，提取 ToolMessage 文本。"""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)


# =============================================================================
# 四、OpenAI SDK 与 LangChain 抓包对比实验
# =============================================================================


BODY_COMPARE_KEYS = [
    "model",
    "messages",
    "tools",
    "tool_choice",
    "stream",
    "temperature",
    "top_p",
    "repetition_penalty",
    "presence_penalty",
    "top_k",
    "min_p",
    "chat_template_kwargs",
]


def selected_body(body: dict[str, Any]) -> dict[str, Any]:
    """抽取需要比较的请求体字段，忽略 provider 自动补充的无关字段。"""
    return {key: body.get(key) for key in BODY_COMPARE_KEYS if key in body}


def sdk_tool_choice(mode: str) -> str | dict[str, Any]:
    """返回 OpenAI SDK 使用的 `tool_choice` 值。"""
    if mode == "named":
        return {"type": "function", "function": {"name": "get_weather"}}
    return mode


def langchain_tool_choice(mode: str) -> str | dict[str, Any]:
    """返回 LangChain 使用的 `tool_choice` 值。"""
    if mode == "named":
        return "get_weather"
    return mode


def prompt_for_mode(mode: str) -> str:
    """为每种工具选择模式提供稳定提示词。"""
    if mode == "none":
        return "不用工具，直接用一句中文说明你已收到请求。"
    if mode == "named":
        return "请查询北京当前天气，只调用 get_weather。"
    if mode == "required":
        return "请查询北京当前天气。必须调用一个合适的工具。"
    if mode == "auto":
        return "请同时查询北京天气和上海人口；两个信息相互独立，可以并行调用工具。"
    return "请查询北京当前天气。"


def run_openai_sdk_tool_call(
    mode: str,
    capture_path: Path,
) -> tuple[Any, HttpCapture]:
    """实验目的：用 OpenAI SDK 直连 vLLM，保存工具调用 HTTP 请求体。

    实验方法：工具 schema 来自 LangChain 的 `convert_to_openai_tool`，
    与 LangChain 侧复用同一份定义；显式传 `stream=False`，避免默认值差异
    干扰抓包对比。vLLM 的 `auto` 模式默认支持一次返回多个 tool calls，
    不额外传并行开关字段。
    """
    capture = HttpCapture(capture_path)
    formatted_tools = [convert_to_openai_tool(t) for t in MODEL_TOOLS]
    client = openai.OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=API_KEY,
        http_client=httpx.Client(
            timeout=httpx.Timeout(120.0, connect=10.0),
            event_hooks=capture.sync_hooks(),
        ),
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是工具调用测试助手。需要工具时只返回工具调用。",
            },
            {"role": "user", "content": prompt_for_mode(mode)},
        ],
        tools=formatted_tools,
        tool_choice=sdk_tool_choice(mode),
        stream=False,
        temperature=0.7,
        top_p=0.8,
        extra_body=VLLM_EXTRA_BODY,
    )
    capture.save()
    return response, capture


def run_langchain_tool_call(
    mode: str,
    capture_path: Path,
) -> tuple[AIMessage, HttpCapture]:
    """实验目的：用 LangChain `bind_tools` 直连 vLLM，保存工具调用请求体。

    实验方法：同一组 `MODEL_TOOLS` 交给 `bind_tools`；`named` 模式故意使用
    LangChain 推荐的字符串工具名，抓包验证其被规范化为 OpenAI dict。
    """
    capture = HttpCapture(capture_path)
    llm = ChatOpenAI(
        model=MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=API_KEY,
        temperature=0.7,
        top_p=0.8,
        extra_body=VLLM_EXTRA_BODY,
        http_client=httpx.Client(
            timeout=httpx.Timeout(120.0, connect=10.0),
            event_hooks=capture.sync_hooks(),
        ),
    ).bind_tools(
        MODEL_TOOLS,
        tool_choice=langchain_tool_choice(mode),
    )
    response = llm.invoke(
        [
            SystemMessage(content="你是工具调用测试助手。需要工具时只返回工具调用。"),
            HumanMessage(content=prompt_for_mode(mode)),
        ]
    )
    capture.save()
    return response, capture


def summarise_sdk_response(response: Any) -> dict[str, Any]:
    """提取 OpenAI SDK 响应中的工具调用信息。"""
    message = response.choices[0].message
    calls = []
    for call in message.tool_calls or []:
        raw_args = call.function.arguments
        try:
            parsed_args = json.loads(raw_args)
        except Exception:
            parsed_args = raw_args
        calls.append(
            {
                "id": call.id,
                "name": call.function.name,
                "args": parsed_args,
                "raw_args": raw_args,
            }
        )
    return {"content": message.content, "tool_calls": calls}


def summarise_langchain_response(message: AIMessage) -> dict[str, Any]:
    """提取 LangChain 响应中的工具调用信息。"""
    return {"content": message.content, "tool_calls": message.tool_calls}


def record_experiment_result(test_key: str, result: dict[str, Any]) -> None:
    """记录单个 pytest 实验结果并立即写入汇总文件。

    目的：pytest 测试函数必须返回 None，但实验仍需要保留机器可读的汇总结果。
    手段：把每个 test 的结果写入模块级 `EXPERIMENT_RESULTS`，并在每个测试
    完成后立即覆盖写入 `00_tool_call_experiment_results.json`。
    结果：只保留 pytest 一个入口；即使后续测试失败，也能看到已经完成的实验汇总。
    """
    EXPERIMENT_RESULTS[test_key] = result
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            EXPERIMENT_RESULTS, ensure_ascii=False, indent=2, default=json_default
        ),
        encoding="utf-8",
    )


def test_01_openai_sdk_and_langchain_request_body_alignment() -> None:
    """测试目的：确认 vLLM 工具调用在 OpenAI SDK 与 LangChain 中请求体一致。

    实验方法：分别覆盖 `auto`、`required`、`none`、命名函数调用。`auto`
    用例提示模型查询两个独立信息，用于确认 vLLM 默认支持一次返回多个
    tool calls；不额外传任何并行开关字段。每组保存 SDK 和 LangChain
    抓包 JSON，并比较关键请求体字段。

    实验结果：返回每种模式的请求体一致性、模型是否返回工具调用、抓包路径。
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        "auto",
        "required",
        "none",
        "named",
    ]
    results: dict[str, Any] = {}
    for mode in cases:
        sdk_response, sdk_capture = run_openai_sdk_tool_call(
            mode,
            CAPTURE_DIR / f"01_{mode}_openai_sdk_capture.json",
        )
        lc_response, lc_capture = run_langchain_tool_call(
            mode,
            CAPTURE_DIR / f"01_{mode}_langchain_capture.json",
        )
        sdk_body = selected_body(sdk_capture.records[0]["request"]["body"])
        lc_body = selected_body(lc_capture.records[0]["request"]["body"])
        bodies_match = sdk_body == lc_body
        results[mode] = {
            "bodies_match": bodies_match,
            "body_diff": (
                {} if bodies_match else {"openai_sdk": sdk_body, "langchain": lc_body}
            ),
            "openai_sdk": summarise_sdk_response(sdk_response),
            "langchain": summarise_langchain_response(lc_response),
            "captures": {
                "openai_sdk": str(CAPTURE_DIR / f"01_{mode}_openai_sdk_capture.json"),
                "langchain": str(CAPTURE_DIR / f"01_{mode}_langchain_capture.json"),
            },
        }
        assert bodies_match, f"{mode} 请求体关键字段不一致"
        if mode == "none":
            assert not results[mode]["openai_sdk"]["tool_calls"]
            assert not results[mode]["langchain"]["tool_calls"]
        elif mode == "auto":
            assert len(results[mode]["openai_sdk"]["tool_calls"]) >= 2
            assert len(results[mode]["langchain"]["tool_calls"]) >= 2
        elif mode == "named":
            assert results[mode]["openai_sdk"]["tool_calls"][0]["name"] == "get_weather"
            assert results[mode]["langchain"]["tool_calls"][0]["name"] == "get_weather"
        else:
            assert results[mode]["openai_sdk"][
                "tool_calls"
            ], f"{mode} SDK 未返回工具调用"
            assert results[mode]["langchain"][
                "tool_calls"
            ], f"{mode} LangChain 未返回工具调用"
    record_experiment_result("test_01_request_body_alignment", results)


# =============================================================================
# 五、LangChain + vLLM tool_choice 语义实验
# =============================================================================


class DescriptionProbeInput(BaseModel):
    """description 路由工具的参数 schema。"""

    city: str = Field(
        description=(
            "参数级 description 可见性探针：city 是工具要处理的城市名称。"
            "本实验要求填写中文城市名，例如“北京”。"
        )
    )


class ParameterDescriptionProbeInput(BaseModel):
    """专门验证参数 description 是否影响模型填参的 schema。"""

    # 上面这个docstring不会被抽取到工具 description 中，只有下面这个 Field(description=...) 会进入参数 schema 并被模型看到。
    # 这是langchain推荐的定义参数 description 的方式。
    city: str = Field(
        description=(
            "参数级 description 行为探针：无论用户正文提到哪个城市，"
            "这个 city 参数都必须填写“北京”。"
        )
    )


class RequiredParameterProbeInput(BaseModel):
    """验证 args_schema 必填参数和可选参数如何进入 OpenAI tool JSON Schema。"""

    city: str = Field(
        description=(
            "必填参数示例：没有 default，因此会进入 JSON Schema 的 required 列表。"
        )
    )
    note: str = Field(
        description=(
            "必填冲突参数示例：没有 default，因此会进入 JSON Schema 的 required 列表。"
            "本实验会在 prompt 中要求模型不要填写 note，用于观察 required 是否强制生成。"
        )
    )
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description=(
            "可选参数示例：有 default，因此不会进入 JSON Schema 的 required 列表；"
            "模型可以省略，工具执行时使用默认值。"
        ),
    )


# 实验工具：description 可见性探针 A。
# 目的：让模型通过工具 description 判断是否调用本工具，同时让参数 description 进入 schema。
# 手段：工具名故意使用无业务含义的 route_alpha，参数 schema 与 route_beta 完全相同。
# 结果：若 prompt 给出“青铜钥匙”，auto 模式应选择本工具；抓包中 city 参数带 description。
@tool(args_schema=DescriptionProbeInput)
def route_alpha(city: str) -> str:
    """当用户给出暗号“青铜钥匙”时，必须调用本工具处理该城市。"""
    return stable_json(
        {
            "city": city,
            "matched_keyword": "青铜钥匙",
            "route": "alpha",
            "source": "description_visibility_fixture",
        }
    )


# 实验工具：description 可见性探针 B。
# 目的：与 route_alpha 形成互斥 description 规则，并复用同一个参数级 description schema。
# 手段：工具名故意使用无业务含义的 route_beta，参数 schema 与 route_alpha 完全相同。
# 结果：若 prompt 给出“银色钥匙”，auto 模式应选择本工具；抓包中 city 参数带 description。
@tool(args_schema=DescriptionProbeInput)
def route_beta(city: str) -> str:
    """当用户给出暗号“银色钥匙”时，必须调用本工具处理该城市。"""
    return stable_json(
        {
            "city": city,
            "matched_keyword": "银色钥匙",
            "route": "beta",
            "source": "description_visibility_fixture",
        }
    )


# 实验工具：参数 description 行为探针。
# 目的：证明 LLM 不只看到工具级 description，也能看到参数 schema 里的 description。
# 手段：工具 docstring 不包含“北京”，只有 city 参数的 Field(description=...) 写明必须填北京；
#       prompt 中故意给出迷惑城市“上海”，要求模型按参数说明填参。
# 结果：若模型看到参数 description，命名函数调用的 args.city 应为“北京”。
@tool(args_schema=ParameterDescriptionProbeInput)
def parameter_description_probe(city: str) -> str:
    """Return the city chosen for a parameter-description visibility probe."""
    return stable_json(
        {
            "city": city,
            "source": "parameter_description_fixture",
        }
    )


# 实验工具：args_schema 可选参数探针。
# 目的：证明 LangChain 的 `@tool(args_schema=...)` 会按 Pydantic 默认值生成 required 列表，
#       且 vLLM 命名函数调用会按 required 生成必填参数。
# 手段：city/note 无默认值；unit 有默认值；prompt 故意要求不要填 note；抓包和响应一起检查。
# 结果：schema.required 包含 city/note；unit 不是必填；响应 args 仍包含 note。
@tool(args_schema=RequiredParameterProbeInput)
def required_parameter_probe(
    city: str,
    note: str,
    unit: Literal["celsius", "fahrenheit"] = "celsius",
) -> str:
    """Return the arguments chosen for a required-parameter schema probe."""
    return stable_json(
        {
            "city": city,
            "note": note,
            "source": "optional_parameter_fixture",
            "unit": unit,
        }
    )


# 实验工具：轻量外部上下文注入探针。
# 目的：不用 LangGraph，也让普通 llm.bind_tools 生成的 tool call 执行时拿到外部变量。
# 手段：account_id 用 InjectedToolArg 标记，LLM schema 中不可见；应用层执行工具时手动注入。
# 结果：抓包没有 account_id，工具结果中能看到应用注入的 account_id。
@tool
def lightweight_account_lookup(
    query: Annotated[str, "用户想查询的当前账户资料主题，例如 language 或 plan。"],
    account_id: Annotated[str, InjectedToolArg],
) -> str:
    """Look up deterministic private account data for the current account."""
    fixture = {
        "account-hidden-42": {
            "language": "zh",
            "plan": "enterprise",
        }
    }
    account_data = fixture.get(account_id, {})
    return stable_json(
        {
            "account_id": account_id,
            "query": query,
            "source": "lightweight_injection_fixture",
            "value": account_data.get(query, ""),
        }
    )


DESCRIPTION_PROBE_TOOLS: list[BaseTool] = [route_alpha, route_beta]
LIGHTWEIGHT_EXTERNAL_CONTEXT = {"account_id": "account-hidden-42"}


def run_langchain_tool_call_with_tools(
    tools: list[BaseTool],
    prompt: str,
    tool_choice_value: str | dict[str, Any],
    capture_path: Path,
    *,
    system_prompt: str = "你是工具调用测试助手。需要工具时只返回工具调用。",
    temperature: float = 0.0,
) -> tuple[AIMessage, HttpCapture]:
    """实验辅助：参数化执行一次 LangChain + vLLM 工具调用并保存抓包。

    目的：复用同一套 `ChatOpenAI.bind_tools` 逻辑，避免新增语义实验复制
    SDK 对齐实验中的客户端初始化代码。
    手段：传入工具列表、prompt、tool_choice 和抓包路径；temperature 默认
    为 0，降低 auto 选择实验的波动。
    结果：返回 LangChain 的 `AIMessage` 和已保存的 `HttpCapture`。
    """
    capture = HttpCapture(capture_path)
    llm = ChatOpenAI(
        model=MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        top_p=0.8,
        extra_body=VLLM_EXTRA_BODY,
        http_client=httpx.Client(
            timeout=httpx.Timeout(120.0, connect=10.0),
            event_hooks=capture.sync_hooks(),
        ),
    ).bind_tools(
        tools,
        tool_choice=tool_choice_value,
    )
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
    )
    capture.save()
    return response, capture


def tool_descriptions_from_capture(capture: HttpCapture) -> dict[str, str]:
    """从第一条 LLM 请求体中提取工具名到 description 的映射。

    目的：把“description 是否真的发给 vLLM”变成可断言的抓包证据。
    手段：读取 `tools[].function.name/description` 字段。
    结果：返回 `{tool_name: description}`，供实验断言和汇总 JSON 使用。
    """
    body = capture.records[0]["request"]["body"]
    return {
        tool_spec["function"]["name"]: tool_spec["function"].get("description", "")
        for tool_spec in body.get("tools", [])
        if tool_spec.get("type") == "function"
    }


def tool_parameter_descriptions_from_capture(
    capture: HttpCapture,
) -> dict[str, dict[str, str]]:
    """从第一条 LLM 请求体中提取工具参数 description。

    目的：把“参数级 description 是否真的发给 vLLM”变成可断言的抓包证据。
    手段：读取 `tools[].function.parameters.properties.*.description` 字段。
    结果：返回 `{tool_name: {param_name: description}}`，供实验断言和汇总 JSON 使用。
    """
    body = capture.records[0]["request"]["body"]
    descriptions: dict[str, dict[str, str]] = {}
    for tool_spec in body.get("tools", []):
        if tool_spec.get("type") != "function":
            continue
        function = tool_spec.get("function", {})
        properties = function.get("parameters", {}).get("properties", {})
        descriptions[function.get("name", "")] = {
            name: spec.get("description", "") for name, spec in properties.items()
        }
    return descriptions


def run_description_visibility_experiment() -> dict[str, Any]:
    """执行 description 可见性实验。

    目的：证明 LangChain 会把工具 description 发送给 vLLM，且 auto 模式下
    模型会使用 description 中的互斥规则选择工具。
    手段：分别给出“青铜钥匙”和“银色钥匙”两个 prompt，不暴露函数名；
    抓包检查 description，响应检查实际 tool_calls。
    结果：两组 case 都应只选择 description 中匹配暗号的工具。
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        {
            "case_name": "bronze_keyword",
            "keyword": "青铜钥匙",
            "expected_tool": "route_alpha",
        },
        {
            "case_name": "silver_keyword",
            "keyword": "银色钥匙",
            "expected_tool": "route_beta",
        },
    ]
    results: dict[str, Any] = {}
    for case in cases:
        capture_path = (
            CAPTURE_DIR / f"02_description_auto_{case['case_name']}_capture.json"
        )
        prompt = (
            "这是一个工具 description 路由实验。请只根据工具说明中的暗号规则"
            f"选择唯一工具；暗号：{case['keyword']}；city=北京。"
            "只返回工具调用，不要解释。"
        )
        response, capture = run_langchain_tool_call_with_tools(
            DESCRIPTION_PROBE_TOOLS,
            prompt,
            "auto",
            capture_path,
            system_prompt=(
                "你是工具调用路由测试助手。必须根据每个工具 description 中的暗号"
                "选择唯一匹配工具，不要根据函数名猜测。"
            ),
        )
        descriptions = tool_descriptions_from_capture(capture)
        parameter_descriptions = tool_parameter_descriptions_from_capture(capture)
        actual_tool_names = [call["name"] for call in response.tool_calls]
        description_seen = case["keyword"] in descriptions.get(
            case["expected_tool"], ""
        )
        city_parameter_description = parameter_descriptions.get(
            case["expected_tool"], {}
        ).get("city", "")
        parameter_description_seen = (
            "参数级 description 可见性探针" in city_parameter_description
        )
        results[case["case_name"]] = {
            "expected_tool": case["expected_tool"],
            "actual_tool_calls": response.tool_calls,
            "actual_tool_names": actual_tool_names,
            "capture_path": str(capture_path),
            "description_seen_in_request": description_seen,
            "parameter_description_seen_in_request": parameter_description_seen,
            "request_tool_choice": capture.records[0]["request"]["body"].get(
                "tool_choice"
            ),
            "request_tool_descriptions": descriptions,
            "request_parameter_descriptions": parameter_descriptions,
        }
        assert (
            description_seen
        ), f"{case['expected_tool']} 的 description 未出现在抓包中"
        assert (
            parameter_description_seen
        ), f"{case['expected_tool']}.city 的参数 description 未出现在抓包中"
        assert actual_tool_names == [
            case["expected_tool"]
        ], f"auto 未按 description 选择唯一工具，实际为 {actual_tool_names}"
    return results


def run_parameter_description_visibility_experiment() -> dict[str, Any]:
    """执行参数级 description 可见性和行为实验。

    目的：证明参数 `Field(description=...)` 不只在本地 schema 中存在，也会进入
    vLLM 请求体，并能影响模型填参。
    手段：工具 docstring 不含“北京”；`city` 参数 description 写明必须填“北京”；
    prompt 中给出迷惑城市“上海”，并要求按参数说明填参。
    结果：抓包中存在 `city.description`，模型返回的 tool call args.city 为“北京”。
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    capture_path = CAPTURE_DIR / "03_parameter_description_probe_capture.json"
    response, capture = run_langchain_tool_call_with_tools(
        [parameter_description_probe],
        (
            "参数 description 可见性实验：请调用工具并严格按参数 schema 中"
            "每个参数的 description 填写参数。用户正文里的迷惑城市是上海。"
        ),
        "parameter_description_probe",
        capture_path,
        system_prompt=(
            "你是参数 schema 可见性测试助手。填工具参数时，优先遵守参数"
            "description 中的取值规则。"
        ),
    )
    parameter_descriptions = tool_parameter_descriptions_from_capture(capture)
    city_description = parameter_descriptions["parameter_description_probe"]["city"]
    actual_tool_calls = response.tool_calls
    actual_city = (
        actual_tool_calls[0]["args"].get("city") if actual_tool_calls else None
    )
    result = {
        "expected_city_from_parameter_description": "北京",
        "actual_city": actual_city,
        "actual_tool_calls": actual_tool_calls,
        "capture_path": str(capture_path),
        "request_tool_choice": capture.records[0]["request"]["body"].get("tool_choice"),
        "request_parameter_descriptions": parameter_descriptions,
    }
    assert "必须填写“北京”" in city_description
    assert (
        actual_city == "北京"
    ), f"模型没有按参数 description 填参，实际 city={actual_city}"
    return result


def run_strong_tool_choice_experiment() -> dict[str, Any]:
    """执行 `required`、`none`、命名函数的强模式反例实验。

    目的：证明这三种 vLLM tool_choice 模式不是 prompt 建议，而是 API/解码层约束。
    手段：让 prompt 与 tool_choice 故意冲突，只使用 LangChain 直连 vLLM。
    结果：正常完成时，required 仍有工具调用；none 没有工具调用；named 返回指定工具。
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    cases = {
        "required_against_prompt": {
            "prompt": "强模式测试：请不要调用任何工具，只回答“收到”。",
            "tool_choice": "required",
            "expected": "has_tool_calls",
        },
        "none_against_prompt": {
            "prompt": "强模式测试：请必须调用 get_weather 查询北京天气，不能直接回答。",
            "tool_choice": "none",
            "expected": "no_tool_calls",
        },
        "named_against_prompt": {
            "prompt": "强模式测试：请调用 get_weather 查询北京天气，不要调用 get_population。",
            "tool_choice": "get_population",
            "expected": "only_get_population",
        },
    }
    results: dict[str, Any] = {}
    for case_name, case in cases.items():
        capture_path = CAPTURE_DIR / f"04_strong_{case_name}_langchain_capture.json"
        response, capture = run_langchain_tool_call_with_tools(
            MODEL_TOOLS,
            str(case["prompt"]),
            str(case["tool_choice"]),
            capture_path,
        )
        actual_tool_names = [call["name"] for call in response.tool_calls]
        result = {
            "prompt": case["prompt"],
            "tool_choice": case["tool_choice"],
            "request_tool_choice": capture.records[0]["request"]["body"].get(
                "tool_choice"
            ),
            "actual_tool_calls": response.tool_calls,
            "actual_tool_names": actual_tool_names,
            "content": response.content,
            "capture_path": str(capture_path),
        }
        results[case_name] = result
        if case["expected"] == "has_tool_calls":
            assert actual_tool_names, "required 正常完成时应至少返回一个 tool_call"
        elif case["expected"] == "no_tool_calls":
            assert not actual_tool_names, "none 模式不应返回 tool_call"
        else:
            assert actual_tool_names and set(actual_tool_names) == {
                "get_population"
            }, f"命名函数模式应只返回 get_population，实际为 {actual_tool_names}"
    return results


def run_required_parameter_schema_experiment() -> dict[str, Any]:
    """执行 `@tool(args_schema=...)` required 参数强约束实验。

    目的：回答“抓包里参数 schema 的 required 是否只是提示，还是会在 vLLM
    命名函数调用时强制模型输出必填参数”。
    手段：用 Pydantic args_schema 定义两个必填字段 city/note 和一个可选字段
    unit；prompt 故意要求不要填写 note；命名函数调用该工具并保存抓包。
    结果：抓包中的 `parameters.required` 包含 `city` 和 `note`；vLLM 正常返回的
    tool call args 仍包含 `note`，说明 required 参数结构约束压过自然语言提示。
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    capture_path = CAPTURE_DIR / "11_required_parameter_over_prompt_capture.json"
    response, capture = run_langchain_tool_call_with_tools(
        [required_parameter_probe],
        (
            "args_schema required 强约束实验：请调用 required_parameter_probe。"
            "工具调用里必须体现城市是北京；请不要填写 note 参数。"
        ),
        "required_parameter_probe",
        capture_path,
        system_prompt=(
            "你是工具参数 schema 测试助手。严格按照工具 schema 生成工具调用；"
            "没有必要时不要填写可选参数。"
        ),
    )
    schema = first_request_tool_schema(capture, "required_parameter_probe")
    parameters = schema.get("parameters", {})
    properties = parameters.get("properties", {})
    required_fields = parameters.get("required", [])
    actual_tool_calls = response.tool_calls
    actual_args = actual_tool_calls[0]["args"] if actual_tool_calls else {}

    assert "city" in properties
    assert "unit" in properties
    assert "note" in properties
    assert required_fields == [
        "city",
        "note",
    ], f"required 应包含 city 和 note，实际为 {required_fields}"
    assert actual_tool_calls, "命名函数调用未返回 tool call"
    assert actual_tool_calls[0]["name"] == "required_parameter_probe"
    assert (
        "city" in actual_args
    ), f"必填参数 city 未出现在模型生成的 args 中：{actual_args}"
    assert (
        "note" in actual_args
    ), f"prompt 要求不要填 note，但 required 必填参数 note 未出现在 args 中：{actual_args}"

    return {
        "capture_path": str(capture_path),
        "request_tool_schema": schema,
        "request_required_fields": required_fields,
        "required_fields_present_despite_prompt": {
            "city": "city" in actual_args,
            "note": "note" in actual_args,
        },
        "optional_fields_not_required": {
            "unit": "unit" not in required_fields,
        },
        "actual_tool_calls": actual_tool_calls,
        "actual_args": actual_args,
        "note": (
            "note 没有 default，因此进入 JSON Schema required；本实验 prompt "
            "明确要求不要填写 note，但 vLLM 命名函数调用仍生成了 note。unit 有 "
            "default，因此不在 required 中，模型可以省略。"
        ),
    }


async def run_lightweight_external_context_injection_experiment() -> dict[str, Any]:
    """执行不用 LangGraph 的外部上下文注入实验。

    目的：回答“只想用 llm with tools，但不想定义 LangGraph 图时，如何注入
    LLM 不可见的外部变量”。
    手段：`account_id` 使用 `InjectedToolArg` 隐藏在 tool schema 外；第一轮
    LLM 只生成 `query`；应用层在执行工具前把 `account_id` 合并进 tool call args。
    结果：抓包 schema 中没有 `account_id`，工具返回值中包含应用注入的 account_id。
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    capture_path = (
        CAPTURE_DIR / "05_lightweight_external_context_injection_capture.json"
    )
    capture = HttpCapture(capture_path)
    async_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0),
        event_hooks=capture.async_hooks(),
    )
    try:
        llm_with_tool = ChatOpenAI(
            model=MODEL,
            base_url=OPENAI_BASE_URL,
            api_key=API_KEY,
            temperature=0.0,
            top_p=0.8,
            extra_body=VLLM_EXTRA_BODY,
            http_async_client=async_client,
        ).bind_tools(
            [lightweight_account_lookup],
            tool_choice="lightweight_account_lookup",
        )
        messages: list[AnyMessage] = [
            SystemMessage(
                content=(
                    "你是轻量工具调用测试助手。只生成工具调用；工具参数里只填写"
                    "模型可见的 query，不要猜测任何账户 ID。"
                )
            ),
            HumanMessage(content="请查询当前账户的 language 设置。"),
        ]
        first_ai = await llm_with_tool.ainvoke(messages)
        assert first_ai.tool_calls, "轻量注入实验中 LLM 未生成 tool call"
        call = first_ai.tool_calls[0]
        injected_call = {
            **call,
            "args": {
                **call["args"],
                "account_id": LIGHTWEIGHT_EXTERNAL_CONTEXT["account_id"],
            },
            "type": "tool_call",
        }
        tool_message = await lightweight_account_lookup.ainvoke(injected_call)
    finally:
        await async_client.aclose()
        capture.save()

    schema = first_request_tool_schema(capture, "lightweight_account_lookup")
    properties = schema.get("parameters", {}).get("properties", {})
    parsed_tool_result = json.loads(tool_message_text(tool_message))
    assert "query" in properties
    assert "account_id" not in properties
    assert (
        parsed_tool_result["account_id"] == LIGHTWEIGHT_EXTERNAL_CONTEXT["account_id"]
    )
    return {
        "capture_path": str(capture_path),
        "request_tool_schema": schema,
        "account_id_visible_to_llm": "account_id" in properties,
        "llm_tool_call": call,
        "injected_tool_call": injected_call,
        "tool_message": tool_message,
        "parsed_tool_result": parsed_tool_result,
    }


def tool_calls_from_custom_state(state: ExperimentState) -> list[dict[str, Any]]:
    """从本实验图状态中提取最后一条 AIMessage 的 tool_calls。

    目的：让 custom node 的输入契约尽量简单，只处理 `ExperimentState`。
    手段：反向扫描 `state["messages"]`，找到最近的 AIMessage。
    结果：返回该 AIMessage 中的 tool_calls；没有时返回空列表。
    """
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return list(message.tool_calls)
    return []


def order_custom_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按业务规则重排工具调用，展示 custom node 的全局执行顺序控制能力。

    目的：证明 custom node 可以在执行前看到整批 tool calls，并做全局排序。
    手段：优先执行 runtime 读取，再执行 Command 写回，最后执行其他工具。
    结果：返回稳定排序后的 tool_calls；同优先级保持模型原始顺序。
    """
    priority = {
        "inspect_runtime": 0,
        "record_runtime_event": 1,
    }
    return [
        call
        for _, call in sorted(
            enumerate(calls),
            key=lambda item: (priority.get(item[1]["name"], 10), item[0]),
        )
    ]


def injected_arg_names_for_custom_node(tool_obj: BaseTool) -> set[str]:
    """识别 custom node 需要剥离并可信注入的工具参数名。

    目的：复刻 ToolNode 对 `runtime: ToolRuntime` 的最小识别能力，尤其兼容
    `from __future__ import annotations` 导致 `_injected_args_keys` 为空的脚本环境。
    手段：先读取 BaseTool 暴露的 injected key 集合，再检查底层函数签名中是否
    存在名为 `runtime` 的参数。
    结果：返回需要从 LLM args 中剥离、由执行节点重新注入的参数名集合。
    """
    injected_keys = set(getattr(tool_obj, "_injected_args_keys", frozenset()))
    source_func = getattr(tool_obj, "func", None) or getattr(
        tool_obj, "coroutine", None
    )
    if (
        source_func is not None
        and "runtime" in inspect.signature(source_func).parameters
    ):
        injected_keys.add("runtime")
    return injected_keys


def normalize_custom_command_update(
    command: Command,
    call: dict[str, Any],
) -> dict[str, Any]:
    """把工具返回的 Command 转成当前 custom node 可直接合并的状态更新。

    目的：支持工具用 Command 写回图状态，但不把 Command 继续交给 LangGraph。
    手段：只接受 `Command(update=dict)`；检查其中包含匹配 tool_call_id 的 ToolMessage。
    结果：返回普通 state update dict，供 custom node 统一合并。
    """
    if not isinstance(command.update, dict):
        raise ValueError("本 custom node 示例只处理 Command(update=dict)")

    update = dict(command.update)
    messages_update = list(update.get("messages", []))
    has_matching_tool_message = False
    for message in messages_update:
        if isinstance(message, ToolMessage) and message.tool_call_id == call["id"]:
            message.name = call["name"]
            has_matching_tool_message = True

    if not has_matching_tool_message:
        raise ValueError(
            f"Command.update 缺少与工具 {call['name']} / {call['id']} 匹配的 ToolMessage"
        )
    update["messages"] = messages_update
    return update


def merge_custom_state_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    """把单个工具产生的状态更新合并到 custom node 的总更新中。

    目的：让 Command 写回和普通 ToolMessage 写回走同一条清晰路径。
    手段：列表字段追加，其他字段覆盖。
    结果：target 原地更新。
    """
    for key, value in update.items():
        if isinstance(value, list):
            target.setdefault(key, [])
            target[key].extend(value)
        else:
            target[key] = value


async def execute_custom_tool_call(
    call: dict[str, Any],
    *,
    tool_by_name: dict[str, BaseTool],
    state: ExperimentState,
    graph_runtime: Runtime[ExperimentContext],
    config: RunnableConfig,
    handle_tool_errors: bool = True,
) -> dict[str, Any]:
    """执行单个 tool call，并返回普通 state update。

    目的：展示 custom node 最核心的四件事：runtime 注入、Command 写回、错误处理、
    以及不依赖 ToolNode 封装的工具调用执行。
    手段：构造可信 ToolRuntime，剥离 LLM 伪造的 injected 参数，调用工具；
    普通 ToolMessage 写入 messages，Command 转成普通 state update。
    结果：返回可由当前节点直接合并的 dict。
    """
    tool_obj = tool_by_name.get(call["name"])
    if tool_obj is None:
        return {
            "messages": [
                ToolMessage(
                    content=f"未知工具：{call['name']}，可用工具：{sorted(tool_by_name)}",
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )
            ]
        }

    tool_runtime = ToolRuntime(
        state=state,
        context=graph_runtime.context,
        config=config,
        stream_writer=graph_runtime.stream_writer,
        tool_call_id=call["id"],
        store=graph_runtime.store,
        execution_info=graph_runtime.execution_info,
        server_info=graph_runtime.server_info,
    )
    injected_keys = injected_arg_names_for_custom_node(tool_obj)
    sanitized_args = {
        key: value
        for key, value in dict(call.get("args", {})).items()
        if key not in injected_keys
    }
    if "runtime" in injected_keys:
        sanitized_args["runtime"] = tool_runtime
    trusted_call = {**call, "args": sanitized_args, "type": "tool_call"}

    try:
        response = await tool_obj.ainvoke(trusted_call, config)
    except Exception as exc:
        if not handle_tool_errors:
            raise
        return {
            "messages": [
                ToolMessage(
                    content=f"工具执行失败：{type(exc).__name__}: {exc}",
                    name=call["name"],
                    tool_call_id=call["id"],
                    status="error",
                )
            ]
        }

    if isinstance(response, Command):
        return normalize_custom_command_update(response, call)
    if isinstance(response, ToolMessage):
        return {"messages": [response]}
    return {
        "messages": [
            ToolMessage(
                content=str(response),
                name=call["name"],
                tool_call_id=call["id"],
            )
        ]
    }


def build_custom_tool_execution_node(
    tools: list[BaseTool],
    *,
    handle_tool_errors: bool = True,
):
    """构造一个逻辑清晰的自定义工具执行 node。

    目的：给需要全局排序、条件跳过、批量前后处理等场景提供一个可读版本，
    不追求兼容 ToolNode 的所有输入输出格式。
    手段：只接收 `ExperimentState`；提取最后一条 AIMessage.tool_calls；
    先按业务规则排序，再逐个执行并合并 state update。
    结果：返回可直接给本实验 `StateGraph` 使用的 async node。
    """
    tool_by_name = {tool_obj.name: tool_obj for tool_obj in tools}

    async def custom_tool_execution_node(
        state: ExperimentState,
        config: RunnableConfig,
        runtime: Runtime[ExperimentContext],
    ) -> dict[str, Any]:
        """LangGraph 节点函数：按自定义顺序执行最后一条 AIMessage 中的 tool calls。"""
        ordered_calls = order_custom_tool_calls(tool_calls_from_custom_state(state))
        current_state: ExperimentState = {
            **state,
            "messages": list(state.get("messages", [])),
            "tool_events": list(state.get("tool_events", [])),
            "custom_execution_order": list(state.get("custom_execution_order", [])),
        }
        combined_update: dict[str, Any] = {
            "messages": [],
            "custom_execution_order": [],
        }
        for call in ordered_calls:
            combined_update["custom_execution_order"].append(call["name"])
            update = await execute_custom_tool_call(
                call,
                tool_by_name=tool_by_name,
                state=current_state,
                graph_runtime=runtime,
                config=config,
                handle_tool_errors=handle_tool_errors,
            )
            merge_custom_state_update(combined_update, update)
            merge_custom_state_update(current_state, update)
        return combined_update

    return custom_tool_execution_node


def compile_custom_tool_graph(
    tools: list[BaseTool],
    *,
    handle_tool_errors: bool = True,
):
    """把 custom tool execution node 放入 `StateGraph`，用于验证自定义执行逻辑。

    目的：验证 custom node 在真实 LangGraph runtime/context 注入路径下工作，
    同时保留完全可控的工具执行顺序。
    手段：图结构与 `compile_tool_graph` 一致，只替换 tools 节点实现。
    结果：返回可执行的 compiled graph。
    """
    builder = StateGraph(ExperimentState, context_schema=ExperimentContext)
    builder.add_node(
        "tools",
        build_custom_tool_execution_node(
            tools,
            handle_tool_errors=handle_tool_errors,
        ),
    )
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile()


async def run_custom_tool_execution_node_experiment() -> dict[str, Any]:
    """执行 custom node 的 runtime、Command、错误和自定义顺序验证。

    目的：展示一个不追求 ToolNode 全格式兼容、但逻辑清晰可控的工具执行节点。
    手段：故意把 record_runtime_event 放在 inspect_runtime 前面输入，custom node
    先做全局排序，再逐个执行；同时验证错误处理。
    结果：返回 runtime 注入、Command 写回、错误 ToolMessage 和实际执行顺序。
    """
    runtime_graph = compile_custom_tool_graph(TRADITIONAL_TOOLNODE_TOOLS)
    runtime_state = initial_tool_state(
        [
            tool_call(
                "record_runtime_event",
                {"event": "custom_runtime_checked"},
                "custom-event-call",
            ),
            tool_call(
                "inspect_runtime", {"query": "读取运行时"}, "custom-runtime-call"
            ),
        ]
    )
    runtime_result = await runtime_graph.ainvoke(
        runtime_state,
        context=ExperimentContext(user_id="custom-user"),
    )
    runtime_messages = latest_tool_messages(runtime_result)
    runtime_parsed = [
        json.loads(tool_message_text(message)) for message in runtime_messages
    ]
    assert any(item.get("user_id") == "custom-user" for item in runtime_parsed)
    assert "custom_runtime_checked" in runtime_result.get("tool_events", [])
    assert runtime_result.get("custom_execution_order") == [
        "inspect_runtime",
        "record_runtime_event",
    ]

    error_graph = compile_custom_tool_graph(TRADITIONAL_TOOLNODE_TOOLS)
    error_result = await error_graph.ainvoke(
        initial_tool_state(
            [
                tool_call(
                    "fail_on_purpose",
                    {"reason": "验证 custom node 错误处理"},
                    "custom-fail-call",
                )
            ]
        ),
        context=ExperimentContext(user_id="custom-user"),
    )
    error_messages = latest_tool_messages(error_result)
    assert error_messages and error_messages[-1].status == "error"

    return {
        "runtime_and_command": {
            "tool_messages": runtime_messages,
            "tool_events": runtime_result.get("tool_events", []),
            "custom_execution_order": runtime_result.get("custom_execution_order", []),
            "parsed": runtime_parsed,
        },
        "error_handling": {"tool_messages": error_messages},
    }


def first_request_tool_schema(capture: HttpCapture, tool_name: str) -> dict[str, Any]:
    """从抓包第一条请求里取指定工具的 OpenAI function schema。

    目的：检查发给 vLLM 的 schema 是否只包含 LLM 可控参数。
    手段：遍历 `tools` 字段，按 function name 匹配。
    结果：返回该工具 schema；找不到时抛出 AssertionError。
    """
    tools = capture.records[0]["request"]["body"].get("tools", [])
    for tool_spec in tools:
        if tool_spec.get("function", {}).get("name") == tool_name:
            return tool_spec["function"]
    raise AssertionError(f"抓包请求体中找不到工具 schema：{tool_name}")


async def run_runtime_hidden_graph_once(
    tool_node_kind: Literal["toolnode", "custom"],
    capture_path: Path,
) -> dict[str, Any]:
    """执行一条最小 LLM -> 工具 -> LLM 图路径。

    目的：在真实 vLLM 请求抓包中证明 `ToolRuntime` 没有进入工具参数 schema，
    同时证明工具执行阶段仍能读取 runtime.state/context。
    手段：第一轮 LLM 使用命名函数强制调用 `inspect_runtime`；tools 节点分别
    用 ToolNode 或 custom node；第二轮 LLM 使用 `tool_choice='none'` 生成最终答案。
    结果：返回两轮请求、工具消息、最终消息和 schema 检查结果。
    """
    capture = HttpCapture(capture_path)
    async_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0),
        event_hooks=capture.async_hooks(),
    )

    async def call_llm(state: ExperimentState) -> dict[str, list[AIMessage]]:
        has_tool_result = any(
            isinstance(message, ToolMessage) for message in state["messages"]
        )
        tool_choice_value = "none" if has_tool_result else "inspect_runtime"
        model_with_tools = ChatOpenAI(
            model=MODEL,
            base_url=OPENAI_BASE_URL,
            api_key=API_KEY,
            temperature=0.0,
            top_p=0.8,
            extra_body=VLLM_EXTRA_BODY,
            http_async_client=async_client,
        ).bind_tools(
            [inspect_runtime],
            tool_choice=tool_choice_value,
        )
        response = await model_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(ExperimentState, context_schema=ExperimentContext)
    builder.add_node("llm", call_llm)
    if tool_node_kind == "toolnode":
        builder.add_node("tools", ToolNode([inspect_runtime], handle_tool_errors=True))
    else:
        builder.add_node(
            "tools",
            build_custom_tool_execution_node([inspect_runtime]),
        )
    builder.add_edge(START, "llm")
    builder.add_conditional_edges(
        "llm", tools_condition, {"tools": "tools", "__end__": END}
    )
    # 第 1 个参数 "llm" — 源节点名称。表示条件边从 "llm" 节点出发。
    # 第 2 个参数 tools_condition — 路由函数（条件判断函数）。它接收当前 state 作为输入，返回一个字符串作为路由键。LangGraph 内置的 tools_condition 函数逻辑是：
    # 如果 LLM 的消息中包含 tool_calls（即模型决定调用工具） → 返回 "tools"
    # 否则（模型没有调用工具，直接给出文本回复） → 返回 "__end__"
    # 第 3 个参数 {"tools": "tools", "__end__": END} — 路由映射表（字典）。key 是路由函数的返回值，value 是目标节点名称：
    # "tools" → 走到名为 "tools" 的节点（执行工具调用）
    # "__end__" → 走到 END，即图执行结束
    builder.add_edge("tools", "llm")
    graph = builder.compile()

    try:
        result = await graph.ainvoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "你是工具调用流程测试助手。第一次需要调用 inspect_runtime；"
                            "收到工具结果后只基于工具结果用一句中文回答，不要再次调用工具。"
                        )
                    ),
                    HumanMessage(content="请读取运行时信息，query=完整流程验证。"),
                ],
                "session_label": f"runtime-hidden-{tool_node_kind}",
                "user_preferences": {"language": "zh"},
                "tool_events": [],
            },
            context=ExperimentContext(user_id=f"{tool_node_kind}-user"),
            config={"recursion_limit": 6},
        )
    finally:
        await async_client.aclose()
        capture.save()

    schema = first_request_tool_schema(capture, "inspect_runtime")
    properties = schema.get("parameters", {}).get("properties", {})
    tool_messages = latest_tool_messages(result)
    parsed_tool_messages = [
        json.loads(tool_message_text(message)) for message in tool_messages
    ]
    final_message = result["messages"][-1]
    assert "query" in properties
    assert "runtime" not in properties
    assert tool_messages, f"{tool_node_kind} 路径没有执行工具"
    assert any(
        item.get("user_id") == f"{tool_node_kind}-user" for item in parsed_tool_messages
    )
    assert isinstance(final_message, AIMessage)
    assert not final_message.tool_calls

    return {
        "capture_path": str(capture_path),
        "llm_request_count": len(capture.records),
        "request_tool_schema": schema,
        "runtime_property_visible_to_llm": "runtime" in properties,
        "tool_messages": tool_messages,
        "parsed_tool_messages": parsed_tool_messages,
        "final_message": final_message,
    }


async def run_runtime_hidden_graph_experiment() -> dict[str, Any]:
    """同时执行 ToolNode 和 custom node 两条 runtime hidden 完整图实验。

    目的：把官方 ToolNode 路径和完全自定义执行路径放在同一证据框架中比较。
    手段：先检查 `convert_to_openai_tool(inspect_runtime)` 的 schema，再分别跑
    两个图并检查抓包第一轮请求。
    结果：两条路径都应隐藏 `runtime` 参数、执行工具、回到 LLM 生成最终消息。
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    openai_schema = convert_to_openai_tool(inspect_runtime)
    schema_properties = openai_schema["function"]["parameters"].get("properties", {})
    assert "query" in schema_properties
    assert "runtime" not in schema_properties

    toolnode_result = await run_runtime_hidden_graph_once(
        "toolnode",
        CAPTURE_DIR / "08_runtime_hidden_toolnode_graph_capture.json",
    )
    custom_result = await run_runtime_hidden_graph_once(
        "custom",
        CAPTURE_DIR / "08_runtime_hidden_custom_graph_capture.json",
    )
    return {
        "convert_to_openai_tool_schema": openai_schema,
        "toolnode_graph": toolnode_result,
        "custom_node_graph": custom_result,
    }


def test_02_langchain_auto_can_use_tool_description() -> None:
    """测试目的：证明 `tool_choice='auto'` 时模型能利用工具 description 做选择。

    实验方法：构造两个名称不含业务语义、参数 schema 完全相同、但 description
    写有互斥暗号规则的工具。prompt 只给暗号，不给函数名；抓包确认
    `tools[].function.description` 已发送给 vLLM，再断言模型选择与 description
    规则匹配的工具。

    实验结果：返回两组暗号实验的抓包路径、description 是否出现在请求体中、
    以及 vLLM 在 auto 模式下实际生成的 tool calls。
    """
    result = run_description_visibility_experiment()
    record_experiment_result("test_02_description_visibility_auto", result)


def test_03_langchain_tool_parameter_description_visible_to_llm() -> None:
    """测试目的：证明工具参数级 description 也会进入 LLM 可见的 JSON Schema。

    实验方法：复用 description 路由工具，但要求其 `city` 参数通过 Pydantic
    `Field(description=...)` 明确写入说明；抓包检查
    `tools[].function.parameters.properties.city.description`。

    实验结果：返回每个工具的参数 schema，并断言 `city.description` 出现在请求体中。
    """
    result = run_parameter_description_visibility_experiment()
    record_experiment_result("test_03_parameter_description_visibility", result)


def test_04_langchain_vllm_tool_choice_strong_modes() -> None:
    """测试目的：用 LangChain 证明 vLLM 的 `required`、`none`、命名函数是强模式。

    实验方法：
    1. `required`：prompt 明确要求不要用工具，但 tool_choice 强制为 required。
    2. `none`：prompt 明确要求调用天气工具，但 tool_choice 强制为 none。
    3. 命名函数：prompt 要求调用天气工具，但 tool_choice 强制为 get_population。

    实验结果：正常完成时，required 仍返回至少一个工具调用；none 不返回工具调用；
    命名函数返回指定函数名。该实验只断言模式约束，不断言参数语义质量。
    """
    result = run_strong_tool_choice_experiment()
    record_experiment_result("test_04_langchain_strong_tool_choice", result)


def test_05_lightweight_llm_tool_external_context_injection() -> None:
    """测试目的：不用 LangGraph，也给普通 `llm.bind_tools` 的工具注入外部变量。

    实验方法：让 vLLM 只看到 `query` 参数；工具真正执行时由一个轻量执行器
    手动把 `account_id` 注入到 `tool.ainvoke` 的参数中。抓包检查 LLM 不可见
    `account_id`，工具结果检查外部变量确实参与执行。

    实验结果：返回请求 schema、工具调用、工具消息和最终注入的外部上下文。
    """

    async def run_all() -> dict[str, Any]:
        return await run_lightweight_external_context_injection_experiment()

    record_experiment_result(
        "test_05_lightweight_external_context_injection",
        asyncio.run(run_all()),
    )


def test_06_traditional_toolnode_runtime_error_and_parallel() -> None:
    """测试目的：验证传统 LangChain 工具与 `ToolNode` 的核心能力。

    实验方法：
    1. 构造带两个 tool calls 的 `AIMessage`，验证 `ToolRuntime` 能访问 state/context，
       且工具可通过 `Command` 写回 `tool_events`。
    2. 故意触发 `ValueError`，验证 `ToolNode(handle_tool_errors=True)` 返回
       `ToolMessage(status='error')`。
    3. 同一条 `AIMessage` 中放两个 `slow_echo`，用总耗时验证异步并行执行。

    实验结果：返回 runtime 解析结果和command回写state、错误 ToolMessage（tool执行返回的valueerror可以被toolnode捕获并包装成ToolMessage(status='error')）、并行耗时(证明toolnode的确是并行执行)。
    """

    async def run_all() -> dict[str, Any]:
        return {
            "runtime_and_command": await _run_traditional_runtime_and_command_test(),
            "error_handling": await _run_traditional_error_test(),
            "parallel_execution": await _run_traditional_parallel_test(),
        }

    record_experiment_result("test_06_traditional_toolnode", asyncio.run(run_all()))


def test_07_custom_tool_execution_node_runtime_command_error_order() -> None:
    """测试目的：验证自定义工具执行 node 的 runtime、Command、错误处理和执行顺序。

    实验方法：手工构造 `AIMessage.tool_calls`，故意把 Command 写回工具排在
    runtime 读取工具前面；custom node 先做全局排序，再逐个执行工具并合并
    state update。另用失败工具验证错误包装为 ToolMessage。

    实验结果：返回 runtime 解析结果、Command 写回的 state 字段、错误消息和
    custom_execution_order，证明执行顺序由节点代码控制，而不是由模型顺序决定。
    """

    async def run_all() -> dict[str, Any]:
        return await run_custom_tool_execution_node_experiment()

    record_experiment_result(
        "test_07_custom_tool_execution_node_order", asyncio.run(run_all())
    )


def test_08_runtime_hidden_minimal_graph_with_toolnode_and_custom_node() -> None:
    """测试目的：跑通最小 LLM -> 工具 -> LLM 图，并抓包证明 ToolRuntime 对 LLM 不可见。

    实验方法：同一个 LangGraph 结构分别接入 ToolNode 和 custom node。第一轮 LLM
    通过命名函数强制调用 `inspect_runtime`，请求抓包检查工具 schema 中只有
    `query`，没有隐藏的 `runtime`；工具执行时再由 node 注入 ToolRuntime 并返回
    state/context 信息；第二轮 LLM 基于 ToolMessage 生成最终答案。

    实验结果：返回 ToolNode/custom node 两条路径的抓包、工具消息、最终消息，
    并断言两条路径都完成完整工具循环且请求体没有暴露 runtime 参数。
    """

    async def run_all() -> dict[str, Any]:
        return await run_runtime_hidden_graph_experiment()

    record_experiment_result(
        "test_08_runtime_hidden_minimal_graph", asyncio.run(run_all())
    )


# =============================================================================
# 六、传统 ToolNode 确定性实验
# =============================================================================


async def _run_traditional_runtime_and_command_test() -> dict[str, Any]:
    """目的：验证@tool定义的传统工具在graph中的ToolNode执行时可以通过预定义的 ToolRuntime 访问 state/context（工具定义中本身需要包含 runtime 参数），并用 Command 写回图状态。
    手段：构造含 inspect_runtime + record_runtime_event 两个 tool_call 的初始状态，
          用 ExperimentContext(user_id='traditional-user') 注入运行时上下文后执行图。
    结果：parsed 列表中含 user_id 匹配项；tool_events 中含 'runtime_checked' 事件。
    """
    graph = compile_tool_graph(TRADITIONAL_TOOLNODE_TOOLS, handle_tool_errors=True)
    state = initial_tool_state(
        [
            tool_call(
                "inspect_runtime", {"query": "读取运行时"}, "runtime-call"
            ),  # ToolRuntime 注入验证探针。
            tool_call(
                "record_runtime_event", {"event": "runtime_checked"}, "event-call"
            ),  # Command 写回验证探针。往图状态tool_events字段中新增runtime_checked
            # 这里传入的initial state如下，tool_events是[]
            # {
            #     "messages": [
            #         HumanMessage(content="请执行预置工具调用。"),
            #         AIMessage(content="", tool_calls=tool_calls),
            #     ],
            #     "session_label": "toolnode-fixture-session",
            #     "user_preferences": {"language": "zh"},
            #     "tool_events": [],
            # }
            # 但是下面经过record_runtime_event工具调用后，tool_events字段会被Command写回更新为["runtime_checked"]
        ]
    )
    result = await graph.ainvoke(
        state, context=ExperimentContext(user_id="traditional-user")
    )
    tool_messages = latest_tool_messages(result)
    parsed = [json.loads(tool_message_text(message)) for message in tool_messages]
    assert any(item.get("user_id") == "traditional-user" for item in parsed)
    assert "runtime_checked" in result.get("tool_events", [])
    return {
        "tool_messages": tool_messages,
        # tool的返回包含从congtext中取到的user_id，以及工具事件记录的结果；这些都是证明 ToolRuntime 注入成功的关键证据
        "tool_events": result.get("tool_events", []),
        "parsed": parsed,
    }


async def _run_traditional_error_test() -> dict[str, Any]:
    """目的：验证 ToolNode(handle_tool_errors=True) 能捕获传统工具抛出的异常。
    手段：向图注入调用 fail_on_purpose 的 tool_call，该工具必然抛出 ValueError。
    结果：最后一条 ToolMessage 的 status 字段为 'error'，包含异常信息。
    """
    graph = compile_tool_graph(TRADITIONAL_TOOLNODE_TOOLS, handle_tool_errors=True)
    state = initial_tool_state(
        [
            tool_call(
                "fail_on_purpose", {"reason": "验证 ToolNode 错误处理"}, "fail-call"
            )
        ]
    )
    result = await graph.ainvoke(
        state, context=ExperimentContext(user_id="traditional-user")
    )
    tool_messages = latest_tool_messages(result)
    assert tool_messages and tool_messages[-1].status == "error"
    return {"tool_messages": tool_messages}


async def _run_traditional_parallel_test() -> dict[str, Any]:
    """目的：验证 ToolNode 对同一 AIMessage 中的多个 tool_call 确实并发执行。
    手段：同时调用两个各需 1.2s 的 slow_echo，若并行则总耗时应 < 2.0s。
    结果：返回两条 ToolMessage 和总耗时；断言耗时 < 2.0s 以证明并行。
    """
    graph = compile_tool_graph(TRADITIONAL_TOOLNODE_TOOLS, handle_tool_errors=True)
    state = initial_tool_state(
        [
            tool_call("slow_echo", {"label": "A", "delay_seconds": 1.2}, "slow-a"),
            tool_call("slow_echo", {"label": "B", "delay_seconds": 1.2}, "slow-b"),
        ]
    )
    start = time.perf_counter()
    result = await graph.ainvoke(
        state, context=ExperimentContext(user_id="traditional-user")
    )
    elapsed = time.perf_counter() - start
    tool_messages = latest_tool_messages(result)
    assert len(tool_messages) == 2
    assert elapsed < 2.0, f"传统 ToolNode 可能未并行执行，耗时 {elapsed:.3f}s"
    return {"elapsed_seconds": round(elapsed, 3), "tool_messages": tool_messages}


# =============================================================================
# 七、异步 MCP + ToolNode 确定性实验
# =============================================================================


async def inject_runtime_context(
    request: MCPToolCallRequest,
    handler: Any,
) -> Any:
    """把 LangGraph runtime 桥接给 MCP server。

    MCP server 独立运行，无法直接访问 LangGraph state/context。此 interceptor
    从 `request.runtime` 读取 `context.user_id` 和 `state.session_label`，再注入
    `context_probe` 的显式工具参数，用于验证 MCP 与 runtime 信息兼容。
    """
    runtime = request.runtime
    if request.name == "context_probe" and runtime is not None:
        user_id = runtime.context.user_id
        session_label = runtime.state.get("session_label", "")
        request = request.override(
            args={
                **request.args,
                "session_label": session_label,
                "user_id": user_id,
            }
        )
    return await handler(request)


async def _run_mcp_equivalence_and_context_test(
    tools: list[BaseTool],
) -> dict[str, Any]:
    """目的：验证 MCP 工具结果与传统 @tool 完全一致，且 runtime context 可通过 interceptor 注入。
    手段：同时调用 get_weather、get_population、context_probe 三个 MCP 工具；
          context_probe 的 user_id 和 session_label 由 inject_runtime_context interceptor 动态填充。
    结果：parsed 中含与传统工具相同的天气/人口数据，及含 user_id 的 context_probe 结果。
    """
    graph = compile_tool_graph(tools, handle_tool_errors=True)
    state = initial_tool_state(
        [
            tool_call(
                "get_weather", {"city": "北京", "unit": "celsius"}, "mcp-weather"
            ),
            tool_call("get_population", {"city": "上海"}, "mcp-population"),
            tool_call("context_probe", {"query": "读取 MCP runtime"}, "mcp-context"),
        ]
    )
    result = await graph.ainvoke(state, context=ExperimentContext(user_id="mcp-user"))
    messages = latest_tool_messages(result)
    parsed = [json.loads(tool_message_text(message)) for message in messages]
    traditional_weather = json.loads(
        get_weather.invoke({"city": "北京", "unit": "celsius"})
    )
    traditional_population = json.loads(get_population.invoke({"city": "上海"}))
    assert traditional_weather in parsed
    assert traditional_population in parsed
    assert any(item.get("user_id") == "mcp-user" for item in parsed)
    assert any(
        item.get("session_label") == "toolnode-fixture-session" for item in parsed
    )
    return {"tool_messages": messages, "parsed": parsed}


async def _run_mcp_error_test(tools: list[BaseTool]) -> dict[str, Any]:
    """目的：验证 MCP 工具在 ToolNode 中抛出异常时的错误处理与传统工具等价。
    手段：向图注入调用 always_fail 的 tool_call，该 MCP 工具必然抛出 ValueError。
    结果：最后一条 ToolMessage 的 status 字段为 'error'，包含 MCP 工具异常信息。
    """
    graph = compile_tool_graph(tools, handle_tool_errors=True)
    state = initial_tool_state(
        [tool_call("always_fail", {"reason": "验证 MCP 错误处理"}, "mcp-fail")]
    )
    result = await graph.ainvoke(state, context=ExperimentContext(user_id="mcp-user"))
    messages = latest_tool_messages(result)
    assert messages and messages[-1].status == "error"
    return {"tool_messages": messages}


async def _run_mcp_parallel_test(tools: list[BaseTool]) -> dict[str, Any]:
    """目的：验证 MCP 工具在 ToolNode 中也能并发执行，而非串行等待。
    手段：同时调用两个各需 1.2s 的 MCP slow_echo，若并行则总耗时应 < 2.4s（MCP 进程启动开销略大）。
    结果：返回两条 ToolMessage 和总耗时；断言耗时 < 2.4s 以证明并行。
    """
    graph = compile_tool_graph(tools, handle_tool_errors=True)
    state = initial_tool_state(
        [
            tool_call(
                "slow_echo", {"label": "mcp-A", "delay_seconds": 1.2}, "mcp-slow-a"
            ),
            tool_call(
                "slow_echo", {"label": "mcp-B", "delay_seconds": 1.2}, "mcp-slow-b"
            ),
        ]
    )
    start = time.perf_counter()
    result = await graph.ainvoke(state, context=ExperimentContext(user_id="mcp-user"))
    elapsed = time.perf_counter() - start
    messages = latest_tool_messages(result)
    assert len(messages) == 2
    assert elapsed < 2.4, f"MCP ToolNode 可能未并行执行，耗时 {elapsed:.3f}s"
    return {"elapsed_seconds": round(elapsed, 3), "tool_messages": messages}


def mcp_tool_schema_summary(tools: list[BaseTool]) -> dict[str, dict[str, Any]]:
    """提取 MCP 工具的 OpenAI function 参数 schema 摘要。

    目的：证明 MCP 工具和传统 `@tool(args_schema=...)` 一样能表达参数
    required、default 和 description。
    手段：对加载后的 MCP BaseTool 调用 `convert_to_openai_tool`，读取
    `function.parameters`。
    结果：返回每个 MCP 工具的参数 schema，供 test_09 断言和汇总。
    """
    return {
        tool_obj.name: convert_to_openai_tool(tool_obj)["function"]["parameters"]
        for tool_obj in tools
    }


def assert_mcp_parameter_schema(schema_by_name: dict[str, dict[str, Any]]) -> None:
    """断言 MCP 参数 schema 中包含 required、default 和 description。

    目的：防止 MCP server 的参数说明退化成只有类型、没有描述或默认值。
    手段：检查代表性工具的 required 列表、默认值和每个参数 description。
    结果：如果 MCP 工具 schema 与传统工具能力不一致，测试会失败。
    """
    weather = schema_by_name["get_weather"]
    assert weather.get("required") == ["city"]
    assert weather["properties"]["city"]["description"]
    assert weather["properties"]["unit"]["default"] == "celsius"
    assert weather["properties"]["unit"]["description"]

    population = schema_by_name["get_population"]
    assert population.get("required") == ["city"]
    assert population["properties"]["city"]["description"]

    slow = schema_by_name["slow_echo"]
    assert slow.get("required") == ["label"]
    assert slow["properties"]["label"]["description"]
    assert slow["properties"]["delay_seconds"]["default"] == 1.2
    assert slow["properties"]["delay_seconds"]["description"]

    context = schema_by_name["context_probe"]
    assert context.get("required") == ["query"]
    assert context["properties"]["query"]["description"]
    assert context["properties"]["user_id"]["default"] == ""
    assert context["properties"]["user_id"]["description"]
    assert context["properties"]["session_label"]["default"] == ""
    assert context["properties"]["session_label"]["description"]

    failure = schema_by_name["always_fail"]
    assert failure.get("required", []) == []
    assert failure["properties"]["reason"]["default"] == "intentional failure"
    assert failure["properties"]["reason"]["description"]


def hide_mcp_context_injected_args_for_llm(tools: list[BaseTool]) -> list[BaseTool]:
    """隐藏 MCP context_probe 的 runtime 注入参数，避免发送给 LLM。

    目的：MCP server 侧为了接收 interceptor 注入，必须显式声明
    user_id/session_label；但 LLM 不应该看到这两个字段。
    手段：复制 LangChain MCP tool，只改 `args_schema`，从 context_probe schema
    中删除 user_id/session_label；工具执行 coroutine 保持不变。
    结果：发给 vLLM 的 tools schema 只包含 query，ToolNode 执行时 interceptor
    仍会把 runtime 中的 user_id/session_label 注入 MCP 请求。
    """
    hidden_args = {"user_id", "session_label"}
    redacted_tools: list[BaseTool] = []
    for tool_obj in tools:
        if tool_obj.name != "context_probe":
            redacted_tools.append(tool_obj)
            continue

        schema = copy.deepcopy(tool_obj.args_schema)
        properties = schema.get("properties", {})
        for arg_name in hidden_args:
            properties.pop(arg_name, None)
        schema["required"] = [
            arg_name
            for arg_name in schema.get("required", [])
            if arg_name not in hidden_args
        ]
        redacted_tools.append(tool_obj.model_copy(update={"args_schema": schema}))
    return redacted_tools


async def run_mcp_llm_runtime_hidden_graph(tools: list[BaseTool]) -> dict[str, Any]:
    """运行 start -> llm -> MCP ToolNode -> llm -> end 最小图并抓包。

    目的：用真实 vLLM 请求证明 MCP runtime 注入字段不会暴露给 LLM。
    手段：第一轮 LLM 命名调用 `context_probe`，抓包检查工具 schema；
    ToolNode 执行 MCP 工具时由 interceptor 注入 user_id/session_label；第二轮
    LLM 使用 `tool_choice='none'` 输出最终回答。
    结果：请求 schema 中没有 user_id/session_label；ToolMessage 中包含二者只是
    为了测试证明注入成功，生产工具不应把隐藏注入参数返回给 LLM。
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    capture_path = CAPTURE_DIR / "09_mcp_runtime_hidden_llm_graph_capture.json"
    capture = HttpCapture(capture_path)
    llm_visible_tools = hide_mcp_context_injected_args_for_llm(tools)
    async_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0),
        event_hooks=capture.async_hooks(),
    )

    async def call_llm(state: ExperimentState) -> dict[str, list[AIMessage]]:
        has_tool_result = any(
            isinstance(message, ToolMessage) for message in state["messages"]
        )
        tool_choice_value = "none" if has_tool_result else "context_probe"
        model_with_tools = ChatOpenAI(
            model=MODEL,
            base_url=OPENAI_BASE_URL,
            api_key=API_KEY,
            temperature=0.0,
            top_p=0.8,
            extra_body=VLLM_EXTRA_BODY,
            http_async_client=async_client,
        ).bind_tools(
            llm_visible_tools,
            tool_choice=tool_choice_value,
        )
        response = await model_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(ExperimentState, context_schema=ExperimentContext)
    builder.add_node("llm", call_llm)
    builder.add_node("tools", ToolNode(llm_visible_tools, handle_tool_errors=True))
    builder.add_edge(START, "llm")
    builder.add_conditional_edges(
        "llm", tools_condition, {"tools": "tools", "__end__": END}
    )
    builder.add_edge("tools", "llm")
    graph = builder.compile()

    try:
        result = await graph.ainvoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "你是 MCP runtime 隐藏参数测试助手。第一次必须调用 "
                            "context_probe；收到工具结果后用一句中文总结，不要再次调用工具。"
                        )
                    ),
                    HumanMessage(
                        content="请读取 MCP runtime 信息，query=隐藏参数验证。"
                    ),
                ],
                "session_label": "mcp-llm-hidden-session",
                "user_preferences": {"language": "zh"},
                "tool_events": [],
            },
            context=ExperimentContext(user_id="mcp-llm-user"),
            config={"recursion_limit": 6},
        )
    finally:
        await async_client.aclose()
        capture.save()

    schema = first_request_tool_schema(capture, "context_probe")
    properties = schema.get("parameters", {}).get("properties", {})
    tool_messages = latest_tool_messages(result)
    parsed_tool_messages = [
        json.loads(tool_message_text(message)) for message in tool_messages
    ]
    final_message = result["messages"][-1]

    assert "query" in properties
    assert "user_id" not in properties
    assert "session_label" not in properties
    assert any(item.get("user_id") == "mcp-llm-user" for item in parsed_tool_messages)
    assert any(
        item.get("session_label") == "mcp-llm-hidden-session"
        for item in parsed_tool_messages
    )
    assert isinstance(final_message, AIMessage)
    assert not final_message.tool_calls

    return {
        "capture_path": str(capture_path),
        "llm_request_count": len(capture.records),
        "llm_visible_context_probe_schema": schema,
        "hidden_args_visible_to_llm": {
            "user_id": "user_id" in properties,
            "session_label": "session_label" in properties,
        },
        "tool_messages": tool_messages,
        "parsed_tool_messages": parsed_tool_messages,
        "production_note": (
            "context_probe 回显 user_id/session_label 仅用于测试 interceptor 注入成功；"
            "生产 MCP 工具不应在 ToolMessage 中暴露这些后端注入的隐藏参数。"
        ),
        "final_message": final_message,
    }


def test_09_async_mcp_toolnode_compatibility() -> None:
    """测试目的：验证异步 MCP 工具 schema、runtime 隐藏注入、错误和并行能力。

    实验方法：
    1. 在测试函数内直接构造 `MultiServerMCPClient`，从本地 stdio MCP server
       建立持久 session 并加载工具。
    2. 检查 MCP 工具参数 schema 的 required/default/description 是否完整。
    3. 跑 start -> llm -> ToolNode -> llm -> end 最小图并抓包，证明
       user_id/session_label 不出现在发给 vLLM 的 tool schema 中，但工具执行时
       能通过 interceptor 注入。
    4. 用 ToolNode 调用 MCP `get_weather`/`get_population`，与传统 `@tool`
       的稳定 JSON 结果做一致性比较。
    5. 分别验证 MCP 工具错误处理和两个异步慢工具的并行执行耗时。
    6. streamable HTTP 无状态示例不进入本 pytest 主流程；同目录
       `http_mcp_server.py` 和 `http_mcp_call_example.py` 给出独立验证方式。

    实验结果：返回 MCP 参数 schema、vLLM 抓包路径、runtime 隐藏注入内容、
    错误 ToolMessage、并行耗时，以及 streamable HTTP 独立示例位置。
    其中 `context_probe` 回显隐藏注入参数只是测试探针行为；生产工具不应
    把这类后端隐藏参数放进 ToolMessage 再交给 LLM。
    """

    async def run_all() -> dict[str, Any]:
        client = MultiServerMCPClient(
            {
                "fixture": {
                    "transport": "stdio",
                    "command": "python",
                    "args": [str(MCP_SERVER_PATH)],
                    "env": {
                        **os.environ,
                        "FASTMCP_CHECK_FOR_UPDATES": "off",
                        "FASTMCP_LOG_LEVEL": "ERROR",
                        "FASTMCP_SHOW_SERVER_BANNER": "false",
                    },
                }
            },
            tool_interceptors=[inject_runtime_context],
        )
        async with client.session("fixture") as session:
            tools = await load_mcp_tools(
                session,
                tool_interceptors=[inject_runtime_context],
                server_name="fixture",
            )
            tool_names = sorted(tool.name for tool in tools)
            schema_by_name = mcp_tool_schema_summary(tools)
            assert_mcp_parameter_schema(schema_by_name)
            return {
                "loaded_tool_names": tool_names,
                "parameter_schema": schema_by_name,
                "session_mode": "persistent_client_session",
                "llm_runtime_hidden_graph": await run_mcp_llm_runtime_hidden_graph(
                    tools
                ),
                "equivalence_and_context": await _run_mcp_equivalence_and_context_test(
                    tools
                ),
                "error_handling": await _run_mcp_error_test(tools),
                "parallel_execution": await _run_mcp_parallel_test(tools),
                "streamable_http_reference": {
                    "server": "zhuxs_learn/03_tool/http_mcp_server.py",
                    "client_example": "zhuxs_learn/03_tool/http_mcp_call_example.py",
                    "verify": (
                        "先运行 `python zhuxs_learn/03_tool/http_mcp_server.py`，"
                        "再运行 `python zhuxs_learn/03_tool/http_mcp_call_example.py`。"
                    ),
                    "note": "HTTP stateless 示例独立于 LangGraph runtime/interceptor。",
                },
            }

    record_experiment_result("test_09_async_mcp_toolnode", asyncio.run(run_all()))


# =============================================================================
# 八、预留 LangGraph 工具循环实验
# =============================================================================


def test_10_langgraph_tool_loop_reserved() -> None:
    """测试目的：为后续完整 LangGraph 工具循环实验保留编号。

    实验方法：当前阶段聚焦 tool schema、ToolNode、custom node 和 MCP 工具能力；
    暂不运行 vLLM 驱动的完整 llm -> tools -> llm agent loop。

    实验结果：仅写入占位汇总，避免现阶段实验偏离工具调用主题。
    """
    record_experiment_result(
        "test_10_langgraph_tool_loop_reserved",
        {
            "status": "reserved",
            "reason": (
                "完整 vLLM + LangGraph 工具循环暂时留空；后续系统学习 LangGraph "
                "后再补端到端 agent loop 实验。"
            ),
        },
    )


def test_11_langchain_args_schema_required_parameter_over_prompt() -> None:
    """测试目的：验证 LangChain `@tool(args_schema=...)` 中 required 参数对 vLLM 的实际约束。

    实验方法：定义 `RequiredParameterProbeInput`，其中 `city` 和 `note` 没有
    默认值，`unit` 有默认值 `"celsius"`。使用命名函数调用
    `required_parameter_probe`，但 prompt 明确要求不要填写 `note`。

    实验结果：抓包中 `required == ["city", "note"]`；`unit` 在 properties 中但
    不在 required 中。实际 tool call 同时包含 `city` 和 `note`，说明在命名函数
    正常完成时，required 参数结构约束会压过“不要填写 note”的自然语言提示。
    """
    result = run_required_parameter_schema_experiment()
    record_experiment_result(
        "test_11_args_schema_required_parameter_over_prompt", result
    )


# 本文件不提供脚本直跑入口；统一使用 pytest 执行。
