#!/usr/bin/env python3
"""用于 LangChain MCP pytest 实验的 stdio 异步工具服务器。

本文件故意保持很小：只暴露和传统 `@tool` 实验等价的业务工具，
并额外提供慢速工具、错误工具、上下文探针工具，用于验证 MCP 工具
在 LangGraph `ToolNode` 中的异步并行、错误处理和 runtime 上下文兼容性。
streamable HTTP 示例已拆到同目录 `http_mcp_server.py` 和
`http_mcp_call_example.py`，避免干扰本文件的 stdio 教学主线。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("LangChain vLLM Tool Calling Experiments")


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


def _json(data: dict[str, Any]) -> str:
    """用稳定 JSON 字符串返回工具结果，方便和传统工具结果做一致性比较。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


# 实验工具：确定性天气查询（MCP 版本）。
# 目的：与传统 @tool 版 get_weather 返回相同 JSON，用于结果一致性校验。
# 手段：从预设 CITY_DATA 读取，asyncio.sleep(0) 保持异步接口一致性。
# 结果：返回含城市、温度（支持摄氏/华氏切换）、天气状况的稳定 JSON 字符串。
@mcp.tool()
async def get_weather(
    city: Annotated[str, Field(description="城市名称，例如“北京”或“上海”。")],
    unit: Annotated[
        Literal["celsius", "fahrenheit"],
        Field(description="温度单位。默认 celsius，可选 fahrenheit。"),
    ] = "celsius",
) -> str:
    """Get deterministic weather for a city."""
    await asyncio.sleep(0)
    data = CITY_DATA.get(city, CITY_DATA["北京"])
    temperature_c = int(data["temperature_c"])
    temperature = (
        temperature_c if unit == "celsius" else round(temperature_c * 9 / 5 + 32)
    )
    return _json(
        {
            "city": city,
            "source": "deterministic_fixture",
            "temperature": temperature,
            "unit": unit,
            "weather": data["weather"],
        }
    )


# 实验工具：确定性人口查询（MCP 版本）。
# 目的：与传统 @tool 版 get_population 返回相同 JSON，供 MCP 结果一致性校验使用。
# 手段：从预设 CITY_DATA 读取，asyncio.sleep(0) 保持异步接口一致性。
# 结果：返回含城市、人口（百万）的稳定 JSON 字符串。
@mcp.tool()
async def get_population(
    city: Annotated[str, Field(description="城市名称，例如“北京”或“上海”。")],
) -> str:
    """Get deterministic population for a city."""
    await asyncio.sleep(0)
    data = CITY_DATA.get(city, CITY_DATA["北京"])
    return _json(
        {
            "city": city,
            "population_millions": data["population_millions"],
            "source": "deterministic_fixture",
        }
    )


# 实验工具：可控延迟回显（MCP 版本），专门用于 MCP 并行执行耗时验证。
# 目的：在 ToolNode 中并发调用两个 MCP slow_echo，通过总耗时判断是否真正并行。
# 手段：asyncio.sleep 模拟 IO 阻塞，精确记录实际耗时写回结果。
# 结果：返回含标签、预期延迟、实际耗时的稳定 JSON 字符串。
@mcp.tool()
async def slow_echo(
    label: Annotated[str, Field(description="回显标签，用于区分并行工具调用。")],
    delay_seconds: Annotated[
        float,
        Field(description="模拟 IO 等待秒数。默认 1.2 秒。"),
    ] = 1.2,
) -> str:
    """Return a label after an async sleep."""
    start = time.perf_counter()
    await asyncio.sleep(delay_seconds)
    return _json(
        {
            "delay_seconds": delay_seconds,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "label": label,
            "source": "deterministic_fixture",
        }
    )


# 实验工具：运行时上下文回显探针（MCP 版本）。
# 目的：验证 LangGraph runtime context/state 能通过 MCPToolCallRequest interceptor 注入 MCP 工具参数。
# 手段：user_id 和 session_label 由 inject_runtime_context interceptor 动态填充，工具直接回显。
# 结果：返回含 user_id、session_label 的稳定 JSON 字符串，用于断言 runtime 信息注入成功。
# 注意：这是测试探针才回显隐藏参数；生产工具不应把后端注入的隐藏参数返回给 LLM。
@mcp.tool()
async def context_probe(
    query: Annotated[str, Field(description="要读取的运行时上下文主题。")],
    user_id: Annotated[
        str,
        Field(description="由 LangChain MCP interceptor 注入的用户 ID。"),
    ] = "",
    session_label: Annotated[
        str,
        Field(description="由 LangChain MCP interceptor 注入的会话标签。"),
    ] = "",
) -> str:
    """测试探针：回显 interceptor 注入字段，用于证明 runtime 注入生效。

    生产工具不应把 `user_id`、`session_label` 这类后端注入的隐藏参数返回给 LLM。
    """
    await asyncio.sleep(0)
    return _json(
        {
            "query": query,
            "session_label": session_label,
            "source": "deterministic_fixture",
            "user_id": user_id,
        }
    )


# 实验工具：故意抛异常（MCP 版本），验证 MCP 工具在 ToolNode 中的错误处理。
# 目的：确认 ToolNode(handle_tool_errors=True) 对 MCP 工具异常的处理与传统工具一致。
# 手段：直接 raise ValueError，模拟 MCP 工具执行失败的场景。
# 结果：ToolNode 返回 ToolMessage(status='error')，content 中包含 MCP 工具的异常信息。
@mcp.tool()
async def always_fail(
    reason: Annotated[
        str,
        Field(description="故意失败的原因文本，用于验证错误处理。"),
    ] = "intentional failure",
) -> str:
    """Raise an error so ToolNode error handling can be verified."""
    await asyncio.sleep(0)
    raise ValueError(f"MCP tool failed intentionally: {reason}")


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
