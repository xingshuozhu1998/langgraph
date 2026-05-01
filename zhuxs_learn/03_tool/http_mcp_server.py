#!/usr/bin/env python3
"""最小 streamable HTTP MCP server 示例。

目的：单独展示 MCP 的 HTTP/stateless transport，不把端口、子进程和轮询逻辑
塞进主 pytest 实验文件。
手段：只暴露一个确定性 `get_weather` 工具，并用
`mcp.run(transport="streamable-http", stateless_http=True)` 启动服务。
结果：启动后可用 `http_mcp_call_example.py` 通过 LangChain MCP adapter 调用。

验证方法：
    终端 A：python zhuxs_learn/03_tool/http_mcp_server.py
    终端 B：python zhuxs_learn/03_tool/http_mcp_call_example.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("LangChain vLLM HTTP MCP Example")


CITY_DATA: dict[str, dict[str, Any]] = {
    "北京": {"temperature_c": 26, "weather": "sunny"},
    "上海": {"temperature_c": 24, "weather": "cloudy"},
    "Tokyo": {"temperature_c": 19, "weather": "rainy"},
}


def _json(data: dict[str, Any]) -> str:
    """用稳定 JSON 字符串返回工具结果，方便直接断言和阅读。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


# 实验工具：最小 HTTP MCP 天气查询。
# 目的：证明 streamable HTTP transport 可以无状态暴露 MCP 工具。
# 手段：`city` 是必填参数，`unit` 有默认值，并且两个参数都有 description。
# 结果：调用方能通过 LangChain MCP adapter 加载 schema 并调用工具。
@mcp.tool()
async def get_weather(
    city: Annotated[str, Field(description="城市名称，例如“北京”或“上海”。")],
    unit: Annotated[
        Literal["celsius", "fahrenheit"],
        Field(description="温度单位。默认 celsius，可选 fahrenheit。"),
    ] = "celsius",
) -> str:
    """查询确定性天气；这是 HTTP MCP 示例的唯一工具。"""
    await asyncio.sleep(0)
    data = CITY_DATA.get(city, CITY_DATA["北京"])
    temperature_c = int(data["temperature_c"])
    temperature = (
        temperature_c if unit == "celsius" else round(temperature_c * 9 / 5 + 32)
    )
    return _json(
        {
            "city": city,
            "source": "http_mcp_stateless_fixture",
            "temperature": temperature,
            "unit": unit,
            "weather": data["weather"],
        }
    )


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
        path="/mcp",
        stateless_http=True,
        show_banner=False,
    )
