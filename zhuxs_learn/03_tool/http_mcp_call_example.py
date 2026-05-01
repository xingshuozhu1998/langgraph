#!/usr/bin/env python3
"""最小 streamable HTTP MCP 调用示例。

目的：和 `http_mcp_server.py` 配套，展示 LangChain 如何通过 HTTP
连接无状态 MCP server。
手段：构造 `MultiServerMCPClient`，使用官方文档中的 `transport="http"`，
并显式打开 session 后加载工具、调用 `get_weather`。
结果：打印加载到的工具名、工具 schema 和一次确定性工具调用结果。

验证方法：
    终端 A：python zhuxs_learn/03_tool/http_mcp_server.py
    终端 B：python zhuxs_learn/03_tool/http_mcp_call_example.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def output_to_text(output: Any) -> str:
    """把 MCP tool 的直接调用结果提取成 JSON 文本。

    目的：不同版本 adapter 可能返回字符串、content block 或 TextContent 对象。
    手段：只做最小兼容，避免把这些兼容逻辑塞回主 pytest 文件。
    结果：返回可 `json.loads` 的文本。
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        text_parts: list[str] = []
        for item in output:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                text_parts.append(str(item.text))
            else:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)
    return str(output)


async def main() -> None:
    """加载 HTTP MCP 工具并调用一次 `get_weather`。

    目的：证明 streamable HTTP 示例不需要 LangGraph runtime 或 interceptor。
    手段：显式使用 `client.session(...)` 控制 HTTP MCP 连接生命周期。
    结果：标准输出打印 schema 和业务结果，供人工检查或重定向保存。
    """
    client = MultiServerMCPClient(
        {
            "http_fixture": {
                "transport": "http",
                "url": "http://127.0.0.1:8000/mcp",
            }
        }
    )
    async with client.session("http_fixture") as session:
        tools = await load_mcp_tools(session, server_name="http_fixture")
        weather_tool = next(tool for tool in tools if tool.name == "get_weather")
        output = await weather_tool.ainvoke({"city": "北京", "unit": "celsius"})
        parsed_result = json.loads(output_to_text(output))

        print(
            json.dumps(
                {
                    "loaded_tool_names": sorted(tool.name for tool in tools),
                    "get_weather_schema": weather_tool.args,
                    "tool_result": parsed_result,
                    "note": "无状态 HTTP 示例不使用 LangGraph runtime，也不使用 MCP interceptor。",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
