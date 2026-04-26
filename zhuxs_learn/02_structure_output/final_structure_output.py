#!/usr/bin/env python3
"""
final_structure_output.py — 可直接运行的结构化输出参考实现

演示两种 JSON Schema 结构化输出方式，均针对本地 vLLM 服务：
  1. LangChain ChatOpenAI + with_structured_output(method="json_schema")
  2. OpenAI SDK 直接调用 + response_format={"type":"json_schema", ...}

核心特性：
  - 嵌套 Pydantic 模型，字段类型覆盖 str / int / float / bool / list / dict
  - 类级 docstring + 字段级 Field(description=...) 同时存在
  - System prompt 包含完整 JSON Schema 定义（因为 vLLM json_schema 引导解码模式下，
    schema 中的 description 字段对模型不可见，必须手动注入到 system prompt）
  - httpx 事件钩子抓取完整请求/响应，结果保存至当前目录的 .json 文件

运行方式：
  python final_structure_output.py

依赖（若缺少）：
  pip install openai langchain-openai pydantic httpx
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import openai
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# =============================================================================
# 一、vLLM 配置（硬编码，无需导入外部 config.py）
# =============================================================================

OPENAI_BASE_URL = "http://192.168.10.99:33627/v1"
MODEL = "qwen36-27b"
API_KEY = "fake-vllm"           # vLLM 不校验 key，填任意非空字符串即可

# vLLM 专有采样参数，经实验效果较好；通过 extra_body 透传给服务端
VLLM_EXTRA_BODY: dict = {
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    "top_k": 20,
    "min_p": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}

# 抓包结果保存路径（保存至当前脚本所在目录）
HERE = Path(__file__).resolve().parent
CAPTURE_LANGCHAIN = HERE / "capture_langchain.json"
CAPTURE_OPENAI    = HERE / "capture_openai.json"


# =============================================================================
# 二、Pydantic 模型定义（嵌套，字段类型全覆盖）
# =============================================================================

class Contributor(BaseModel):
    """对分析工作做出贡献的人员。

    从提供的文本中填写每个字段。
    缺少的姓名使用空字符串，未知的布尔标志使用 false，
    未知的整数计数使用 0。
    """

    name: str = Field(
        description="Contributor 的全名，例如 'Alan Turing'。"
    )
    institution: str = Field(
        description=(
            "Contributor 所属的机构或公司。"
            "如果文本中未明确提及，使用空字符串。"
        )
    )
    is_corresponding: bool = Field(
        description=(
            "如果该 Contributor 被明确标记为通讯作者（corresponding author），则为 true，"
            "否则为 false。"
        )
    )
    h_index: int = Field(
        description=(
            "Contributor 的学术 h-index，非负整数。"
            "如果文本中未提及，使用 0。"
        )
    )


class ResearchPaper(BaseModel):
    """从研究论文描述中提取结构化元数据。

    仔细从提供的文本中解析每个字段，并相应填充 JSON 对象：
    - 缺失的文本字段 → 空字符串 ""
    - 缺失的整数     → 0
    - 缺失的浮点数   → 0.0
    - 缺失的布尔值   → false
    - 缺失的列表     → 空列表 []
    - 缺失的字典条目 → 空字典 {}
    """

    title: str = Field(
        description="论文的完整标题，与文本中完全一致。"
    )
    abstract_word_count: int = Field(
        description=(
            "摘要的字数估计值，非负整数。"
            "如果未明确说明，请根据上下文估计。"
        )
    )
    impact_factor: float = Field(
        description=(
            "发表期刊的 impact factor，浮点数。"
            "如果未知或文本中未提及，使用 0.0。"
        )
    )
    is_open_access: bool = Field(
        description=(
            "如果论文可以免费在线获取（无付费墙），则为 true，"
            "否则为 false。"
        )
    )
    keywords: list[str] = Field(
        description=(
            "论文的主题关键词或学科标签列表。"
            "提取文本中出现的所有关键词。如果未找到，使用 []。"
        )
    )
    contributors: list[Contributor] = Field(
        description=(
            "所有具名作者或 Contributor 的列表，每个条目为一个 Contributor 对象。"
            "保留他们在文本中出现的顺序。"
        )
    )
    citation_counts: dict[str, int] = Field(
        description=(
            "每年获得的引用次数，以四位年份字符串为键。"
            '示例：{"2022": 31, "2023": 78}。如果没有引用数据，使用 {}。'
        )
    )


# =============================================================================
# 三、工具函数
# =============================================================================

def build_system_prompt(model_cls: type) -> str:
    """将完整 JSON Schema 序列化后注入 system prompt。

    vLLM 在 json_schema 引导解码模式下，schema 中的 description 字段
    仅用于结构约束（guided decoding），不会出现在模型的上下文窗口中。
    因此必须通过 system prompt 手动注入 schema（含所有 description），
    才能让模型了解每个字段的提取规则。

    详见：zhuxs_learn/02_structure_output/test_vllm_json_schema_description_visibility.py
    """
    schema_json = json.dumps(
        model_cls.model_json_schema(), indent=2, ensure_ascii=False
    )
    return (
        "You are a helpful assistant that always responds with JSON that matches this schema:\n"
        f"```json\n{schema_json}\n```\n\n"
        'Follow these rules:\n    1. Your entire response must be valid JSON that adheres to the schema above\n    2. Do not include any explanations, preambles, or text outside the JSON\n    3. Do not include markdown formatting such as ```json or ``` around the JSON\n    4. Make sure all required fields in the schema are included\n    5. Use the correct data types for each field as specified in the schema\n    6. If boolean values are required, use true or false (lowercase without quotes)\n    7. If integer values are required, don\'t use quotes around them\n\n Example of a good response format:\n    {"key1": "value1", "key2": 42, "key3": false}\n    '
    )


# =============================================================================
# 四、HTTP 抓包记录器
# =============================================================================

class HttpCapture:
    """httpx 事件钩子：捕获最近一次请求/响应，并写入 JSON 文件。

    用法示例：
        cap = HttpCapture(save_path=Path("capture.json"))
        client = httpx.Client(
            event_hooks=cap.hooks(),
            timeout=httpx.Timeout(120.0),
        )
        # ... 使用 client 发起请求 ...
        cap.save()   # 将 {"request": ..., "response": ...} 写入文件
    """

    def __init__(self, save_path: Path) -> None:
        self.save_path = save_path
        self._req: dict = {}
        self._resp: dict = {}

    # ── httpx 钩子 ──────────────────────────────────────────────────────────

    def _on_request(self, request: httpx.Request) -> None:
        body: object = None
        if request.content:
            try:
                body = json.loads(request.content.decode("utf-8"))
            except Exception:
                body = request.content.decode("utf-8", errors="replace")
        self._req = {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": body,
        }

    def _on_response(self, response: httpx.Response) -> None:
        response.read()          # 对流式响应必须先 read() 才能访问 .text
        body: object = None
        try:
            body = json.loads(response.text)
        except Exception:
            body = response.text
        self._resp = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": body,
        }

    def hooks(self) -> dict:
        """返回可直接传给 httpx.Client(event_hooks=...) 的钩子字典。"""
        return {
            "request":  [self._on_request],
            "response": [self._on_response],
        }

    def save(self) -> None:
        """将捕获的请求/响应写入 JSON 文件。"""
        payload = {"request": self._req, "response": self._resp}
        self.save_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  → 抓包已保存：{self.save_path}")


# =============================================================================
# 五、方式 1 — LangChain ChatOpenAI + with_structured_output
# =============================================================================

def run_langchain(user_text: str, system_prompt: str) -> ResearchPaper:
    """使用 LangChain with_structured_output(method='json_schema') 发起请求。

    LangChain 会自动把 Pydantic schema 转换为
    response_format={"type":"json_schema", "json_schema": {...}} 发给 vLLM，
    并在收到响应后自动做 Pydantic 验证。

    返回经验证的 ResearchPaper 实例。
    """
    cap = HttpCapture(CAPTURE_LANGCHAIN)

    llm = ChatOpenAI(
        model=MODEL,
        base_url=OPENAI_BASE_URL,
        api_key=API_KEY,
        temperature=0.7,
        top_p=0.8,
        extra_body=VLLM_EXTRA_BODY,
        http_client=httpx.Client(
            timeout=httpx.Timeout(120.0, connect=10.0),
            event_hooks=cap.hooks(),
        ),
    )

    # method="json_schema" 是 vLLM 上唯一可靠的结构化输出方式
    # method="json_mode" 会发送 {"type":"json_object"}，vLLM 不支持该格式
    structured_llm = llm.with_structured_output(
        ResearchPaper,
        method="json_schema",
        include_raw=False,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ]
    result: ResearchPaper = structured_llm.invoke(messages)  # type: ignore[assignment]
    cap.save()
    return result


# =============================================================================
# 六、方式 2 — OpenAI SDK 直接调用
# =============================================================================

def run_openai_sdk(user_text: str, system_prompt: str) -> ResearchPaper:
    """使用原生 OpenAI SDK + response_format json_schema 发起请求。

    手动构造 response_format 字段，发出与 LangChain 等价的请求体。
    适合不依赖 LangChain、需要精确控制请求参数的场景。

    返回经 Pydantic 验证的 ResearchPaper 实例。
    """
    cap = HttpCapture(CAPTURE_OPENAI)

    client = openai.OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=API_KEY,
        http_client=httpx.Client(
            timeout=httpx.Timeout(120.0, connect=10.0),
            event_hooks=cap.hooks(),
        ),
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ResearchPaper",
                "strict": True,
                "schema": ResearchPaper.model_json_schema(),
            },
        },
        temperature=0.7,
        top_p=0.8,
        extra_body=VLLM_EXTRA_BODY,
    )

    raw_json: str = response.choices[0].message.content  # type: ignore[assignment]
    result = ResearchPaper.model_validate(json.loads(raw_json))
    cap.save()
    return result


# =============================================================================
# 七、主程序
# =============================================================================

# 测试用输入：涵盖所有字段，方便验证提取结果
USER_TEXT = (
    '一篇题为"Efficient Attention Mechanisms for Long-Context LLMs"的论文 '
    "发表在 Journal of Machine Learning Research "
    "(JMLR, impact factor 6.8) 上。"
    "摘要大约包含 220 个词。"
    "该论文为开放获取（open access）。"
    "关键词：transformer, attention, long context, efficiency。"
    "作者：Yann LeCun 博士（Meta AI, 通讯作者, h-index 130）"
    "和 Yoshua Bengio 博士（Mila / Université de Montréal, h-index 155）。"
    "引用数据：2023 年 42 次引用，2024 年 189 次引用。"
)


def main() -> None:
    system_prompt = build_system_prompt(ResearchPaper)

    SEP = "=" * 70

    print(SEP)
    print("JSON Schema（发给 API 的实际 schema）：")
    print(json.dumps(ResearchPaper.model_json_schema(), indent=2, ensure_ascii=False))

    print(f"\n{SEP}")
    print("System Prompt（注入完整 schema，使模型能看到所有 description）：")
    print(system_prompt)

    # ── 方式 1：LangChain ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("【方式 1】LangChain ChatOpenAI + with_structured_output(method='json_schema')")
    print("-" * 70)
    lc_result = run_langchain(USER_TEXT, system_prompt)
    print("解析结果（Pydantic 实例）：")
    print(lc_result.model_dump_json(indent=2))

    # ── 方式 2：OpenAI SDK ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("【方式 2】OpenAI SDK 直接调用")
    print("-" * 70)
    sdk_result = run_openai_sdk(USER_TEXT, system_prompt)
    print("解析结果（Pydantic 实例）：")
    print(sdk_result.model_dump_json(indent=2))

    print(f"\n{SEP}")
    print("完成。抓包文件位置：")
    print(f"  LangChain  → {CAPTURE_LANGCHAIN}")
    print(f"  OpenAI SDK → {CAPTURE_OPENAI}")


if __name__ == "__main__":
    main()
