# =============================================================================
# 本文件是针对 LangChain + vLLM 结构化输出能力的抓包测试套件。
#
# 【如何运行测试】
#
# 运行本文件全部测试（在项目根目录执行）：
#   pytest -q zhuxs_learn/02_structure_output/test_langchain_vllm_capture.py
#
# 运行某一个测试函数（用 :: 指定函数名）：
#   pytest -q zhuxs_learn/02_structure_output/test_langchain_vllm_capture.py::test_vllm_regex_passthrough
#
# 用关键字过滤，运行名称包含某个词的所有测试（-k 参数）：
#   pytest -q zhuxs_learn/02_structure_output/test_langchain_vllm_capture.py -k "passthrough"
#   pytest -q zhuxs_learn/02_structure_output/test_langchain_vllm_capture.py -k "json_schema or json_mode"
#
# 常用选项：
#   -q          精简输出，只显示 .（通过）/ F（失败）和摘要
#   -v          详细输出，逐条打印测试名称和结果
#   -s          不捕获 print/日志，适合调试时查看标准输出
#   --tb=short  失败时只打印简短的错误堆栈（默认 long）
#
# 【测试目的】
# 通过拦截真实 HTTP 请求，把 LangChain 发给 vLLM（OpenAI 兼容接口）的
# 实际请求体/响应体落盘，验证"LangChain 的各种结构化输出写法最终会把
# 什么参数发给服务端"。
#
# 【运行了哪些测试，分别证明了什么】
#
# ① test_with_structured_output_pydantic_json_schema
#    — 用 Pydantic model + method="json_schema"
#    — 证明：LangChain 会把 Pydantic schema 转成 response_format.json_schema
#      发给服务端；解析结果是 PersonRecord 实例（有类型信息）
#
# ② test_with_structured_output_typeddict_json_schema
#    — 用 TypedDict + method="json_schema"
#    — 证明：LangChain 同样生成 json_schema 格式请求；但结果是 dict，
#      TypedDict 无运行时类型，LangChain 只把它当普通字典返回
#
# ③ test_with_structured_output_json_schema_dict_json_schema
#    — 直接传原始 JSON Schema dict + method="json_schema"
#    — 证明：也会正确生成 json_schema 请求，schema title 作为 name 字段
#
# ④ test_with_structured_output_pydantic_json_mode_validation
#    — 用 Pydantic StrictStr model + method="json_mode"，故意让模型返回
#      类型错误的 JSON（name 是 int，email 是 list）
#    — 证明：json_mode 只发 response_format={"type":"json_object"}，
#      服务端不保证 schema 合规；客户端 Pydantic 验证失败会抛
#      OutputParserException
#    — ⚠️ 重要：vLLM 本身不支持 response_format={"type":"json_object"}，
#      该格式仅被 DeepSeek 等云端厂商实现。即使服务端支持，json_mode
#      也只保证输出是合法 JSON，完全不保证符合任何 schema——这是它
#      与 json_schema 的本质差距，生产环境应始终优先使用 json_schema。
#
# ⑤ test_with_structured_output_typeddict_json_mode_no_validation
#    — 同样的类型错误 JSON，但 schema 改为 TypedDict
#    — 证明：TypedDict 无运行时类型验证，即使字段类型错误也不报错，
#      直接把原始 dict 透传出来——这是 json_mode + TypedDict 的已知盲区
#    — ⚠️ 双重风险：vLLM 不支持 json_object 格式 + TypedDict 没有验证，
#      实际上完全得不到任何保障
#
# ⑥ test_create_agent_auto_strategy_falls_back_to_tool_strategy
#    — create_agent(response_format=PersonRecord)，不指定策略
#    — 证明：对自定义 base_url 的 vLLM，自动策略会降级为 ToolStrategy，
#      实际请求里有 tools 字段、没有 response_format；结果仍是 PersonRecord
#
# ⑦ test_create_agent_provider_strategy_forces_response_format
#    — create_agent(response_format=ProviderStrategy(PersonRecord))
#    — 证明：显式 ProviderStrategy 强制走 json_schema 路径，
#      请求里 tools=[]、response_format.type=="json_schema"；
#      结果同样是 PersonRecord
#
# ⑧ test_vllm_regex_passthrough
#    — 通过 extra_body={"structured_outputs": {"regex": ...}} 透传 vLLM 专有参数
#    — 证明：LangChain 会原封不动地把 extra_body 里的内容放进请求体，
#      服务端约束解码由 vLLM 自己处理
#
# ⑨ test_vllm_json_schema_passthrough
#    — 通过 llm.bind(response_format=...) 手动透传完整的 json_schema 格式参数
#    — 证明：不经过 with_structured_output 也可以手动控制 response_format，
#      适合需要精确控制请求体格式的场景
#
# ⑩ test_vllm_grammar_passthrough
#    — 通过 extra_body={"structured_outputs": {"grammar": ...}} 透传 vLLM EBNF 文法约束
#    — 证明：LangChain 与 OpenAI SDK 都能原封不动地把 grammar 字符串发给服务端，
#      两侧请求体完全一致；vLLM 负责按文法约束生成输出
#
# ⑪ test_vllm_structural_tag_passthrough
#    — 通过 extra_body/response_format 透传 vLLM structural_tag 格式参数
#    — 证明：LangChain 侧用 extra_body 注入、OpenAI SDK 侧用 response_format 注入，
#      两种写法最终发出的请求体 response_format 字段完全相同；
#      structural_tag 可用于约束模型在指定标签对之间输出合法 JSON
#
# 【注意】8~11 四个 vllm_passthrough 测试均采用"双轨对比"模式：
# 同一约束参数同时用 LangChain 和原生 OpenAI SDK 各发一次请求，
# 并将两侧的实际请求体写入 comparison_summary.json 供对比验证，
# 确认 LangChain 的透传行为与 OpenAI SDK 完全等价。
# =============================================================================

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Annotated

import httpx
import openai
import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, StrictStr
from typing_extensions import TypedDict

# 当前文件所在目录
HERE = Path(__file__).resolve().parent
# 项目根目录（向上两级，用于加载 zhuxs_learn/config.py）
REPO_ROOT = HERE.parents[1]
# 每条测试用例的 HTTP 抓包结果存放目录
RESULT_ROOT = HERE / "httpx_result"

# ── 动态加载 zhuxs_learn/config.py ────────────────────────────────────────────
# 这里没有用普通 import，是因为 config.py 不在标准 Python 包路径里，
# 只能通过 importlib 按绝对路径加载。
_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "zhuxs_learn_config",
    REPO_ROOT / "zhuxs_learn" / "config.py",
)
assert _CONFIG_SPEC and _CONFIG_SPEC.loader
_CONFIG_MODULE = importlib.util.module_from_spec(_CONFIG_SPEC)
_CONFIG_SPEC.loader.exec_module(_CONFIG_MODULE)

# 从 config.py 中读取 vLLM 服务地址、模型名、以及预构造好的 LLM 实例
OPENAI_BASE_URL = _CONFIG_MODULE.OPENAI_BASE_URL
MODEL = _CONFIG_MODULE.MODEL
llm_instruct = _CONFIG_MODULE.llm_instruct

# ── 测试用 Prompt ──────────────────────────────────────────────────────────────

# 正常提取场景：让模型从自然语言中提取结构化数据
EXTRACTION_PROMPT = (
    "Extract structured data from this sentence: "
    "Ada Lovelace, ada@example.com, London."
)

# 故意返回类型不合规的 JSON：name 是 int、email 是 list、city 是 bool，
# 用来测试客户端解析器的类型验证是否生效
INVALID_JSON_PROMPT = (
    'Return exactly this JSON object and nothing else: '
    '{"name": 123, "email": ["ada@example.com"], "city": true}'
)

# 正则约束测试：让模型生成一个邮件地址，并以换行符结尾
REGEX_PROMPT = (
    "Generate an email address for Alan Turing at Enigma. "
    "The answer must end with a newline."
)
# 对应 vLLM guided decoding 的正则表达式，限制邮件格式及结尾换行
REGEX_PATTERN = r"[a-z0-9.]{1,20}@\w{6,10}\.com\n"

# 文法约束测试：复用 vllm_structured_outputs.py 中的 grammar 示例。
GRAMMAR_PROMPT = (
    "Generate an SQL query to show the 'username' and 'email' from the 'users' table."
)
GRAMMAR = """
root ::= select_statement

select_statement ::= "SELECT " column " from " table " where " condition

column ::= "col_1 " | "col_2 "

table ::= "table_1 " | "table_2 "

condition ::= column "= " number

number ::= "1 " | "2 "
"""

# structural_tag 约束测试：复用 vLLM 示例（zhuxs_learn/02_structure_output/vllm_structured_outputs.py）中的函数标签格式，但把问题收敛到一个城市。
STRUCTURAL_TAG_PROMPT = """
You have access to the following function to retrieve the weather in a city:

{
    "name": "get_weather",
    "parameters": {
        "city": {
            "param_type": "string",
            "description": "The city to get the weather for",
            "required": True
        }
    }
}

If a you choose to call a function ONLY reply in the following format:
<{start_tag}={function_name}>{parameters}{end_tag}
where

start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function
              argument value as value.
end_tag => `</function>`

Here is an example,
<function=example_function_name>{"example_name": "example_value"}</function>

Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line

You are a helpful assistant.

Given the previous instructions, what is the weather in Boston?
"""
STRUCTURAL_TAG_RESPONSE_FORMAT = {
    "type": "structural_tag",
    "structures": [
        {
            "begin": "<function=get_weather>",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "end": "</function>",
        }
    ],
    "triggers": ["<function="],
}


# ── Schema 定义 ────────────────────────────────────────────────────────────────

class PersonRecord(BaseModel):
    """用 Pydantic BaseModel 定义的人员记录 schema。

    这是最常用的结构化输出 schema 形式。
    with_structured_output 会把它的字段信息转成 JSON Schema 发给服务端，
    解析结果也会还原成 PersonRecord 实例（有类型信息）。
    """

    name: str = Field(description="The person's full name")
    email: str = Field(description="The person's email address")
    city: str = Field(description="The person's city")


class PersonRecordStrict(BaseModel):
    """使用 StrictStr 的严格版 Pydantic schema，专门用于验证失败的测试场景。

    StrictStr 要求字段必须是字符串类型，不接受 int/bool/list 等。
    配合 INVALID_JSON_PROMPT 使用，可以触发 OutputParserException，
    证明 json_mode 下客户端仍然会做 Pydantic 类型验证。
    """

    name: StrictStr = Field(description="The person's full name")
    email: StrictStr = Field(description="The person's email address")
    city: StrictStr = Field(description="The person's city")


class PersonRecordDict(TypedDict):
    """用 TypedDict 定义的人员记录 schema。

    TypedDict 只是类型注解，没有运行时验证能力。
    用它做 with_structured_output 时，LangChain 同样能生成正确的 json_schema 请求，
    但解析结果是普通 dict，而且即使字段类型与注解不符也不会报错。
    这种行为和 Pydantic 的区别是本文件若干测试的核心对比点。
    """

    name: Annotated[str, ..., "The person's full name"]
    email: Annotated[str, ..., "The person's email address"]
    city: Annotated[str, ..., "The person's city"]


# 直接用原始 dict 表示的 JSON Schema，不依赖任何 Python 类型系统。
# 用来测试"直接传 schema dict"这条路径，
# 结果中 schema title 会被当作 json_schema.name 字段透传给服务端。
PERSON_JSON_SCHEMA = {
    "title": "PersonRecordJsonSchema",
    "description": "Normalized person record extracted from text.",
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The person's full name",
        },
        "email": {
            "type": "string",
            "description": "The person's email address",
        },
        "city": {
            "type": "string",
            "description": "The person's city",
        },
    },
    "required": ["name", "email", "city"],
}

PASSTHROUGH_JSON_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "person_record_passthrough",
        "schema": PERSON_JSON_SCHEMA,
    },
}

# ── 工具函数 ───────────────────────────────────────────────────────────────────


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def _json_dump(path: Path, payload: Any) -> None:
    """把任意 Python 对象序列化为 JSON 文件，写入指定路径。

    使用 ensure_ascii=False 保留 Unicode 字符，
    sort_keys=True 让输出稳定便于 diff，
    文件末尾追加换行符符合 POSIX 约定。
    """
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _decode_json_bytes(content: bytes) -> Any:
    """把 HTTP 请求/响应的原始 bytes 解码为 Python 对象。

    优先尝试 JSON 解析，失败则以字符串形式返回，
    空内容返回 None。用于抓包时统一记录请求体/响应体。
    """
    if not content:
        return None
    text = content.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _normalize(value: Any) -> Any:
    """把各种 Python 对象递归地转换成可以 JSON 序列化的形式。

    处理了以下类型：
    - Pydantic BaseModel → model_dump(mode="json")
    - LangChain AIMessage → model_dump(mode="json")
    - dict / list / tuple → 递归处理
    - Path → 转字符串
    - Exception → {"type": ..., "message": ...}
    - 其他有 model_dump 的对象 → 尝试调用
    - str / int / float / bool / None → 原样返回
    - 其余 → repr()
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, AIMessage):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Exception):
        return {"type": value.__class__.__name__, "message": str(value)}
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个 dict，override 中的值覆盖 base 中对应的值。

    遇到两边都是 dict 的键时递归合并，而不是直接替换整个子 dict。
    用于把 llm_instruct 原有的 extra_body 与测试需要追加的参数合并在一起。
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class HttpRecorder:
    """HTTP 抓包记录器：同时充当 httpx 客户端工厂和请求/响应落盘工具。

    工作原理：
    - 内部持有一个 httpx.Client，并注册了 event_hooks。
    - 每次 LangChain 通过这个 client 发出请求，_on_request 和 _on_response
      会被自动触发，把请求/响应的关键信息序列化到磁盘，同时存入内存字典。
    - 每条测试用例对应一个独立的 case_dir 子目录，避免结果互相覆盖。

    字段说明：
    - case_name：用例名称，也是 httpx_result/ 下子目录的名称
    - case_dir：落盘目录
    - _counter：自增编号，保证同一用例多次请求的文件名不冲突
    - requests / responses：内存中的请求/响应快照，供测试断言使用
    - client：实际用于发送请求的 httpx.Client
    """

    def __init__(self, case_name: str) -> None:
        self.case_name = case_name
        self.case_dir = RESULT_ROOT / case_name
        # 确保落盘目录存在
        self.case_dir.mkdir(parents=True, exist_ok=True)
        # 清理当前 case 目录里的旧抓包，避免失败重跑时混入过期 JSON 文件。
        for path in self.case_dir.glob("*.json"):
            path.unlink()
        # 从 1 开始的序号，保证请求和响应文件名对齐
        self._counter = itertools.count(1)
        self.requests: dict[int, dict[str, Any]] = {}
        self.responses: dict[int, dict[str, Any]] = {}
        self.client = httpx.Client(
            timeout=httpx.Timeout(60.0, read=60.0, write=60.0, connect=10.0),
            event_hooks={
                "request": [self._on_request],
                "response": [self._on_response],
            },
        )

    def _sanitize_headers(self, headers: httpx.Headers) -> dict[str, str]:
        """过滤掉 Authorization 等敏感请求头，避免 API Key 明文落盘。"""
        return {
            key: value
            for key, value in headers.items()
            if key.lower() != "authorization"
        }

    def _on_request(self, request: httpx.Request) -> None:
        """httpx 请求钩子：在请求发出前触发，记录请求体并落盘。

        把序号 index 存入 request.extensions，供响应钩子配对使用。
        """
        index = next(self._counter)
        request.extensions["capture_index"] = index
        payload = {
            "index": index,
            "method": request.method,
            "url": str(request.url),
            "headers": self._sanitize_headers(request.headers),
            "body": _decode_json_bytes(request.content),
        }
        self.requests[index] = payload
        _json_dump(self.case_dir / f"{index:02d}_request.json", payload)

    def _on_response(self, response: httpx.Response) -> None:
        """httpx 响应钩子：在响应到达后触发，记录响应体并落盘。

        通过 request.extensions 里的 capture_index 与请求配对，
        保证请求和响应文件的序号一致（如 01_request.json / 01_response.json）。
        注意：必须先调用 response.read() 才能访问 response.content。
        """
        response.read()
        index = int(response.request.extensions["capture_index"])
        payload = {
            "index": index,
            "status_code": response.status_code,
            "reason_phrase": response.reason_phrase,
            "headers": self._sanitize_headers(response.headers),
            "body": _decode_json_bytes(response.content),
        }
        self.responses[index] = payload
        _json_dump(self.case_dir / f"{index:02d}_response.json", payload)

    def close(self) -> None:
        """关闭底层 httpx.Client，释放连接资源。测试结束后必须调用。"""
        self.client.close()


def _build_llm(case_name: str, *, extra_body: dict[str, Any] | None = None) -> tuple[ChatOpenAI, HttpRecorder]:
    """为指定用例构建一个绑定了抓包记录器的 ChatOpenAI 实例。

    参数说明：
    - case_name：用例名称，用于创建落盘子目录
    - extra_body：要追加到请求体的额外参数（如 vLLM 专有字段），
                  会与 llm_instruct 原有的 extra_body 深度合并

    返回值：(ChatOpenAI 实例, HttpRecorder 实例)
    测试结束后需要调用 recorder.close()。
    """
    recorder = HttpRecorder(case_name)
    # 从全局 llm_instruct 实例中继承 extra_body（如 vLLM 的 guided_decoding_backend 等）
    base_extra_body = copy.deepcopy(getattr(llm_instruct, "extra_body", {}) or {})
    # 如果本次测试有额外参数，深度合并而非直接覆盖，保证原有配置不丢失
    final_extra_body = (
        _merge_dict(base_extra_body, extra_body) if extra_body else base_extra_body
    )
    llm = ChatOpenAI(
        model=MODEL,
        base_url=OPENAI_BASE_URL,
        api_key="fake-vllm",          # vLLM 不校验 API Key，填任意非空字符串即可
        http_client=recorder.client,  # 注入抓包 client，替换默认的 httpx.Client
        max_retries=0,                # 抓包测试只需要一次请求，避免失败时产生重试噪音
        temperature=getattr(llm_instruct, "temperature", None),
        top_p=getattr(llm_instruct, "top_p", None),
        extra_body=final_extra_body,
    )
    return llm, recorder


def _build_openai_client(
    case_name: str,
) -> tuple[openai.OpenAI, HttpRecorder]:
    """为 OpenAI SDK 调用构建一个复用同款抓包逻辑的 client。"""
    recorder = HttpRecorder(case_name)
    client = openai.OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key="fake-vllm",
        http_client=recorder.client,
        max_retries=0,
    )
    return client, recorder


def _build_with_structured_summary(result: dict[str, Any]) -> dict[str, Any]:
    """从 with_structured_output 的原始结果中提取关键摘要字段。

    with_structured_output(include_raw=True) 返回的 dict 包含三个键：
    - raw：模型原始 AIMessage 响应
    - parsed：解析后的结构化对象（可能是 Pydantic 实例或 dict）
    - parsing_error：解析失败时的异常对象，成功时为 None

    这个函数把它们规范化，并额外记录 Python 类型名，
    方便测试用 == 断言，而不必深入比较对象内容。
    """
    parsed = result.get("parsed")
    return {
        "raw": _normalize(result.get("raw")),
        "parsed": _normalize(parsed),
        # 解析结果的 Python 类型名，如 "PersonRecord"、"dict"
        "parsed_python_type": None if parsed is None else type(parsed).__name__,
        "parsing_error": _normalize(result.get("parsing_error")),
        # 解析错误的异常类型名，如 "OutputParserException"
        "parsing_error_type": (
            None
            if result.get("parsing_error") is None
            else type(result["parsing_error"]).__name__
        ),
    }


def _run_case(
    case_name: str,
    runner: Callable[[ChatOpenAI], Any],
) -> tuple[dict[str, Any], HttpRecorder]:
    """通用的用例执行器：构建 LLM → 执行 runner → 把结果和抓包数据序列化落盘。

    参数说明：
    - case_name：用例名称，对应 httpx_result/ 下的子目录名
    - runner：接收 ChatOpenAI 实例并返回任意结果的可调用对象，
              通常是一个 lambda，封装了 with_structured_output / create_agent 等调用

    返回值：(包含结果和抓包快照的 summary dict, HttpRecorder 实例)

    执行逻辑：
    1. 构建绑定抓包 client 的 LLM 实例
    2. 调用 runner(llm) 得到结果
    3. 把结果、抓包数据一起序列化为 summary.json 写入 case_dir
    4. 如果 runner 抛异常，同样落盘错误信息，然后重新抛出
    5. 无论成功失败，finally 中关闭 httpx.Client

    summary 里的特殊处理：
    - 如果结果包含 "structured_response" 键，额外记录其 Python 类型名
    - 如果结果包含 "raw/parsed/parsing_error" 键，构建 with_structured_output 专用摘要
    """
    llm, recorder = _build_llm(case_name)
    try:
        result = runner(llm)
        summary = {
            "case_name": case_name,
            "status": "ok",
            "result_python_type": type(result).__name__,
            "result": _normalize(result),
            "requests": recorder.requests,
            "responses": recorder.responses,
        }
        # create_agent 返回的结果通常包含 structured_response 键，单独记录其类型
        if isinstance(result, dict) and "structured_response" in result:
            summary["structured_response_python_type"] = type(
                result["structured_response"]
            ).__name__
        # with_structured_output(include_raw=True) 的结果包含 raw/parsed/parsing_error
        if isinstance(result, dict) and {"raw", "parsed", "parsing_error"} <= set(result):
            summary["with_structured_output"] = _build_with_structured_summary(result)
        _json_dump(recorder.case_dir / "summary.json", summary)
        return summary, recorder
    except Exception as exc:  # pragma: no cover - saved for manual inspection
        # 异常时同样落盘，方便事后人工排查
        summary = {
            "case_name": case_name,
            "status": "error",
            "error": _normalize(exc),
            "requests": recorder.requests,
            "responses": recorder.responses,
        }
        _json_dump(recorder.case_dir / "summary.json", summary)
        raise
    finally:
        recorder.close()


def _run_passthrough_case(
    case_name: str,
    *,
    extra_body: dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    prompt: str,
    raise_errors: bool = True,
) -> tuple[dict[str, Any], HttpRecorder]:
    """专门用于"透传参数"测试的用例执行器。

    与 _run_case 的区别在于：
    - 不使用 with_structured_output，直接调用 llm.invoke(prompt)
    - 如果提供了 response_format，用 llm.bind() 把它附加到请求上
    - 如果提供了 extra_body，在构建 LLM 时注入（透传 vLLM 专有字段用）

    这种方式模拟"完全绕过 LangChain 结构化输出抽象，手动控制请求体"的场景，
    适合验证 vLLM 的 regex / grammar / choice 等底层约束解码参数是否能正确透传。
    用于 test_vllm_regex_passthrough 和 test_vllm_json_schema_passthrough 上面

    参数说明：
    - case_name：用例名称
    - extra_body：透传给 vLLM 的扩展字段，如 {"structured_outputs": {"regex": ...}}
    - response_format：手动指定的 response_format dict，如 {"type": "json_schema", ...}
    - prompt：直接传给 llm.invoke 的字符串 prompt
    """
    llm, recorder = _build_llm(case_name, extra_body=extra_body)
    try:
        # 如果有 response_format，用 bind 注入；否则直接使用原始 llm
        bound = llm.bind(response_format=response_format) if response_format else llm
        result = bound.invoke(prompt)
        summary = {
            "case_name": case_name,
            "status": "ok",
            "result_python_type": type(result).__name__,
            "result": _normalize(result),
            "requests": recorder.requests,
            "responses": recorder.responses,
        }
        _json_dump(recorder.case_dir / "summary.json", summary)
        return summary, recorder
    except Exception as exc:  # pragma: no cover - saved for manual inspection
        summary = {
            "case_name": case_name,
            "status": "error",
            "error": _normalize(exc),
            "requests": recorder.requests,
            "responses": recorder.responses,
        }
        _json_dump(recorder.case_dir / "summary.json", summary)
        if raise_errors:
            raise
        return summary, recorder
    finally:
        recorder.close()


def _run_openai_sdk_passthrough_case(
    case_name: str,
    *,
    extra_body: dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    prompt: str,
    raise_errors: bool = True,
) -> tuple[dict[str, Any], HttpRecorder]:
    """用原生 OpenAI SDK 发同一类请求，并用 HttpRecorder 抓包落盘。"""
    client, recorder = _build_openai_client(case_name)
    base_extra_body = copy.deepcopy(getattr(llm_instruct, "extra_body", {}) or {})
    final_extra_body = (
        _merge_dict(base_extra_body, extra_body) if extra_body else base_extra_body
    )
    try:
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": getattr(llm_instruct, "temperature", None),
            "top_p": getattr(llm_instruct, "top_p", None),
        }
        if final_extra_body:
            kwargs["extra_body"] = final_extra_body
        if response_format is not None:
            kwargs["response_format"] = response_format

        result = client.chat.completions.create(**kwargs)
        summary = {
            "case_name": case_name,
            "status": "ok",
            "result_python_type": type(result).__name__,
            "result": _normalize(result),
            "requests": recorder.requests,
            "responses": recorder.responses,
        }
        _json_dump(recorder.case_dir / "summary.json", summary)
        return summary, recorder
    except Exception as exc:  # pragma: no cover - saved for manual inspection
        summary = {
            "case_name": case_name,
            "status": "error",
            "error": _normalize(exc),
            "requests": recorder.requests,
            "responses": recorder.responses,
        }
        _json_dump(recorder.case_dir / "summary.json", summary)
        if raise_errors:
            raise
        return summary, recorder
    finally:
        recorder.close()


def _run_vllm_pair_case(
    case_name: str,
    *,
    prompt: str,
    langchain_extra_body: dict[str, Any] | None = None,
    langchain_response_format: dict[str, Any] | None = None,
    openai_extra_body: dict[str, Any] | None = None,
    openai_response_format: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """同一 vLLM 约束分别用 LangChain 与 OpenAI SDK 调用，并写入对比摘要。"""
    langchain_summary, _ = _run_passthrough_case(
        case_name,
        extra_body=langchain_extra_body,
        response_format=langchain_response_format,
        prompt=prompt,
        raise_errors=False,
    )
    openai_summary, _ = _run_openai_sdk_passthrough_case(
        f"{case_name}/openai_sdk",
        extra_body=openai_extra_body,
        response_format=openai_response_format,
        prompt=prompt,
        raise_errors=False,
    )
    comparison = {
        "case_name": case_name,
        "langchain_request_body": _first_request_body(langchain_summary),
        "openai_sdk_request_body": _first_request_body(openai_summary),
        "langchain_status": langchain_summary["status"],
        "openai_sdk_status": openai_summary["status"],
        "langchain_result_python_type": langchain_summary.get("result_python_type"),
        "openai_sdk_result_python_type": openai_summary.get("result_python_type"),
        "langchain_error": langchain_summary.get("error"),
        "openai_sdk_error": openai_summary.get("error"),
    }
    _json_dump(RESULT_ROOT / case_name / "comparison_summary.json", comparison)
    return langchain_summary, openai_summary


def _first_request_body(summary: dict[str, Any]) -> dict[str, Any]:
    """从 summary 中取出第一次 HTTP 请求的 body（即发给 vLLM 的实际请求体）。

    大多数测试场景只发一次请求，所以取 index=1 就够了。
    这是测试断言里最核心的入口：通过检查请求体，
    确认 LangChain 把什么参数实际发给了服务端。
    """
    return summary["requests"][1]["body"]


# ── 测试用例 ───────────────────────────────────────────────────────────────────


# ── 测试用例 ───────────────────────────────────────────────────────────────────

def test_with_structured_output_pydantic_json_schema() -> None:
    """测试：Pydantic BaseModel + method="json_schema"

    验证点：
    1. LangChain 把 PersonRecord 的字段信息转成了 response_format.json_schema 格式发给服务端
    2. 请求体中 json_schema.schema.properties.name.description 等字段正确携带
    3. 解析结果是 PersonRecord 实例（parsed_python_type == "PersonRecord"），
       说明 Pydantic 提供了运行时类型还原

    这是最推荐的单次结构化调用写法。
    """
    summary, _ = _run_case(
        "with_structured_output_pydantic_json_schema",
        lambda llm: llm.with_structured_output(
            PersonRecord,
            method="json_schema",
            include_raw=True,
        ).invoke(EXTRACTION_PROMPT),
    )

    body = _first_request_body(summary)
    assert body["response_format"]["type"] == "json_schema"
    assert (
        body["response_format"]["json_schema"]["schema"]["properties"]["name"]["description"]
        == "The person's full name"
    )
    assert summary["with_structured_output"]["parsed_python_type"] == "PersonRecord"


def test_with_structured_output_typeddict_json_schema() -> None:
    """测试：TypedDict + method="json_schema"

    验证点：
    1. TypedDict 同样能生成正确的 json_schema 格式请求
    2. json_schema.description 来自 TypedDict 的 docstring
    3. 解析结果是 dict（parsed_python_type == "dict"），
       因为 TypedDict 只是静态注解，无运行时类型信息，LangChain 无法还原为具体类

    对比 test_with_structured_output_pydantic_json_schema：
    请求体完全相同，但返回的 Python 类型不同。
    """
    summary, _ = _run_case(
        "with_structured_output_typeddict_json_schema",
        lambda llm: llm.with_structured_output(
            PersonRecordDict,
            method="json_schema",
            include_raw=True,
        ).invoke(EXTRACTION_PROMPT),
    )

    body = _first_request_body(summary)
    assert body["response_format"]["type"] == "json_schema"
    typed_dict_description = body["response_format"]["json_schema"]["description"]
    assert "TypedDict" in typed_dict_description
    assert "LangChain 同样能生成正确的 json_schema 请求" in typed_dict_description
    assert (
        body["response_format"]["json_schema"]["schema"]["properties"]["email"]["description"]
        == "The person's email address"
    )
    assert summary["with_structured_output"]["parsed_python_type"] == "dict"


def test_with_structured_output_json_schema_dict_json_schema() -> None:
    """测试：原始 JSON Schema dict + method="json_schema"

    验证点：
    1. 直接传 dict 形式的 schema，LangChain 同样能生成正确的 json_schema 请求
    2. schema 的 title 字段被用作请求体中的 json_schema.name
    3. 解析结果是 dict（因为没有 Pydantic 类型信息可还原）

    适用场景：schema 来自外部配置或动态生成，不方便定义 Pydantic 类的情况。
    """
    summary, _ = _run_case(
        "with_structured_output_json_schema_dict_json_schema",
        lambda llm: llm.with_structured_output(
            PERSON_JSON_SCHEMA,
            method="json_schema",
            include_raw=True,
        ).invoke(EXTRACTION_PROMPT),
    )

    body = _first_request_body(summary)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "PersonRecordJsonSchema"
    assert (
        body["response_format"]["json_schema"]["schema"]["properties"]["city"]["description"]
        == "The person's city"
    )
    assert summary["with_structured_output"]["parsed_python_type"] == "dict"


def test_with_structured_output_pydantic_json_mode_validation() -> None:
    """测试：Pydantic StrictStr + method="json_mode"，故意触发解析失败

    使用 INVALID_JSON_PROMPT 让模型返回类型不合规的 JSON：
    name 是 int、email 是 list、city 是 bool。

    验证点：
    1. method="json_mode" 发的是 response_format={"type":"json_object"}，
       而不是 json_schema——服务端只保证输出是合法 JSON，不保证符合任何 schema
    2. 请求体中没有 "tools" 字段（不是 function_calling 路径）
    3. 客户端收到响应后，Pydantic StrictStr 验证失败，
       parsing_error_type == "OutputParserException"

    ⚠️ 注意：vLLM 本身不支持 response_format={"type":"json_object"}。
    该参数格式只被 DeepSeek、OpenAI 等云端厂商实现，对本地 vLLM 服务
    发此参数会被忽略或报错，模型输出不受任何约束。
    即使在支持 json_mode 的服务端，它也只保证"合法 JSON"，不保证
    "符合 schema 的 JSON"——这是 json_mode 与 json_schema 的本质差距。

    结论：json_mode 在 vLLM 上无效，且即使在支持它的服务端也不如 json_schema，
    生产环境应始终优先使用 json_schema。
    """
    summary, _ = _run_case(
        "with_structured_output_pydantic_json_mode_invalid",
        lambda llm: llm.with_structured_output(
            PersonRecordStrict,
            method="json_mode",
            include_raw=True,
        ).invoke(INVALID_JSON_PROMPT),
    )

    body = _first_request_body(summary)
    assert body["response_format"] == {"type": "json_object"}
    assert "tools" not in body
    assert summary["with_structured_output"]["parsing_error_type"] == "OutputParserException"


def test_with_structured_output_typeddict_json_mode_no_validation() -> None:
    """测试：TypedDict + method="json_mode"，故意触发类型不合规但不报错的场景

    同样使用 INVALID_JSON_PROMPT，让模型返回类型错误的 JSON。

    验证点：
    1. 请求体同样是 response_format={"type":"json_object"}（json_mode）
    2. parsing_error_type 是 None——TypedDict 没有运行时验证，
       类型错误的字段被原封不动地放进 dict 里，不会抛异常
    3. parsed_python_type == "dict"，内容是原始 JSON 解析结果

    ⚠️ 双重风险叠加：
    - vLLM 不支持 response_format={"type":"json_object"}，发送该参数无效
    - TypedDict 没有运行时类型验证，类型错误静默透传
    调用方拿到的是完全没有保障的"脏数据"，是所有写法里最危险的组合。
    """
    summary, _ = _run_case(
        "with_structured_output_typeddict_json_mode_invalid",
        lambda llm: llm.with_structured_output(
            PersonRecordDict,
            method="json_mode",
            include_raw=True,
        ).invoke(INVALID_JSON_PROMPT),
    )

    body = _first_request_body(summary)
    assert body["response_format"] == {"type": "json_object"}
    assert summary["with_structured_output"]["parsing_error_type"] is None
    assert summary["with_structured_output"]["parsed_python_type"] == "dict"


def test_create_agent_auto_strategy_falls_back_to_tool_strategy() -> None:
    """测试：create_agent 自动策略选择，对 vLLM 降级为 ToolStrategy

    create_agent(response_format=PersonRecord) 不指定策略时，
    LangChain 会根据 model profile 自动决定走 ProviderStrategy 还是 ToolStrategy。

    验证点：
    1. 对自定义 base_url 的 vLLM，自动策略实际上走了 ToolStrategy：
       请求体里有 "tools" 字段、没有 "response_format" 字段
    2. 即使走的是工具调用路径，最终结果仍然是 PersonRecord 实例
    3. structured_response.name == "Ada Lovelace"（正确提取）

    结论：自动策略对 vLLM 不可靠，生产环境建议显式指定策略。
    """
    summary, _ = _run_case(
        "create_agent_auto_strategy_schema",
        lambda llm: create_agent(
            model=llm,
            tools=[],
            response_format=PersonRecord,
        ).invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": EXTRACTION_PROMPT,
                    }
                ]
            }
        ),
    )

    body = _first_request_body(summary)
    assert "tools" in body
    assert "response_format" not in body
    assert summary["structured_response_python_type"] == "PersonRecord"
    assert summary["result"]["structured_response"]["name"] == "Ada Lovelace"


def test_create_agent_provider_strategy_forces_response_format() -> None:
    """测试：create_agent 显式指定 ProviderStrategy，强制走 json_schema 路径

    使用 ProviderStrategy(PersonRecord) 明确告知 LangChain：
    "用服务端原生结构化输出，不要转成工具调用"。

    验证点：
    1. 请求体里有 response_format.type == "json_schema"（走了 provider 原生路径）
    2. tools 字段为空列表（没有工具调用）
    3. 最终结果仍然是 PersonRecord 实例，email 正确提取

    结论：如果 vLLM 部署的模型支持 json_schema，显式 ProviderStrategy 是
    在 agent 场景下最稳妥、最直接的结构化输出方式。
    """
    summary, _ = _run_case(
        "create_agent_provider_strategy_schema",
        lambda llm: create_agent(
            model=llm,
            tools=[],
            response_format=ProviderStrategy(PersonRecord),
        ).invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": EXTRACTION_PROMPT,
                    }
                ]
            }
        ),
    )

    body = _first_request_body(summary)
    assert body["response_format"]["type"] == "json_schema"
    assert body["tools"] == []
    assert summary["structured_response_python_type"] == "PersonRecord"
    assert summary["result"]["structured_response"]["email"] == "ada@example.com"


def test_vllm_regex_passthrough() -> None:
    """测试：通过 extra_body 把 vLLM 正则约束解码参数透传到请求体

    不使用任何 LangChain 结构化输出抽象，直接通过 extra_body 注入
    vLLM 专有字段 structured_outputs.regex，
    让 vLLM 在推理阶段用 guided decoding 约束输出必须匹配 REGEX_PATTERN。

    验证点：
    1. 请求体中 structured_outputs.regex 等于 REGEX_PATTERN（参数正确透传）
    2. LangChain 负责发请求，vLLM 负责约束解码，两者职责分离

    适用场景：需要严格格式约束（如邮件地址、日期格式、枚举值）
    且 LangChain 的通用抽象无法满足时，直接透传 vLLM 参数。
    """
    langchain_summary, openai_summary = _run_vllm_pair_case(
        "vllm_regex_passthrough",
        langchain_extra_body={"structured_outputs": {"regex": REGEX_PATTERN}},
        openai_extra_body={"structured_outputs": {"regex": REGEX_PATTERN}},
        prompt=REGEX_PROMPT,
    )

    langchain_body = _first_request_body(langchain_summary)
    openai_body = _first_request_body(openai_summary)
    assert langchain_body["structured_outputs"]["regex"] == REGEX_PATTERN
    assert openai_body["structured_outputs"]["regex"] == REGEX_PATTERN
    assert langchain_body["structured_outputs"] == openai_body["structured_outputs"]
    assert langchain_summary["status"] == "ok"
    assert openai_summary["status"] == "ok"


def test_vllm_json_schema_passthrough() -> None:
    """测试：通过 llm.bind(response_format=...) 手动透传完整 json_schema 参数

    不使用 with_structured_output，而是直接用 llm.bind() 把完整的
    response_format dict 注入请求体，绕过 LangChain 的 schema 转换逻辑。

    验证点：
    1. 请求体中 response_format.type == "json_schema"
    2. json_schema.name == "person_record_passthrough"（自定义名称被正确保留）

    适用场景：
    - 需要精确控制请求体格式，不希望 LangChain 做任何转换
    - schema 已经是标准 JSON Schema 格式，直接透传即可
    - 想验证服务端是否支持某个特定的 json_schema 参数格式
    """
    langchain_summary, openai_summary = _run_vllm_pair_case(
        "vllm_json_schema_passthrough",
        langchain_response_format=PASSTHROUGH_JSON_RESPONSE_FORMAT,
        openai_response_format=PASSTHROUGH_JSON_RESPONSE_FORMAT,
        prompt=EXTRACTION_PROMPT,
    )

    langchain_body = _first_request_body(langchain_summary)
    openai_body = _first_request_body(openai_summary)
    assert langchain_body["response_format"]["type"] == "json_schema"
    assert openai_body["response_format"]["type"] == "json_schema"
    assert (
        langchain_body["response_format"]["json_schema"]["name"]
        == openai_body["response_format"]["json_schema"]["name"]
        == "person_record_passthrough"
    )
    assert langchain_summary["status"] == "ok"
    assert openai_summary["status"] == "ok"


def test_vllm_grammar_passthrough() -> None:
    """测试：LangChain 与 OpenAI SDK 都能透传 vLLM grammar 约束。"""
    langchain_summary, openai_summary = _run_vllm_pair_case(
        "vllm_grammar_passthrough",
        langchain_extra_body={"structured_outputs": {"grammar": GRAMMAR}},
        openai_extra_body={"structured_outputs": {"grammar": GRAMMAR}},
        prompt=GRAMMAR_PROMPT,
    )

    langchain_body = _first_request_body(langchain_summary)
    openai_body = _first_request_body(openai_summary)
    assert langchain_body["structured_outputs"]["grammar"] == GRAMMAR
    assert openai_body["structured_outputs"]["grammar"] == GRAMMAR
    assert langchain_body["structured_outputs"] == openai_body["structured_outputs"]
    assert langchain_summary["status"] == "ok"
    assert openai_summary["status"] == "ok"


def test_vllm_structural_tag_passthrough() -> None:
    """测试：LangChain 与 OpenAI SDK 都能发送 vLLM structural_tag response_format。"""
    langchain_summary, openai_summary = _run_vllm_pair_case(
        "vllm_structural_tag_passthrough",
        langchain_extra_body={"response_format": STRUCTURAL_TAG_RESPONSE_FORMAT},
        openai_response_format=STRUCTURAL_TAG_RESPONSE_FORMAT,
        prompt=STRUCTURAL_TAG_PROMPT,
    )

    langchain_body = _first_request_body(langchain_summary)
    openai_body = _first_request_body(openai_summary)
    assert langchain_body["response_format"]["type"] == "structural_tag"
    assert openai_body["response_format"]["type"] == "structural_tag"
    assert langchain_body["response_format"] == openai_body["response_format"]
    assert langchain_summary["status"] == "ok"
    assert openai_summary["status"] == "ok"
