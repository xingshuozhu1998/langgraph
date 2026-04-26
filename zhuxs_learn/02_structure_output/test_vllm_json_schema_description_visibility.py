# %% [markdown]
# ## 鲁棒实验：json_schema 的 description 字段对模型是否可见？
# 
# **实验假设**：vLLM 在 json_schema 模式下，字段的 `description` 内容对模型不可见。
# 
# **实验设计**：
# - **哨兵值（Sentinel）**：在字段 description 中嵌入特殊字符串，只有读到 description 的模型才会输出它
# - **语义消除**：使用不透明字段名（`f_name` / `f_city` / `f_answer`），消除字段名本身的提示作用
# - **对照组**：手动将所有字段 description 注入 system prompt，确保模型肯定能看到指令
# - **可验证性**：所有关键检查均可用 `assert` 自动验证
# 
# | 实验组 | 实验条件 | 预期结论 |
# |--------|---------|----------|
# | A | json_schema only，无 system prompt | 若哨兵/计算题失败 → description 不可见 |
# | B（对照）| json_schema + system prompt 注入 description | 应当通过所有检查 |

# %%
from pydantic import BaseModel, Field
import json, openai

# ─── 占位符（用常见英文词 + 唯一数字，避免 sentinel 这种生僻词）────────────────
# 这些字符串只在 description 中出现，模型不读 description 就不会输出它们
PLACEHOLDER_NAME  = "unknown_name_7749"
PLACEHOLDER_EMAIL = "no_email_7749@example.com"
PLACEHOLDER_CITY  = "missing_city_7749"
EXPECTED_ANSWER   = str(157 * 13)  # "2041"


class ExperimentRecord(BaseModel):
    """Extract structured information from the user's message.

    Important rule: when no person name is present in the user text,
    set the f_name field to the literal string 'unknown_name_7749'.
    """
    # ↑↑↑ 【实验1：类级 docstring 可见性】
    # docstring 在 model_json_schema() 中会成为顶层 "description" 字段
    # 占位规则只放在这里，不放进 f_name 的字段 description
    # → 若 f_name == 'unknown_name_7749' 则证明类级 docstring 对模型可见

    f_name: str = Field(
        description="The person's full name extracted from the user text."
        # 【实验1的承载字段】：消息中无 name；字段 description 不含占位规则
        # 规则全部来自 class docstring，用于检测类级 description 是否可见
    )
    f_email: str = Field(
        description=(
            "Extract the person's email address from the user text. "
            f"If no email is found, return exactly this string: '{PLACEHOLDER_EMAIL}'"
        )
        # 【基础提取测试】：消息中含 email；用于排除"模型/API 本身坏掉"
        # 此项必须在 A/B 两组都 PASS，否则结果不可信
    )
    f_city: str = Field(
        description=(
            "Extract the person's city from the user text. "
            f"If no city is found, return exactly this string: '{PLACEHOLDER_CITY}'"
        )
        # 【实验2：字段 description 中的"占位规则"是否可见】
        # 消息中无 city，占位规则只在字段 description 中
        # → 若 f_city == 'missing_city_7749' 则证明字段 description 对模型可见
    )
    f_answer: str = Field(
        description=(
            "Find the two standalone integers in the user message, "
            "multiply them together, and return ONLY the numeric result as a string."
        )
        # 【实验3：字段 description 中的"操作指令"是否可见】
        # 消息只给两个裸数字 157 和 13，未提及"乘/multiply/compute"
        # 若描述不可见，模型大概率默认相加得 170 或留空
        # → 若 f_answer == '2041' 则证明字段 description 对模型可见
    )


schema = ExperimentRecord.model_json_schema()
print("实际发给 API 的 JSON Schema：")
print(json.dumps(schema, indent=2, ensure_ascii=False))
print(f"\nf_answer 期望值：{EXPECTED_ANSWER}（= 157 × 13）")
print(f"docstring 进入 schema 顶层 description：{'description' in schema}")

# %%
OPENAI_BASE_URL = "http://192.168.10.99:33627/v1"
MODEL = "qwen36-27b"

def build_system_prompt_from_schema(model_cls: type) -> str:
    """将完整 JSON Schema 序列化后注入 system prompt。
    
    直接使用 model_json_schema() 的输出，确保类级 description（docstring）
    和所有字段级 description 都被包含，比手动逐字段拼接更完整。
    """
    schema_json = json.dumps(model_cls.model_json_schema(), indent=2, ensure_ascii=False)
    return (
        "You must return a JSON object conforming to the following JSON Schema. "
        "Follow all field descriptions strictly:\n\n"
        f"```json\n{schema_json}\n```"
    )


def call_structured_api(messages: list[dict]) -> openai.types.chat.ChatCompletion:
    """发起 json_schema 结构化输出请求。"""
    client = openai.OpenAI(base_url=OPENAI_BASE_URL, api_key="sk-xxx")
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "experiment_record",
                "schema": ExperimentRecord.model_json_schema(),
            },
        },
        extra_body={
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "top_k": 20,
            "min_p": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def parse_and_validate(response, label: str) -> dict[str, bool]:
    """解析响应并打印逐项检验结果，返回 {检查描述: bool} 字典。"""
    raw = response.choices[0].message.content
    data = json.loads(raw)

    checks = {
        "name  == placeholder    [规则在 class docstring 中]":      data.get("f_name")  == PLACEHOLDER_NAME,
        "email == 'bob@smith.com' [基础提取]":                       data.get("f_email") == "bob@smith.com",
        "city  == placeholder    [规则在字段 description 中]":       data.get("f_city")  == PLACEHOLDER_CITY,
        f"answer== '{EXPECTED_ANSWER}'(157*13) [乘法指令在字段 description 中]":
                                                                     data.get("f_answer") == EXPECTED_ANSWER,
    }

    width = 60
    print(f"\n{'='*width}")
    print(f"  实验: {label}")
    print(f"{'='*width}")
    print(f"  原始输出: {raw}")
    print(f"  解析字段: {data}")
    print()
    for desc, passed in checks.items():
        icon = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{icon}]  {desc}")

    name_ok   = checks["name  == placeholder    [规则在 class docstring 中]"]
    city_ok   = checks["city  == placeholder    [规则在字段 description 中]"]
    answer_ok = checks[f"answer== '{EXPECTED_ANSWER}'(157*13) [乘法指令在字段 description 中]"]
    print(f"\n  ➜ 类级 description{'【可见】✓' if name_ok else '【不可见】✗'}"
          f"  |  字段级 description{'【可见】✓' if (city_ok and answer_ok) else '【不可见】✗'}")
    return checks


print("辅助函数已定义：build_system_prompt_from_schema / call_structured_api / parse_and_validate")

# %%
# ─── 测试消息设计说明 ─────────────────────────────────────────────────────────
# 消息中只含 email + 裸数字，name 和 city 均不存在
USER_MESSAGE = "Contact: bob@smith.com. Extra numbers: 157 and 13."

print("用户消息：", USER_MESSAGE)
print()
print("测试矩阵（避免重复）：")
print(f"  f_name   -> 占位规则在【class docstring】，期望 {PLACEHOLDER_NAME!r}")
print(f"  f_email  -> 提取测试，期望 'bob@smith.com'")
print(f"  f_city   -> 占位规则在【字段 description】，期望 {PLACEHOLDER_CITY!r}")
print(f"  f_answer -> 乘法指令在【字段 description】，期望 {EXPECTED_ANSWER!r}")

# %%
# ─── 实验组 A：仅 json_schema，无 system prompt ──────────────────────────────
# 模型只能靠 schema 中的 description 字段理解规则
# 若 city/answer 两项均失败 → 证明 description 不可见

messages_A = [{"role": "user", "content": USER_MESSAGE}]
response_A = call_structured_api(messages_A)
results_A = parse_and_validate(response_A, "A: json_schema only（无 system prompt）")

# %%
# ─── 对照组 B：json_schema + 手动将 description 注入 system prompt ───────────
# 明确把每个字段的指令写入 system prompt，模型一定能看到
# 预期：city=SENTINEL, answer=703 均应通过

system_prompt_B = build_system_prompt_from_schema(ExperimentRecord)
print("注入的 system prompt：\n")
print(system_prompt_B)
print()

messages_B = [
    {"role": "system", "content": system_prompt_B},
    {"role": "user", "content": USER_MESSAGE},
]
response_B = call_structured_api(messages_B)
results_B = parse_and_validate(response_B, "B: json_schema + system prompt 注入 description")

# %%
# ─── 汇总对比 & Assertions ────────────────────────────────────────────────────
W = 75
print(f"\n{'='*W}")
print("  汇总对比")
print(f"{'='*W}")

all_keys = list(results_A.keys())
col_a, col_b = "A(schema-only)", "B(+sysprompt)"
print(f"  {'检查项':<55} {col_a:>14} {col_b:>14}")
print(f"  {'-'*55} {'-'*14} {'-'*14}")
for key in all_keys:
    a = "PASS ✓" if results_A[key] else "FAIL ✗"
    b = "PASS ✓" if results_B[key] else "FAIL ✗"
    print(f"  {key:<55} {a:>14} {b:>14}")

NAME_KEY   = "name  == placeholder    [规则在 class docstring 中]"
EMAIL_KEY  = "email == 'bob@smith.com' [基础提取]"
CITY_KEY   = "city  == placeholder    [规则在字段 description 中]"
ANSWER_KEY = f"answer== '{EXPECTED_ANSWER}'(157*13) [乘法指令在字段 description 中]"

class_desc_seen_A = results_A[NAME_KEY]
field_desc_seen_A = results_A[CITY_KEY] and results_A[ANSWER_KEY]
class_desc_seen_B = results_B[NAME_KEY]
field_desc_seen_B = results_B[CITY_KEY] and results_B[ANSWER_KEY]

print(f"\n{'='*W}")
print("  结论（分别检测【类级】与【字段级】description 可见性）")
print(f"{'='*W}")
print(f"  A（json_schema only）  : 类级 docstring {'可见✓' if class_desc_seen_A else '不可见✗'}  "
      f"|  字段 description {'可见✓' if field_desc_seen_A else '不可见✗'}")
print(f"  B（+system prompt）    : 类级 docstring {'可见✓' if class_desc_seen_B else '不可见✗'}  "
      f"|  字段 description {'可见✓' if field_desc_seen_B else '不可见✗'}")

# ─── 自动化 Assertions ────────────────────────────────────────────────────────
assert results_B[EMAIL_KEY],  "B: email 提取失败，检查模型或 API"
assert results_B[NAME_KEY],   "B: class docstring 中的占位规则未被遵循"
assert results_B[CITY_KEY],   "B: 字段 description 中的 city 占位规则未被遵循"
assert results_B[ANSWER_KEY], "B: 字段 description 中的乘法指令未被遵循"

print("\n✓ 对照组 B 所有 assert 通过")
if not (class_desc_seen_A and field_desc_seen_A):
    miss = []
    if not class_desc_seen_A: miss.append("类级 docstring")
    if not field_desc_seen_A: miss.append("字段级 description")
    print(f"\n最终结论：vLLM json_schema 模式下 [{ '、'.join(miss) }] 对模型不可见，需手动注入 system prompt")
else:
    print("\n最终结论：vLLM json_schema 模式下 description 全部可见")

# %% [markdown]
# ## Workaround 封装：`structured_chat_with_schema_prompt`
# 
# 针对 vLLM json_schema description 不可见问题的实用封装。  
# 自动将 `model_json_schema()` 注入 system prompt，使 description 指令对模型可见。

# %%
from typing import TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_VLLM_EXTRA_BODY_DEFAULTS = {
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
    "top_k": 20,
    "min_p": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}


def structured_chat_with_schema_prompt(
    model_cls: type[T],
    user_message: str,
    openai_client: openai.OpenAI,
    model_name: str,
    system_prompt_prefix: str = "",
    extra_body: dict | None = None,
) -> T:
    """json_schema 结构化输出 + 自动将 schema 注入 system prompt。

    vLLM 的 json_schema guided decoding 仅用 schema 约束输出格式，
    不会将 description 字段内容暴露给模型。调用此函数可自动将
    model_json_schema() 的完整 JSON（含类级 docstring 与字段 description）
    注入 system prompt，使所有指令对模型可见。

    Args:
        model_cls: Pydantic BaseModel 子类，docstring 与字段 description 即为对模型的指令。
        user_message: 用户消息文本。
        openai_client: 已配置好的 openai.OpenAI 实例。
        model_name: vLLM 上的模型名称。
        system_prompt_prefix: 可选的额外 system prompt 前缀（追加在 schema 之前）。
        extra_body: 传给 vLLM 的额外参数，默认使用常用 vLLM 参数。

    Returns:
        解析并验证后的 Pydantic model 实例。
    """
    schema_json = json.dumps(model_cls.model_json_schema(), indent=2, ensure_ascii=False)
    schema_block = (
        "You must return a JSON object conforming to the following JSON Schema. "
        "Follow all field descriptions strictly:\n\n"
        f"```json\n{schema_json}\n```"
    )
    system_content = f"{system_prompt_prefix}\n\n{schema_block}".lstrip()

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message},
    ]

    response = openai_client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": model_cls.__name__.lower(),
                "schema": model_cls.model_json_schema(),
            },
        },
        extra_body=extra_body if extra_body is not None else _VLLM_EXTRA_BODY_DEFAULTS,
    )

    raw = response.choices[0].message.content
    return model_cls.model_validate_json(raw)


# ─── 快速验证封装函数 ─────────────────────────────────────────────────────────
client = openai.OpenAI(base_url=OPENAI_BASE_URL, api_key="sk-xxx")
result = structured_chat_with_schema_prompt(
    model_cls=ExperimentRecord,
    user_message=USER_MESSAGE,
    openai_client=client,
    model_name=MODEL,
)
print("structured_chat_with_schema_prompt 返回结果：")
print(result.model_dump_json(indent=2))
print()
print(f"name  (docstring 规则): {'✓' if result.f_name  == PLACEHOLDER_NAME  else '✗'}  {result.f_name!r}")
print(f"email (基础提取):       {'✓' if result.f_email == 'bob@smith.com'    else '✗'}  {result.f_email!r}")
print(f"city  (字段 description): {'✓' if result.f_city  == PLACEHOLDER_CITY  else '✗'}  {result.f_city!r}")
print(f"answer (157*13):        {'✓' if result.f_answer == EXPECTED_ANSWER   else '✗'}  {result.f_answer!r}")


