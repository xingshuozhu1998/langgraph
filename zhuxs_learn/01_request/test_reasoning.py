import os

from anthropic import Anthropic
from openai import OpenAI


OPENAI_BASE_URL = "http://192.168.10.99:33527/v1"
ANTHROPIC_BASE_URL = "http://192.168.10.99:33527"
MODEL = "qwen35-27b"
SYSTEM_PROMPT = (
    "你是一个非常精确的计算器，只输出计算结果。"
)
USER_PROMPT = "1 + 1 = "


os.environ.setdefault("OPENAI_API_KEY", "fake-vllm")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake-vllm")


def summarize_openai_response(label, response):
    reasoning_blocks = [item for item in response.output if item.type == "reasoning"]
    message_blocks = [item for item in response.output if item.type == "message"]
    reasoning_text = ""
    if reasoning_blocks:
        reasoning_parts = reasoning_blocks[0].content or []
        reasoning_text = "\n".join(
            part.text for part in reasoning_parts if getattr(part, "text", None)
        )

    output_text = response.output_text or ""

    print(f"\n==== {label} ====")
    print(f"推理块数量: {len(reasoning_blocks)}")
    print(f"推理字符数: {len(reasoning_text)}")
    print(f"输出文本: {output_text!r}")
    print(f"消息块数量: {len(message_blocks)}")


def summarize_chat_completion(label, response):
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning", None)

    print(f"\n==== {label} ====")
    print(f"是否存在推理: {reasoning is not None}")
    print(f"推理字符数: {len(reasoning or '')}")
    print(f"推理内容: {reasoning}")
    print(f"内容: {message.content!r}")
    print(f"使用量: {response.usage}")


def summarize_anthropic_response(label, response):
    thinking_blocks = [block for block in response.content if block.type == "thinking"]
    text_blocks = [block for block in response.content if block.type == "text"]
    thinking_text = "\n".join(block.thinking for block in thinking_blocks)
    text_output = "\n".join(block.text for block in text_blocks)

    print(f"\n==== {label} ====")
    print(f"思考块数量: {len(thinking_blocks)}")
    print(f"思考字符数: {len(thinking_text)}")
    print(f"文本输出: {text_output!r}")
    print(f"使用量: {response.usage}")


def test_openai_responses_api():
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=os.environ["OPENAI_API_KEY"])

    # 这是通过 OpenAI Responses API 请求推理的最简洁方式
    default_response = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": USER_PROMPT}],
        instructions=SYSTEM_PROMPT,
        reasoning={"effort": "medium", "summary": "auto"},
    )
    summarize_openai_response("openai responses: 默认推理", default_response)

    # vLLM 文档中提到了 thinking_token_budget 用于 /v1/chat/completions
    # 在此部署中，通过 Responses API 传递它虽被接受，但在手动测试中
    # 未能以可靠方式禁用或明显缩短推理内容
    budget_response = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": USER_PROMPT}],
        instructions=SYSTEM_PROMPT,
        reasoning={"effort": "medium", "summary": "auto"},
        extra_body={"thinking_token_budget": 32},
    )
    summarize_openai_response("openai responses: 预算尝试", budget_response)

    # vLLM 文档中提到了通过 chat_template_kwargs 使用 enable_thinking 控制聊天补全
    # 在此部署中，通过 Responses API 传递它虽被接受，但响应中仍出现推理内容，
    # 因此将此视为不支持/不生效
    disabled_response = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": USER_PROMPT}],
        instructions=SYSTEM_PROMPT,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    summarize_openai_response("openai responses: 禁用尝试", disabled_response)


def test_openai_chat_completions_controls():
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=os.environ["OPENAI_API_KEY"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    # 这是 vLLM 文档中用于控制推理行为的方式
    # 对于 Qwen3 系列模型，当启用推理时，thinking 默认开启
    default_response = client.chat.completions.create(model=MODEL, messages=messages)
    summarize_chat_completion("chat completions: 默认", default_response)

    # 这在当前部署中可靠地禁用了推理
    disabled_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    summarize_chat_completion(
        "chat completions: 禁用思考", disabled_response
    )

    # 当服务器默认全局更改时，显式启用仍然有用
    enabled_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    summarize_chat_completion(
        "chat completions: 启用思考", enabled_response
    )

    # vLLM 文档中提到的思考预算控制在当前部署中也有效
    budget_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        extra_body={"thinking_token_budget": 8},
    )
    summarize_chat_completion("chat completions: 思考预算 8", budget_response)


def test_anthropic_sdk():
    client = Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )

    default_response = client.messages.create(
        model=MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT}],
        max_tokens=256,
    )
    summarize_anthropic_response("anthropic sdk: 默认思考", default_response)

    # Anthropic SDK 暴露了 thinking + budget_tokens，vLLM 接受此参数
    enabled_budget_response = client.messages.create(
        model=MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT}],
        max_tokens=256,
        thinking={"type": "enabled", "budget_tokens": 64},
    )
    summarize_anthropic_response(
        "anthropic sdk: 显式思考预算", enabled_budget_response
    )

    # SDK 也接受 thinking={"type": "disabled"}，但在当前 vLLM 部署中
    # 思考块仍然返回，因此将此禁用视为不支持或 Anthropic 兼容端点对此模型/服务器不生效
    disabled_response = client.messages.create(
        model=MODEL,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT}],
        max_tokens=256,
        thinking={"type": "disabled"},
    )
    summarize_anthropic_response("anthropic sdk: 禁用尝试", disabled_response)


def main():
    test_openai_responses_api()
    test_openai_chat_completions_controls()
    test_anthropic_sdk()


if __name__ == "__main__":
    main()
