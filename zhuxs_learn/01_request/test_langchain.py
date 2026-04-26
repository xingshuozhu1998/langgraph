import os

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler


os.environ.setdefault(
    "LANGFUSE_SECRET_KEY", "sk-lf-34fb8f1a-302b-4994-95b6-651723563d4c"
)
os.environ.setdefault(
    "LANGFUSE_PUBLIC_KEY", "pk-lf-9012c5bb-b369-4fc7-adb1-668743d77cb9"
)
os.environ.setdefault("LANGFUSE_BASE_URL", "http://192.168.10.5:13000")


OPENAI_BASE_URL = "http://192.168.10.99:33527/v1"
ANTHROPIC_BASE_URL = "http://192.168.10.99:33527"
MODEL = "qwen35-27b"
SYSTEM_PROMPT = "你是一个非常精确的计算器，只输出计算结果。"
USER_PROMPT = "1 + 1 = "


def print_result(label, message, trace_id):
    print(f"\n==== {label} ====")
    print(f"内容: {message.content!r}")
    print(f"附加参数: {message.additional_kwargs}")
    print(f"响应元数据: {message.response_metadata}")
    print(f"trace_id: {trace_id}")


def run_langchain_openai_responses():
    handler = CallbackHandler()

    # 使用 ChatOpenAI + Responses API 是让推理内容在 Langfuse 中干净展示的首选方式
    # vLLM 的 Responses API 返回标准内容块，包括 type="reasoning"
    llm = ChatOpenAI(
        model=MODEL,
        base_url=OPENAI_BASE_URL,
        api_key="fake-vllm",
        use_responses_api=True,
        output_version="responses/v1",
        temperature=0,
    )

    message = llm.invoke(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)],
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_tags": ["langchain", "openai", "responses"]},
        },
    )

    print_result("langchain-openai responses api", message, handler.last_trace_id)


def run_langchain_anthropic_messages():
    handler = CallbackHandler()

    # vLLM 也提供了兼容 Anthropic 的 /v1/messages 端点
    # ChatAnthropic 返回 Anthropic 风格的内容块，其中 thinking 是明确的
    llm = ChatAnthropic(
        model=MODEL,
        base_url=ANTHROPIC_BASE_URL,
        api_key="fake-vllm",
        max_tokens=512,
    )

    message = llm.invoke(
        USER_PROMPT,
        config={
            "callbacks": [handler],
            "metadata": {"langfuse_tags": ["langchain", "anthropic", "messages-api"]},
        },
        system=SYSTEM_PROMPT,
    )

    print_result("langchain-anthropic messages api", message, handler.last_trace_id)


def main():
    langfuse = get_client()
    # 这里故意不使用 langchain-deepseek
    # 虽然它可以暴露 reasoning_content，但 ChatOpenAI Responses API 和 ChatAnthropic
    # 都更直接，并且现在可以与这个 vLLM 部署一起工作，同时保留 Langfuse 可以显示的
    # 推理/思考块
    run_langchain_openai_responses()
    run_langchain_anthropic_messages()

    # 短生命周期脚本：在退出前刷新待处理的 Langfuse 事件
    langfuse.flush()


if __name__ == "__main__":
    main()
