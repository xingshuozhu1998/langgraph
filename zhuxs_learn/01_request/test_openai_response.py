import os

from langfuse import get_client
from langfuse.openai import openai as langfuse_openai
from openai import OpenAI


os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-34fb8f1a-302b-4994-95b6-651723563d4c"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-9012c5bb-b369-4fc7-adb1-668743d77cb9"
os.environ["LANGFUSE_BASE_URL"] = "http://192.168.10.5:13000"


VLLM_BASE_URL = "http://192.168.10.5:33527/v1"
MODEL = "qwen35-27b"
SYSTEM_PROMPT = (
    "你是一个非常精确的计算器，只输出计算结果。"
)
USER_PROMPT = "1 + 1 = "


def dump_response(label, response):
    print(f"\n==== {label} ====")
    print(f"输出文本: {response.output_text!r}")

    for index, item in enumerate(response.output):
        item_type = getattr(item, "type", type(item).__name__)
        print(f"输出[{index}].类型: {item_type}")
        print(item)


def create_response(client):
    return client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": USER_PROMPT}],
        instructions=SYSTEM_PROMPT,
        metadata={"someMetadataKey": "someValue"},
    )


langfuse_client = langfuse_openai.OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="fake-vllm",
)

raw_client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="fake-vllm",
)

langfuse_response = create_response(langfuse_client)
dump_response("langfuse responses", langfuse_response)

raw_response = create_response(raw_client)
dump_response("openai responses", raw_response)


# 短生命周期脚本：在退出前刷新待处理的 Langfuse 事件
get_client().flush()
