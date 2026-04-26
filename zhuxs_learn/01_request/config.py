# OPENAI_BASE_URL = "http://192.168.10.99:33527/v1"
# ANTHROPIC_BASE_URL = "http://192.168.10.99:33527"
# OPENAI_BASE_URL = "http://192.168.10.5:33527/v1"
OPENAI_BASE_URL = "http://192.168.10.99:33527/v1"
ANTHROPIC_BASE_URL = "http://192.168.10.5:33527"
MODEL = "qwen35-27b"

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

llm_response = ChatOpenAI(
    model=MODEL,
    base_url=OPENAI_BASE_URL,
    api_key="fake-vllm",
    use_responses_api=True,
    output_version="responses/v1",
    temperature=1.0,
    top_p=0.95,
    max_tokens=8192,
    extra_body={
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "top_k": 20,
        "min_p": 0.0,
    },
)

# "chat_template_kwargs": {"enable_thinking": False}在vllm的response api中不被支持，只有chat completion api才支持
# 但是只有response api的思考过程才会被记录在langfuse中，而chat completion api的思考过程不会被记录在langfuse中

llm_instruct = ChatOpenAI(
    model=MODEL,
    base_url=OPENAI_BASE_URL,
    api_key="fake-vllm",
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    max_tokens=8192,
    extra_body={
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}
        # "thinking_token_budget": 32
        # 只有vllm serve添加了--reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}'才能生效
        # zhuxs_learn/docker/vllm_langfuse_reasoning.md
    }
)

# from langchain_deepseek import ChatDeepSeek
# ChatDeepSeek继承了chatopenai，唯一的不同是需要将api_key换成api_base，以及输出中有reasoning


llm_think = ChatDeepSeek(
    model=MODEL,
    api_base=OPENAI_BASE_URL,
    api_key="fake-vllm",
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    max_tokens=8192,
    extra_body={
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": True}
        # "thinking_token_budget": 32
        # 只有vllm serve添加了--reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}'才能生效
        # zhuxs_learn/docker/vllm_langfuse_reasoning.md
    }
)

# langchain中可以用invoke的参数覆盖初始化时的参数，例如extra_body={"chat_template_kwargs": {"enable_thinking": False}}就会覆盖掉初始化时extra_body中的"chat_template_kwargs": {"enable_thinking": True}，从而在这个请求中关闭思考过程
# llm_think.invoke("光的速度是多少？", extra_body={"chat_template_kwargs": {"enable_thinking": False}})