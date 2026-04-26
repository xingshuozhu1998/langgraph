# %%
OPENAI_BASE_URL = "http://192.168.10.99:33527/v1"
ANTHROPIC_BASE_URL = "http://192.168.10.99:33527"
MODEL = "qwen35-27b"

import httpx
import json
from langchain_openai import ChatOpenAI
from openai import OpenAI

# 自定义请求/响应钩子
def log_request(request: httpx.Request) -> None:
    """在请求发送前调用，打印请求信息"""
    print("=" * 50)
    print(f"➡️ 请求: {request.method} {request.url}")
    print(f"Headers: {dict(request.headers)}")
    
    # 打印请求体（JSON格式化）
    if request.content:
        try:
            body = json.loads(request.content.decode("utf-8"))
            print(f"Body:\n{json.dumps(body, indent=2, ensure_ascii=False)}")
        except:
            print(f"Body: {request.content.decode('utf-8', errors='replace')}")
    print("=" * 50)

def log_response(response: httpx.Response) -> None:
    """在收到响应后调用"""
    print(f"⬅️ 响应: {response.status_code} {response.reason_phrase}")
    print(f"Headers: {dict(response.headers)}")
    # 确保可以读取响应内容
    response.read()
    try:
        body = json.loads(response.text)
        print(f"Body:\n{json.dumps(body, indent=2, ensure_ascii=False)[:2000]}")
    except:
        print(f"Body: {response.text[:1000]}")
    print("=" * 50)

# 创建自定义 httpx 客户端
custom_http_client = httpx.Client(
    timeout=httpx.Timeout(60.0, read=60.0, write=60.0, connect=10.0),
    event_hooks={
        "request": [log_request],
        "response": [log_response],
    },
)

# 使用自定义 http_client 初始化 ChatOpenAI
llm_response = ChatOpenAI(
    model=MODEL,
    base_url=OPENAI_BASE_URL,
    api_key="fake-vllm",
    http_client=custom_http_client,  # 注入自定义客户端
    use_responses_api=True,
    output_version="responses/v1",
    temperature=1.0,
    top_p=0.95,
    max_tokens=16,
    extra_body={
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "top_k": 20,
        "min_p": 0.0,
    },
)
# 这里经过httpx抓取，和openai SDK保持一致
# Body:
# {
#   "input": [
#     {
#       "content": "你好",
#       "role": "user",
#       "type": "message"
#     }
#   ],
#   "max_output_tokens": 16,
#   "model": "qwen35-27b",
#   "stream": false,
#   "temperature": 1.0,
#   "top_p": 0.95,
#   "repetition_penalty": 1.0,
#   "presence_penalty": 1.5,
#   "top_k": 20,
#   "min_p": 0.0
# }

# 使用示例
response = llm_response.invoke("你好")
print(response)


llm_instruct = ChatOpenAI(
    model=MODEL,
    base_url=OPENAI_BASE_URL,
    api_key="fake-vllm",
    http_client=custom_http_client,  # 注入自定义客户端
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}
        # "thinking_token_budget": 32
    }
)
# 这里经过httpx抓取，和openai SDK的chat completion保持一致
# Body:
# {
#   "messages": [
#     {
#       "content": "你好",
#       "role": "user"
#     }
#   ],
#   "model": "qwen35-27b",
#   "presence_penalty": 1.5,
#   "stream": false,
#   "temperature": 0.7,
#   "top_p": 0.8,
#   "repetition_penalty": 1.0,
#   "top_k": 20,
#   "min_p": 0.0,
#   "chat_template_kwargs": {
#     "enable_thinking": false
#   }
# }
# 使用示例
response = llm_instruct.invoke("你好")
print(response)

# ========== 使用 OpenAI SDK 初始化客户端和上面的langchain进行对比 ==========

client = OpenAI(
    base_url=OPENAI_BASE_URL,           # 你的 base URL
    api_key="fake-vllm",                # 你的 API key
    http_client=custom_http_client,     # 注入自定义 httpx 客户端
)


# ========== 4. 使用 Responses API和 chat completion 发送请求 ==========

try:
    # 对应 use_responses_api=True 和 output_version="responses/v1"
    response = client.responses.create(
        model=MODEL,                    # 你的模型名称
        input = [
            {
            "content": "你好",
            "role": "user",
            "type": "message"
            }
        ],                   # 用户输入
        temperature=1.0,
        top_p=0.95,
        max_output_tokens=16,           # 对应 max_tokens
        extra_body={                    # vLLM 特有的额外参数
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "top_k": 20,
            "min_p": 0.0,
        },
        stream=False
    )
    
    print("输出内容:", response.output_text)
    
    response = client.chat.completions.create(
        model=MODEL,                    # 你的模型名称
        messages=[
            {"role": "user", "content": "hello"}
        ],              # 用户输入
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
            "top_k": 20,
            "min_p": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}
            # "thinking_token_budget": 32
        }
    )

    print("输出内容:", response.choices[0].message)
finally:
    # 关闭 httpx 客户端
    custom_http_client.close()


# 可以用python xxx.py > capture.log 2>&1 来捕获日志输出到文件中，方便后续分析请求和响应的细节