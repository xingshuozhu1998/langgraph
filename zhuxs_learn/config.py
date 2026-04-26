# %%
OPENAI_BASE_URL = "http://192.168.10.99:33627/v1"
MODEL = "qwen36-27b"
import httpx
import json
from langchain_openai import ChatOpenAI

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

llm_instruct = ChatOpenAI(
    model=MODEL,
    base_url=OPENAI_BASE_URL,
    api_key="fake-vllm",
    http_client=custom_http_client,  # 注入自定义客户端
    # use_responses_api=True,
    # output_version="responses/v1",
    # output_version="v1",
    temperature=0.7,
    top_p=0.8,
    extra_body={
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}
        # "thinking_token_budget": 32
    }
)
