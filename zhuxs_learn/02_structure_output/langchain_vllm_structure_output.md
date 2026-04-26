# LangChain 与 vLLM 结构化输出分层说明

这份笔记回答的是一个比 `create_agent(..., response_format=...)` 更底层的问题：

- LangChain 到底是怎么做 structured output 的
- `function_calling`、`json_mode`、`json_schema` 的区别是什么
- vLLM 提供的约束解码和 LangChain 抽象分别对应哪一层
- 在后端是 vLLM 的前提下，应该优先用哪种方式

## TL;DR

- `create_agent(response_format=...)` 是最高层封装，重点是“让 agent 最终返回结构化结果”。
- `llm.with_structured_output(...)` 是更底层、也更适合单次结构化调用的入口。
- vLLM 的 `choice`、`regex`、`grammar`、`structural_tag` 属于更底层的约束解码能力，LangChain 的通用 structured output 并不会完整抽象这些能力。
- 如果你的目标只是“稳定输出一个 JSON 对象”，优先用 `with_structured_output(schema, method="json_schema")`。
- 如果你需要 `regex`、`grammar`、`choice` 这种更强约束，就不要只停留在 LangChain 的通用抽象，应该直接把 vLLM 的参数透传下去。

## 1. LangChain 的 structured output 分几层

从高到低，可以把 LangChain 和后端能力分成 4 层：

| 层级 | 入口 | 作用 |
| --- | --- | --- |
| 1 | `create_agent(response_format=...)` | 让 agent 的最终答案变成结构化结果 |
| 2 | `ToolStrategy` / `ProviderStrategy` | 显式指定 agent 用工具调用还是 provider 原生结构化输出 |
| 3 | `llm.with_structured_output(schema, method=...)` | 直接对单次模型调用做结构化输出 |
| 4 | vLLM `json_schema` / `regex` / `grammar` / `choice` | 推理阶段的底层约束解码 |

### 1.1 `create_agent` 这一层

在 `langchain_docs/build/oss/python/langchain/structured-output.mdx` 中，`create_agent` 的 `response_format` 支持：

- `ToolStrategy`
- `ProviderStrategy`
- 直接传 schema type
- `None`

它的核心目标不是“暴露底层解码能力”，而是“把 agent 最终结果收敛成结构化对象”。

如果直接传 schema，LangChain 会自动在 `ProviderStrategy` 和 `ToolStrategy` 之间做选择。问题是，这个自动选择依赖模型 profile。对于自定义 `base_url` 指向的 vLLM，这个能力信息不一定可靠，所以不要过度依赖自动推断。

### 1.2 `with_structured_output` 这一层

这是更适合单次结构化调用的 API。`langchain_docs/src/oss/langchain/models.mdx` 明确给出了 3 种 method：

- `json_schema`
- `function_calling`
- `json_mode`

如果你不是在做 agent 的最终输出，而只是想让某一次 LLM 调用返回结构化对象，这一层通常比 `create_agent` 更直接。

### 1.3 vLLM 约束解码这一层

这是最底层。这里已经不是“LangChain 帮你组织 structured output”，而是“推理引擎直接限制模型下一 token 能输出什么”。

这一层的能力最强，但也最绑定后端。

## 2. `with_structured_output` 的 3 种方式到底差在哪

`langchain_openai/chat_models/base.py` 里可以直接看到实现分支：

- `function_calling`：把 schema 转成 tool schema，然后强制模型产出 tool call
- `json_mode`：发送 `response_format={"type": "json_object"}`，只保证输出是合法 JSON
- `json_schema`：发送 `response_format={"type": "json_schema", ...}`，由服务端按 schema 约束输出

可以把它们理解成下面这样：

| 方式 | 本质 | 约束发生位置 | 强度 | 适合场景 |
| --- | --- | --- | --- | --- |
| `function_calling` | 把 schema 包装成一个 tool call | provider/tool-call 协议层 | 中 | 模型支持工具调用，但不一定支持原生 schema |
| `json_mode` | 要求模型返回 JSON | provider 返回格式层 + 客户端解析 | 较弱 | 只要合法 JSON，不强求严格 schema |
| `json_schema` | 原生 schema 约束输出 | provider 结构化输出层 | 强 | 想稳定拿到符合 schema 的 JSON 对象 |

### 2.1 `function_calling`

它不是约束解码意义上的“语法限制”，而是把输出转成工具调用协议。

优点：

- 通常兼容性最好
- 很多支持 tool calling 的模型都能用

缺点：

- 它本质上是“协议约束”，不是 `regex` / `grammar` 这种 token 级限制
- 输出形态是 tool call，不是普通文本 JSON

### 2.2 `json_mode`

它要求服务端返回 JSON，但不等于严格遵守你脑子里的 schema。

关键点：

- LangChain 这边会再用 `PydanticOutputParser` 或 `JsonOutputParser` 做解析
- schema 约束更多还是靠提示词和客户端解析
- 如果字段缺失、字段名漂移，仍然可能失败

所以它比纯 prompt + parser 强一点，但比 `json_schema` 弱很多。

### 2.3 `json_schema`

这是最接近“原生结构化输出”的模式。对于 OpenAI 兼容接口，它会发送 `response_format={"type": "json_schema", ...}`。

优点：

- 最接近“服务端保证输出符合 schema”
- 如果后端真支持 schema 约束，通常最稳

缺点：

- 依赖后端实际支持
- 不是所有 OpenAI 兼容服务都完整实现

对 vLLM 来说，如果你只是想稳定产出一个对象，这通常是第一选择。

## 3. 你提供的 vLLM 示例实际上展示了什么

文件：`/data/zhuxs/llm_agent/langgraph/zhuxs_learn/02_structure_output/vllm_structured_outputs.py`

这个示例展示了 5 种约束：

| 约束 | 位置 | 含义 |
| --- | --- | --- |
| `choice` | `PARAMS["choice"]` | 强制输出只能从若干候选值中选择 |
| `regex` | `PARAMS["regex"]` | 强制生成结果匹配正则 |
| `json_schema` | `PARAMS["json"]` | 强制输出满足 JSON Schema |
| `grammar` | `PARAMS["grammar"]` | 用文法约束可生成文本 |
| `structural_tag` | `PARAMS["structural_tag"]` | 用结构标签 + schema 约束特定片段 |

### 3.1 哪些是“解码层约束”

上面 5 种里，真正重要的是：它们都不是“先生成文本，再解析”。

它们都是由 vLLM 在服务端生成阶段直接限制输出空间，也就是常说的 guided decoding / constrained decoding。

这和 LangChain 的 output parser 是两回事。

### 3.2 `structural_tag` 不是标准 tool calling

`structural_tag` 看起来像函数调用：

```xml
<function=get_weather>{"city": "Boston"}</function>
```

但它不是 OpenAI 标准 tool calling，也不是 LangChain 的 `ToolStrategy`。

它更像是：

- 提示词里告诉模型要输出什么格式
- vLLM 再对指定标签内的 JSON 结构做约束

所以它和 `function_calling` 是“用途接近”，但机制不同。

## 4. LangChain 概念和 vLLM 能力的对应关系

| LangChain 概念 | 最接近的 vLLM 能力 | 说明 |
| --- | --- | --- |
| `ProviderStrategy(schema)` | `response_format={"type": "json_schema", ...}` | 前提是后端真的支持原生 schema 输出 |
| `with_structured_output(..., method="json_schema")` | vLLM `json_schema` | 最直接的一一对应 |
| `ToolStrategy(schema)` | 标准 tool calling | 不等于 `regex` / `grammar` / `choice` |
| `with_structured_output(..., method="function_calling")` | 标准工具调用协议 | 更偏协议层，不是 token 级约束 |
| `with_structured_output(..., method="json_mode")` | `json_object` 模式 | 只保证 JSON，不保证严格 schema |
| output parser | 无 | 完全是客户端后处理，不是服务端约束 |
| `extra_body={"structured_outputs": ...}` | `choice` / `regex` / `grammar` | 这是 vLLM 特有扩展，不在 LangChain 通用抽象里 |

## 5. 对你这个后端的建议

### 建议 1：普通对象提取优先用 `json_schema`

如果目标是抽取一个结构化对象，比如：

- 联系人信息
- 路由结果
- 分类结果
- 评分结果

优先用：

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str


llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="your-model",
)

structured_llm = llm.with_structured_output(
    ContactInfo,
    method="json_schema",
)

result = structured_llm.invoke("Extract contact info from John Doe, john@example.com, 555-1234")
print(result)
```

这是最贴近 vLLM 原生 schema 约束的 LangChain 写法。

### 建议 2：需要更强格式限制时，直接透传 vLLM 参数

如果你的需求是下面这些：

- 输出必须匹配某个正则
- 输出只能是几个枚举值之一
- 输出必须满足 CFG / EBNF 文法

那就不要只依赖 `with_structured_output`，而应该直接透传 vLLM 参数：

```python
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    model="your-model",
    extra_body={
        "structured_outputs": {
            "regex": r"[a-z0-9.]{1,20}@\w{6,10}\.com\n"
        }
    },
)

msg = llm.invoke("Generate an email address for Alan Turing at Enigma")
print(msg.content)
```

这个写法的本质不是 LangChain 的通用 structured output，而是“LangChain 负责发请求，vLLM 负责做约束解码”。

### 建议 3：在 agent 里不要过度依赖自动策略选择

如果你在 agent 里写：

```python
response_format=MySchema
```

LangChain 会尝试根据 model profile 自动判断走 `ProviderStrategy` 还是 `ToolStrategy`。但对自定义 `base_url` 的 vLLM，这个自动判断不一定准确。

更稳妥的方式是：

- 显式使用 `ProviderStrategy(MySchema)`
- 或者直接在 agent 的某个节点里调用 `with_structured_output(..., method="json_schema")`

### 建议 4：如果要混用 tools 和 structured output，要做实测

文档里明确提到，不同 OpenAI 兼容后端的能力差异很大。vLLM 集成页也只说“具体能力取决于被部署的模型”。

所以如果你打算同时使用：

- tool calling
- 最终结构化输出
- streaming
- reasoning

一定要按你的模型和 vLLM 版本做实测，不要只看抽象接口是否存在。

## 6. 一个实用的决策方式

可以按这个顺序选：

1. 只是想拿到一个稳定 JSON 对象
   - 用 `with_structured_output(schema, method="json_schema")`
2. 后端不支持原生 schema，但支持工具调用
   - 用 `with_structured_output(schema, method="function_calling")`
3. 只要求返回合法 JSON，不要求严格 schema
   - 用 `method="json_mode"`
4. 需要严格正则、文法、枚举、标签结构
   - 直接使用 vLLM `extra_body["structured_outputs"]`
5. 是 agent 最终答案需要结构化
   - 再考虑 `create_agent(response_format=...)`

## 7. 一句话总结几种方式的本质区别

- `create_agent`：面向 agent 最终返回值
- `with_structured_output`：面向单次模型调用
- `function_calling`：靠工具调用协议拿结构化数据
- `json_mode`：只保证返回 JSON
- `json_schema`：靠服务端原生 schema 约束输出
- vLLM `regex` / `grammar` / `choice`：推理阶段的更底层约束解码

## 8. 本次查看过的关键文件

- `langchain_docs/build/oss/python/langchain/structured-output.mdx`
- `langchain_docs/src/oss/langchain/models.mdx`
- `langchain_docs/src/oss/python/integrations/chat/openai.mdx`
- `langchain_docs/src/oss/python/integrations/chat/google_generative_ai.mdx`
- `langchain_docs/src/oss/python/integrations/chat/outlines.mdx`
- `langchain_docs/src/oss/python/integrations/chat/vllm.mdx`
- `/usr/local/lib/python3.12/site-packages/langchain_openai/chat_models/base.py`
- `/data/zhuxs/llm_agent/langgraph/zhuxs_learn/02_structure_output/vllm_structured_outputs.py`

其中最有用的几处是：

- `chat_models/base.py:2050-2312`，能直接看到 `with_structured_output` 对三种 method 的分支处理
- `chat_models/base.py:780-798` 和 `3002-3081`，说明 `extra_body` 就是给 vLLM 这类 OpenAI 兼容后端透传自定义参数用的
- `vllm_structured_outputs.py:94-213`，展示了 `choice`、`regex`、`json_schema`、`grammar`、`structural_tag` 五种约束方式

## 9. 最终建议

如果你的后端已经是 vLLM，并且你真正关心的是“约束输出质量”，那就把这两层分开用：

- 通用结构化对象：用 LangChain 的 `with_structured_output(..., method="json_schema")`
- 高级约束解码：直接使用 vLLM `structured_outputs` 扩展

不要把所有需求都塞进 `create_agent(response_format=...)`。它适合做 agent 最终结果收口，但不是理解和使用底层约束解码的最佳入口。
