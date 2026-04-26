> ## 文档索引
> 获取完整的文档索引请访问：https://docs.langchain.com/llms.txt
> 在进一步探索之前，可使用此文件发现所有可用页面。

# 结构化输出

结构化输出允许智能体以特定且可预测的格式返回数据。无需解析自然语言响应，你可以直接获得以 JSON 对象、[Pydantic 模型](https://docs.pydantic.dev/latest/concepts/models/#basic-model-usage) 或 dataclass 形式呈现的结构化数据，供你的应用程序直接使用。

<Tip>
  本页介绍使用 `create_agent` 的智能体结构化输出。如需在模型上直接使用结构化输出（在智能体之外），请参阅 [Models - Structured output](/oss/python/langchain/models#structured-output)。
</Tip>

LangChain 的 [`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) 会自动处理结构化输出。用户设置所需的结构化输出模式，当模型生成结构化数据时，它会被捕获、验证，并返回到智能体状态的 `'structured_response'` 键中。

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
def create_agent(
    ...
    response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
        None,
    ]
```

## 响应格式

使用 `response_format` 来控制智能体如何返回结构化数据：

* **`ToolStrategy[StructuredResponseT]`**：使用工具调用（tool calling）进行结构化输出
* **`ProviderStrategy[StructuredResponseT]`**：使用提供商原生的结构化输出
* **`type[StructuredResponseT]`**：模式类型 — 根据模型能力自动选择最佳策略
* **`None`**：未显式请求结构化输出

当直接提供模式类型时，LangChain 会自动选择：

* 如果所选模型和提供商支持原生结构化输出（例如 [OpenAI](/oss/python/integrations/providers/openai)、[Anthropic (Claude)](/oss/python/integrations/providers/anthropic) 或 [xAI (Grok)](/oss/python/integrations/providers/xai)），则使用 `ProviderStrategy`。
* 对于所有其他模型，使用 `ToolStrategy`。

<Note>
  如果使用 `langchain>=1.1`，原生结构化输出功能的支持会动态从模型的 [profile data](/oss/python/langchain/models#model-profiles) 中读取。如果数据不可用，请使用其他条件或手动指定：

  ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  custom_profile = {
      "structured_output": True,
      # ...
  }
  model = init_chat_model("...", profile=custom_profile)
  ```

  如果指定了工具，则模型必须同时支持工具和结构化输出。
</Note>

结构化响应会在智能体的最终状态的 `structured_response` 键中返回。

## 提供商策略（Provider strategy）

部分模型提供商通过其 API 原生支持结构化输出（例如 OpenAI、xAI (Grok)、Gemini、Anthropic (Claude)）。这是可用时最可靠的方法。

要使用此策略，请配置 `ProviderStrategy`：

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    strict: bool | None = None
```

<Info>
  `strict` 参数需要 `langchain>=1.2`。
</Info>

<ParamField path="schema" required>
  定义结构化输出格式的模式。支持：

  * **Pydantic 模型**：带有字段验证的 `BaseModel` 子类。返回已验证的 Pydantic 实例。
  * **Dataclasses**：带有类型注解的 Python dataclass。返回字典（dict）。
  * **TypedDict**：类型化字典类。返回字典（dict）。
  * **JSON Schema**：带有 JSON Schema 规范的字典。返回字典（dict）。
</ParamField>

<ParamField path="strict">
  可选的布尔参数，用于启用严格的模式遵守。部分提供商支持（例如 [OpenAI](/oss/python/integrations/chat/openai) 和 [xAI](/oss/python/integrations/chat/xai)）。默认为 `None`（禁用）。
</ParamField>

当你直接将模式类型传递给 [`create_agent.response_format`](https://reference.langchain.com/python/langchain/agents/factory/create_agent) 且模型支持原生结构化输出时，LangChain 会自动使用 `ProviderStrategy`：

<CodeGroup>
  ```python Pydantic Model theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from pydantic import BaseModel, Field
  from langchain.agents import create_agent


  class ContactInfo(BaseModel):
      """Contact information for a person."""
      name: str = Field(description="The name of the person")
      email: str = Field(description="The email address of the person")
      phone: str = Field(description="The phone number of the person")

  agent = create_agent(
      model="gpt-5.4",
      response_format=ContactInfo  # Auto-selects ProviderStrategy
  )

  result = agent.invoke({
      "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
  })

  print(result["structured_response"])
  # ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
  ```

  ```python Dataclass theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from dataclasses import dataclass
  from langchain.agents import create_agent


  @dataclass
  class ContactInfo:
      """Contact information for a person."""
      name: str # The name of the person
      email: str # The email address of the person
      phone: str # The phone number of the person

  agent = create_agent(
      model="gpt-5.4",
      tools=tools,
      response_format=ContactInfo  # Auto-selects ProviderStrategy
  )

  result = agent.invoke({
      "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
  })

  result["structured_response"]
  # {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
  ```

  ```python TypedDict theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from typing_extensions import TypedDict
  from langchain.agents import create_agent


  class ContactInfo(TypedDict):
      """Contact information for a person."""
      name: str # The name of the person
      email: str # The email address of the person
      phone: str # The phone number of the person

  agent = create_agent(
      model="gpt-5.4",
      tools=tools,
      response_format=ContactInfo  # Auto-selects ProviderStrategy
  )

  result = agent.invoke({
      "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
  })

  result["structured_response"]
  # {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
  ```

  ```python JSON Schema theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent


  contact_info_schema = {
      "type": "object",
      "description": "Contact information for a person.",
      "properties": {
          "name": {"type": "string", "description": "The name of the person"},
          "email": {"type": "string", "description": "The email address of the person"},
          "phone": {"type": "string", "description": "The phone number of the person"}
      },
      "required": ["name", "email", "phone"]
  }

  agent = create_agent(
      model="gpt-5.4",
      tools=tools,
      response_format=ProviderStrategy(contact_info_schema)
  )

  result = agent.invoke({
      "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
  })

  result["structured_response"]
  # {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
  ```
</CodeGroup>

提供商原生结构化输出由于模型提供商会强制执行模式，因此具有高可靠性和严格的验证。在可用时优先使用。

<Note>
  如果提供商对你的模型选择原生支持结构化输出，那么写 `response_format=ProductReview` 与写 `response_format=ProviderStrategy(ProductReview)` 在功能上是等效的。

  无论哪种情况，如果不支持结构化输出，智能体都会回退到工具调用策略。
</Note>

## 工具调用策略（Tool calling strategy）

对于不支持原生结构化输出的模型，LangChain 使用工具调用来实现相同的结果。这适用于所有支持工具调用的模型（大多数现代模型）。

要使用此策略，请配置 `ToolStrategy`：

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None
    handle_errors: Union[
        bool,
        str,
        type[Exception],
        tuple[type[Exception], ...],
        Callable[[Exception], str],
    ]
```

<ParamField path="schema" required>
  定义结构化输出格式的模式。支持：

  * **Pydantic 模型**：带有字段验证的 `BaseModel` 子类。返回已验证的 Pydantic 实例。
  * **Dataclasses**：带有类型注解的 Python dataclass。返回字典（dict）。
  * **TypedDict**：类型化字典类。返回字典（dict）。
  * **JSON Schema**：带有 JSON Schema 规范的字典。返回字典（dict）。
  * **Union 类型**：多种模式选项。模型会根据上下文选择最合适的模式。
</ParamField>

<ParamField path="tool_message_content">
  生成结构化输出时返回的工具消息的自定义内容。
  如果未提供，则默认显示一条展示结构化响应数据的消息。
</ParamField>

<ParamField path="handle_errors">
  结构化输出验证失败的错误处理策略。默认为 `True`。

  * **`True`**：使用默认错误模板捕获所有错误
  * **`str`**：使用此自定义消息捕获所有错误
  * **`type[Exception]`**：仅使用默认消息捕获此异常类型
  * **`tuple[type[Exception], ...]`**：仅使用默认消息捕获这些异常类型
  * **`Callable[[Exception], str]`**：返回错误消息的自定义函数
  * **`False`**：不进行重试，让异常向上传播
</ParamField>

<CodeGroup>
  ```python Pydantic Model theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from pydantic import BaseModel, Field
  from typing import Literal
  from langchain.agents import create_agent
  from langchain.agents.structured_output import ToolStrategy


  class ProductReview(BaseModel):
      """Analysis of a product review."""
      rating: int | None = Field(description="The rating of the product", ge=1, le=5)
      sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
      key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

  agent = create_agent(
      model="gpt-5.4",
      tools=tools,
      response_format=ToolStrategy(ProductReview)
  )

  result = agent.invoke({
      "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
  })
  result["structured_response"]
  # ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
  ```

  ```python Dataclass theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from dataclasses import dataclass
  from typing import Literal
  from langchain.agents import create_agent
  from langchain.agents.structured_output import ToolStrategy


  @dataclass
  class ProductReview:
      """Analysis of a product review."""
      rating: int | None  # The rating of the product (1-5)
      sentiment: Literal["positive", "negative"]  # The sentiment of the review
      key_points: list[str]  # The key points of the review

  agent = create_agent(
      model="gpt-5.4",
      tools=tools,
      response_format=ToolStrategy(ProductReview)
  )

  result = agent.invoke({
      "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
  })
  result["structured_response"]
  # {'rating': 5, 'sentiment': 'positive', 'key_points': ['fast shipping', 'expensive']}
  ```

  ```python TypedDict theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from typing import Literal
  from typing_extensions import TypedDict
  from langchain.agents import create_agent
  from langchain.agents.structured_output import ToolStrategy


  class ProductReview(TypedDict):
      """Analysis of a product review."""
      rating: int | None  # The rating of the product (1-5)
      sentiment: Literal["positive", "negative"]  # The sentiment of the review
      key_points: list[str]  # The key points of the review

  agent = create_agent(
      model="gpt-5.4",
      tools=tools,
      response_format=ToolStrategy(ProductReview)
  )

  result = agent.invoke({
      "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
  })
  result["structured_response"]
  # {'rating': 5, 'sentiment': 'positive', 'key_points': ['fast shipping', 'expensive']}
  ```

  ```python JSON Schema theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from langchain.agents import create_agent
  from langchain.agents.structured_output import ToolStrategy


  product_review_schema = {
      "type": "object",
      "description": "Analysis of a product review.",
      "properties": {
          "rating": {
              "type": ["integer", "null"],
              "description": "The rating of the product (1-5)",
              "minimum": 1,
              "maximum": 5
          },
          "sentiment": {
              "type": "string",
              "enum": ["positive", "negative"],
              "description": "The sentiment of the review"
          },
          "key_points": {
              "type": "array",
              "items": {"type": "string"},
              "description": "The key points of the review"
          }
      },
      "required": ["sentiment", "key_points"]
  }

  agent = create_agent(
      model="gpt-5.4",
      tools=tools,
      response_format=ToolStrategy(product_review_schema)
  )

  result = agent.invoke({
      "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
  })
  result["structured_response"]
  # {'rating': 5, 'sentiment': 'positive', 'key_points': ['fast shipping', 'expensive']}
  ```

  ```python Union Types theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  from pydantic import BaseModel, Field
  from typing import Literal, Union
  from langchain.agents import create_agent
  from langchain.agents.structured_output import ToolStrategy


  class ProductReview(BaseModel):
      """Analysis of a product review."""
      rating: int | None = Field(description="The rating of the product", ge=1, le=5)
      sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
      key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

  class CustomerComplaint(BaseModel):
      """A customer complaint about a product or service."""
      issue_type: Literal["product", "service", "shipping", "billing"] = Field(description="The type of issue")
      severity: Literal["low", "medium", "high"] = Field(description="The severity of the complaint")
      description: str = Field(description="Brief description of the complaint")

  agent = create_agent(
      model="gpt-5.4",
      tools=tools,
      response_format=ToolStrategy(Union[ProductReview, CustomerComplaint])
  )

  result = agent.invoke({
      "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
  })
  result["structured_response"]
  # ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
  ```
</CodeGroup>

### 自定义工具消息内容

`tool_message_content` 参数允许你自定义生成结构化输出时出现在对话历史中的消息：

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class MeetingAction(BaseModel):
    """Action items extracted from a meeting transcript."""
    task: str = Field(description="The specific task to be completed")
    assignee: str = Field(description="Person responsible for the task")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")

agent = create_agent(
    model="gpt-5.4",
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="Action item captured and added to meeting notes!"
    )
)

agent.invoke({
    "messages": [{"role": "user", "content": "From our meeting: Sarah needs to update the project timeline as soon as possible"}]
})
```

```
================================ Human Message =================================

From our meeting: Sarah needs to update the project timeline as soon as possible
================================== Ai Message ==================================
Tool Calls:
  MeetingAction (call_1)
 Call ID: call_1
  Args:
    task: Update the project timeline
    assignee: Sarah
    priority: high
================================= Tool Message =================================
Name: MeetingAction

Action item captured and added to meeting notes!
```

如果没有设置 `tool_message_content`，最终的 [`ToolMessage`](https://reference.langchain.com/python/langchain-core/messages/tool/ToolMessage) 将会是：

```
================================= Tool Message =================================
Name: MeetingAction

Returning structured response: {'task': 'update the project timeline', 'assignee': 'Sarah', 'priority': 'high'}
```

### 错误处理

模型在通过工具调用生成结构化输出时可能会出错。LangChain 提供了智能的重试机制来自动处理这些错误。

#### 多个结构化输出错误

当模型错误地调用了多个结构化输出工具时，智能体会在 [`ToolMessage`](https://reference.langchain.com/python/langchain-core/messages/tool/ToolMessage) 中提供错误反馈，并提示模型重试：

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")

agent = create_agent(
    model="gpt-5.4",
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails])  # Default: handle_errors=True
)

agent.invoke({
    "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})
```

```
================================ Human Message =================================

Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th
None
================================== Ai Message ==================================
Tool Calls:
  ContactInfo (call_1)
 Call ID: call_1
  Args:
    name: John Doe
    email: john@email.com
  EventDetails (call_2)
 Call ID: call_2
  Args:
    event_name: Tech Conference
    date: March 15th
================================= Tool Message =================================
Name: ContactInfo

Error: Model incorrectly returned multiple structured responses (ContactInfo, EventDetails) when only one is expected.
 Please fix your mistakes.
================================= Tool Message =================================
Name: EventDetails

Error: Model incorrectly returned multiple structured responses (ContactInfo, EventDetails) when only one is expected.
 Please fix your mistakes.
================================== Ai Message ==================================
Tool Calls:
  ContactInfo (call_3)
 Call ID: call_3
  Args:
    name: John Doe
    email: john@email.com
================================= Tool Message =================================
Name: ContactInfo

Returning structured response: {'name': 'John Doe', 'email': 'john@email.com'}
```

#### 模式验证错误

当结构化输出与预期模式不匹配时，智能体会提供具体的错误反馈：

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductRating(BaseModel):
    rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")

agent = create_agent(
    model="gpt-5.4",
    tools=[],
    response_format=ToolStrategy(ProductRating),  # Default: handle_errors=True
    system_prompt="You are a helpful assistant that parses product reviews. Do not make any field or value up."
)

agent.invoke({
    "messages": [{"role": "user", "content": "Parse this: Amazing product, 10/10!"}]
})
```

```
================================ Human Message =================================

Parse this: Amazing product, 10/10!
================================== Ai Message ==================================
Tool Calls:
  ProductRating (call_1)
 Call ID: call_1
  Args:
    rating: 10
    comment: Amazing product
================================= Tool Message =================================
Name: ProductRating

Error: Failed to parse structured output for tool 'ProductRating': 1 validation error for ProductRating.rating
  Input should be less than or equal to 5 [type=less_than_equal, input_value=10, input_type=int].
 Please fix your mistakes.
================================== Ai Message ==================================
Tool Calls:
  ProductRating (call_2)
 Call ID: call_2
  Args:
    rating: 5
    comment: Amazing product
================================= Tool Message =================================
Name: ProductRating

Returning structured response: {'rating': 5, 'comment': 'Amazing product'}
```

#### 错误处理策略

你可以使用 `handle_errors` 参数来自定义错误处理方式：

**自定义错误消息：**

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
ToolStrategy(
    schema=ProductRating,
    handle_errors="Please provide a valid rating between 1-5 and include a comment."
)
```

如果 `handle_errors` 是字符串，智能体将始终使用固定的工具消息提示模型重试：

```
================================= Tool Message =================================
Name: ProductRating

Please provide a valid rating between 1-5 and include a comment.
```

**仅处理特定异常：**

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
ToolStrategy(
    schema=ProductRating,
    handle_errors=ValueError  # Only retry on ValueError, raise others
)
```

如果 `handle_errors` 是异常类型，智能体仅当引发的异常是指定类型时才会重试（使用默认错误消息）。在所有其他情况下，异常会被抛出。

**处理多种异常类型：**

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
ToolStrategy(
    schema=ProductRating,
    handle_errors=(ValueError, TypeError)  # Retry on ValueError and TypeError
)
```

如果 `handle_errors` 是异常元组，智能体仅当引发的异常是指定类型之一时才会重试（使用默认错误消息）。在所有其他情况下，异常会被抛出。

**自定义错误处理函数：**

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}

from langchain.agents.structured_output import StructuredOutputValidationError
from langchain.agents.structured_output import MultipleStructuredOutputsError

def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"


agent = create_agent(
    model="gpt-5.4",
    tools=[],
    response_format=ToolStrategy(
                        schema=Union[ContactInfo, EventDetails],
                        handle_errors=custom_error_handler
                    )  # Default: handle_errors=True
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})

for msg in result['messages']:
    # If message is actually a ToolMessage object (not a dict), check its class name
    if type(msg).__name__ == "ToolMessage":
        print(msg.content)
    # If message is a dictionary or you want a fallback
    elif isinstance(msg, dict) and msg.get('tool_call_id'):
        print(msg['content'])

```

在 `StructuredOutputValidationError` 时：

```
================================= Tool Message =================================
Name: ToolStrategy

There was an issue with the format. Try again.
```

在 `MultipleStructuredOutputsError` 时：

```
================================= Tool Message =================================
Name: ToolStrategy

Multiple structured outputs were returned. Pick the most relevant one.
```

在其他错误时：

```
================================= Tool Message =================================
Name: ToolStrategy

Error: <error message>
```

**不处理错误：**

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
response_format = ToolStrategy(
    schema=ProductRating,
    handle_errors=False  # All errors raised
)
```

***

<div className="source-links">
  <Callout icon="edit">
    [在 GitHub 上编辑此页面](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/structured-output.mdx) 或 [提交问题](https://github.com/langchain-ai/docs/issues/new/choose)。
  </Callout>

  <Callout icon="terminal-2">
    [将这些文档](/use-these-docs) 通过 MCP 连接到 Claude、VSCode 等工具以获取实时解答。
  </Callout>
</div>
