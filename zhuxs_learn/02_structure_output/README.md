# 02_structure_output — 结构化输出实验概述

> **实验目标**：系统性地搞清楚 LangChain + vLLM 结构化输出的底层行为。  
> **核心问题**：`json_schema` 的 `description` 到底对模型可不可见？各种写法最终发给服务端的请求体长什么样？

---

## 目录结构

```
02_structure_output/
├── config.py                                        # 共享 LLM 配置（在 zhuxs_learn/ 下）
├── vllm_structured_outputs.py                       # vLLM 官方示例移植版
├── structure_output.ipynb                           # 主实验 Notebook
├── test_vllm_json_schema_description_visibility.py  # description 可见性 pytest 版本
├── test_langchain_vllm_capture.py                   # LangChain 抓包测试套件
├── langchain_structure_output.md                    # LangChain 官方文档（英文）
├── langchain_structure_output_zh.md                 # LangChain 官方文档（中文）
├── langchain_vllm_structure_output.md               # 整合笔记：分层说明与选型建议
└── httpx_result/                                    # 测试抓包结果落盘目录
```

---

## 文件详解

### 1. `config.py`（zhuxs_learn/ 下）

**共享配置文件**，被所有测试和 Notebook 引用。

- 定义 `OPENAI_BASE_URL` 和 `MODEL`（指向本地 vLLM 服务）
- 构造带 httpx event hook 的 `llm_instruct` 实例，每次请求/响应均打印到终端
- `test_langchain_vllm_capture.py` 通过 `importlib` 按绝对路径动态加载本文件

---

### 2. `vllm_structured_outputs.py`

**vLLM 官方结构化输出示例的移植版**，从 vLLM 文档直接迁移而来，作为"底层能力参考"。

#### 演示的五种约束解码方式

| 约束类型 | 传参方式 | 效果 |
|----------|----------|------|
| `choice` | `extra_body.structured_outputs.choice` | 输出只能是指定枚举值之一 |
| `regex` | `extra_body.structured_outputs.regex` | 输出必须匹配正则表达式 |
| `json` | `response_format.json_schema` | 输出必须符合 Pydantic/JSON Schema |
| `grammar` | `extra_body.structured_outputs.grammar` | 输出必须符合 EBNF 文法 |
| `structural_tag` | `response_format.structural_tag` | 函数调用标签格式约束 |

#### 如何运行

```bash
# 运行全部约束（非流式）
uv run vllm_structured_outputs.py

# 只运行 regex 和 json_schema
uv run vllm_structured_outputs.py --constraint regex json

# 启用流式输出
uv run vllm_structured_outputs.py --stream
```

---

### 3. `structure_output.ipynb`

**主实验 Notebook**，包含两部分内容。

#### 第一部分：早期探索（前几个 cell）

- 用 Pydantic `BaseModel` 定义 `PersonRecord`，class docstring 和字段 `description` 都包含对模型的指令
- 直接用 `openai.OpenAI` 调用 vLLM，观察模型是否遵循 description 中的规则
- 也展示了 LangChain `ChatOpenAI` 的带日志钩子写法

#### 第二部分：鲁棒实验（核心部分）

证明 **vLLM `json_schema` 模式下 description 对模型不可见**。

##### 实验设计

| 字段 | 测试维度 | 占位规则位置 |
|------|----------|-------------|
| `f_name` | 实验1：类级 docstring 可见性 | class docstring |
| `f_email` | 基础提取（控制变量） | 消息中直接包含 |
| `f_city` | 实验2：字段 description 占位规则可见性 | 字段 `description` |
| `f_answer` | 实验3：字段 description 操作指令可见性 | 字段 `description`（乘法指令） |

**关键设计**：
- 用户消息中故意不含 `name` 和 `city`，只有 email 和两个裸数字（157, 13）
- 占位符用 `unknown_name_7749` / `missing_city_7749` 这种"常见词+唯一数字"的格式，模型不可能靠语义猜出
- `f_answer` 期望输出 `2041`（157×13），但消息中没有"乘法"提示，乘法指令只在字段 description 中

##### 实验组对比

| 组别 | 实验条件 |
|------|----------|
| **A 组** | 只传 `json_schema`，无 system prompt |
| **B 组（对照）** | `json_schema` + 将完整 `model_json_schema()` JSON 注入 system prompt |

##### 实验结果

```
A 组（json_schema only）：类级 docstring 不可见 ✗ | 字段 description 不可见 ✗
B 组（+system prompt）  ：类级 docstring 可见   ✓ | 字段 description 可见   ✓
```

- A 组：`f_city` 输出 `"Not specified"`（幻觉），`f_answer` 输出 `"170"`（默认相加）
- B 组：`f_city` 输出 `"missing_city_7749"`，`f_answer` 输出 `"2041"`，全部正确

##### Workaround 封装

Notebook 末尾提供了 `structured_chat_with_schema_prompt` 函数，自动将 `model_json_schema()` 注入 system prompt：

```python
result = structured_chat_with_schema_prompt(
    model_cls=MyModel,
    user_message="...",
    openai_client=client,
    model_name="qwen36-27b",
)
# 返回 Pydantic 实例，description 指令对模型完全可见
```

---

### 4. `test_vllm_json_schema_description_visibility.py`

**Notebook 实验的 pytest 可执行版本**，逻辑与 Notebook 相同，可集成到 CI。

#### 如何运行

```bash
# 运行全部测试
pytest -q zhuxs_learn/02_structure_output/test_vllm_json_schema_description_visibility.py

# 查看详细输出（不捕获 print）
pytest -s zhuxs_learn/02_structure_output/test_vllm_json_schema_description_visibility.py
```

---

### 5. `test_langchain_vllm_capture.py`

**LangChain 结构化输出行为的抓包测试套件**，通过拦截真实 HTTP 请求，验证"LangChain 各种写法最终把什么参数发给 vLLM"。

#### 核心机制

```
LangChain 调用
    ↓
httpx.Client（注入 event_hooks）
    ↓ _on_request / _on_response
HttpRecorder（自动落盘到 httpx_result/<case_name>/）
    ↓
vLLM 服务（http://192.168.10.99:33627/v1）
```

每个测试用例在 `httpx_result/<case_name>/` 下生成：
- `01_request.json` / `01_response.json`：完整的 HTTP 请求/响应
- `summary.json`：结果摘要（含请求体快照和解析结果）

#### 测试用例一览

| 编号 | 函数名 | 实验条件 | 验证什么 |
|------|--------|----------|----------|
| ① | `test_with_structured_output_pydantic_json_schema` | Pydantic + `json_schema` | 请求体有 `response_format.json_schema`，返回 Pydantic 实例 |
| ② | `test_with_structured_output_typeddict_json_schema` | TypedDict + `json_schema` | 同样生成 `json_schema` 请求，但返回 `dict`（无运行时类型） |
| ③ | `test_with_structured_output_json_schema_dict_json_schema` | 原始 dict schema + `json_schema` | `schema.title` 作为 `json_schema.name` 透传 |
| ④ | `test_with_structured_output_pydantic_json_mode_validation` | Pydantic StrictStr + `json_mode`，故意触发类型错误 | `json_mode` 只发 `{"type":"json_object"}`；⚠️ **vLLM 不支持此格式**（仅 DeepSeek 等云端厂商实现）；即使服务端支持也只保证合法 JSON，不保证符合 schema；Pydantic 验证失败抛 `OutputParserException` |
| ⑤ | `test_with_structured_output_typeddict_json_mode_no_validation` | TypedDict + `json_mode`，类型错误 | 双重风险：⚠️ vLLM 不支持 `json_object` + TypedDict 无运行时验证，脏数据完全静默透传，是最危险的组合 |
| ⑥ | `test_create_agent_auto_strategy_falls_back_to_tool_strategy` | `create_agent` 自动策略 | 对自定义 `base_url` 降级为 `ToolStrategy`（有 `tools` 字段） |
| ⑦ | `test_create_agent_provider_strategy_forces_response_format` | `create_agent` + `ProviderStrategy` | 强制走 `json_schema`，`tools=[]` |
| ⑧ | `test_vllm_regex_passthrough` | `extra_body.structured_outputs.regex` 透传 | LangChain 与 OpenAI SDK 请求体完全相同 |
| ⑨ | `test_vllm_json_schema_passthrough` | `llm.bind(response_format=...)` 手动透传 | 不经过 `with_structured_output` 也能精确控制请求体 |
| ⑩ | `test_vllm_grammar_passthrough` | `extra_body.structured_outputs.grammar` 透传 | EBNF 文法约束正确透传 |
| ⑪ | `test_vllm_structural_tag_passthrough` | `response_format.structural_tag` 透传 | structural_tag 两侧请求体完全相同 |

> **注**：⑧~⑪ 均采用"双轨对比"模式：同一参数同时用 LangChain 和原生 OpenAI SDK 各发一次，生成 `comparison_summary.json` 对比两侧请求体是否等价。

#### 如何运行

```bash
# 运行全部（在项目根目录）
pytest -q zhuxs_learn/02_structure_output/test_langchain_vllm_capture.py

# 只运行某一个
pytest -q zhuxs_learn/02_structure_output/test_langchain_vllm_capture.py::test_vllm_regex_passthrough

# 用关键字过滤
pytest -q zhuxs_learn/02_structure_output/test_langchain_vllm_capture.py -k "passthrough"
```

---

### 6. 参考文档（`.md` 文件）

| 文件 | 内容 |
|------|------|
| `langchain_structure_output.md` | LangChain `create_agent` 结构化输出官方文档（英文原版） |
| `langchain_structure_output_zh.md` | 同上，中文翻译版 |
| `langchain_vllm_structure_output.md` | **整合笔记**：4 层抽象分层说明、`json_schema`/`json_mode`/`function_calling` 对比、vLLM 专有能力透传方案、选型建议 |

---

## 核心结论

### 1. vLLM `json_schema` 的 description 对模型不可见

```
json_schema 的 description / docstring
    → 只用于 JSON 格式约束（guided decoding）
    → 不会出现在模型的 context window 中
    → 模型看不到，因此不遵循其中的指令
```

**解决方案**：将 `model_json_schema()` 序列化后注入 system prompt。

### 2. LangChain 各种写法的请求体差异

| 写法 | 请求体特征 |
|------|-----------|
| `with_structured_output(schema, method="json_schema")` | `response_format.type = "json_schema"` |
| `with_structured_output(schema, method="json_mode")` | `response_format.type = "json_object"`；⚠️ **vLLM 不支持**（仅 DeepSeek 等云端厂商实现）；即使在支持的服务端也只保证合法 JSON 而非符合 schema 的 JSON——全面劣于 `json_schema` |
| `with_structured_output(schema, method="function_calling")` | `tools = [...]`，无 `response_format` |
| `create_agent(response_format=Schema)` 自动策略 + 自定义 base_url | 降级为 `tools = [...]` |
| `create_agent(response_format=ProviderStrategy(Schema))` | `response_format.type = "json_schema"` |

### 3. `json_mode` 的两大局限性

> **结论：在 vLLM 上应完全避免使用 `json_mode`，任何情况下 `json_schema` 都是更好的选择。**

**局限 1：vLLM 不支持 `json_object` 格式**

`json_mode` 向服务端发送 `response_format={"type":"json_object"}`。该参数格式只被 DeepSeek、OpenAI 等云端厂商实现，本地 vLLM 服务不识别此参数，模型输出不受任何约束。

**局限 2：即使服务端支持，也只保证"合法 JSON"**

`json_mode` 的语义是"输出必须是可解析的 JSON"，但对 JSON 的结构（字段名、类型、是否必填）完全不做约束。举例：

```python
# json_mode 下，以下输出均被视为"合法"：
{}                          # 空对象，缺少所有字段
{"name": 123}              # 字段类型错误
{"random_key": "value"}    # 完全不相关的字段
```

`json_schema` 则通过 guided decoding 在 **token 生成阶段**强制约束输出结构，不依赖模型的"遵从指令"能力。

### 4. vLLM 专有能力（regex/grammar/structural_tag）的透传

LangChain 不抽象这些能力，需要通过 `extra_body` 直接透传：

```python
llm = ChatOpenAI(
    extra_body={"structured_outputs": {"regex": r"[a-z]+@\w+\.com\n"}}
)
# 或
llm.bind(response_format={"type": "structural_tag", ...})
```
