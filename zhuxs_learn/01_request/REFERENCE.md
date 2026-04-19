# 01_request Reference

## 本地与外部资料

1. vLLM Quickstart - OpenAI-compatible server
   https://docs.vllm.ai/en/stable/getting_started/quickstart/#openai-chat-completions-api-with-vllm
2. SiliconFlow Chat Completions
   https://docs.siliconflow.cn/cn/api-reference/chat-completions/chat-completions
3. SiliconFlow Messages / Anthropic-compatible
   https://docs.siliconflow.cn/cn/api-reference/chat-completions/messages
4. MiniMax OpenAI-compatible API
   https://platform.minimaxi.com/docs/api-reference/text-openai-api
5. MiniMax Anthropic-compatible API
   https://platform.minimaxi.com/docs/api-reference/text-anthropic-api
6. DeepSeek Thinking Mode
   https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
7. DeepSeek Anthropic API
   https://api-docs.deepseek.com/zh-cn/guides/anthropic_api
8. LangChain ChatOpenAI integration docs
   `/workspace/langgraph-dev/langchain_docs/src/oss/python/integrations/chat/openai.mdx`
9. LangChain ChatAnthropic integration docs
   `/workspace/langgraph-dev/langchain_docs/src/oss/python/integrations/chat/anthropic.mdx`
10. LangChain vLLM integration docs
    `/workspace/langgraph-dev/langchain_docs/src/oss/python/integrations/chat/vllm.mdx`

## 当前已确认事实

1. `ChatOpenAI` 文档明确说明：第三方 provider 的非标准字段如 `reasoning_content`、`reasoning_details` 不会被标准化保留。
2. `ChatAnthropic` 支持 `thinking` 参数，并能通过 `content_blocks` 与 `tool_calls` 暴露 reasoning / tool use。
3. `DeepSeek` 提供 Anthropic 兼容接口，`ANTHROPIC_BASE_URL` 可指向 `https://api.deepseek.com/anthropic`。
4. `MiniMax` 文档强调：多轮 function call 对话中，必须完整回传上一轮 assistant content，才能保持 thinking 连续。
5. `SiliconFlow` 的 `/messages` 文档声明支持 `disable_parallel_tool_use`，说明 Anthropic-like parallel tool use 需要单独验证。
6. 仓库 `.env` 已存在以下相关变量名：
   `SILICONFLOW_*`、`MINIMAX_*`、`DEEPSEEK_*`、`VLLM_QWEN_*`、`VLLM_MINIMAX_*`

## 实现含义

1. OpenAI-like LangChain 版本必须显式记录“LangChain 层丢失了哪些 provider 扩展字段”。
2. Anthropic-like 脚本要特别验证 `tool_use` / `tool_result` 往返，以及多轮时是否完整回传 assistant content。
3. `results/*.json` 应该优先保存真实 request / response，避免额外加工影响学习观察。

## 运行后新增观察

1. `SILICONFLOW_ANTHROPIC` 在使用 `anthropic` SDK 时，base URL 必须是 `https://api.siliconflow.cn`，因为 SDK 会自行追加 `/v1/messages`。
2. `SiliconFlow + DeepSeek-V3.2` 的 OpenAI / Anthropic 兼容接口虽然支持 tools 字段，但模型会明确说明自己不能并行调用两个工具，因此不适合做 parallel tools 的稳定示例。
3. `MiniMax` 的 OpenAI / Anthropic 兼容接口在本次实测中都能一次返回两个工具调用，适合作为 parallel tools 的默认演示 provider。
4. `DeepSeek` reasoning 模型的原始 OpenAI-like 响应中包含 `reasoning_content`，SDK 版本可以直接从抓包看到该字段。
5. `ChatOpenAI` 调用 `DEEPSEEK_OPENAI` 时，原始抓包里仍能看到 `reasoning_content`，但 LangChain 标准化后的 `AIMessage.additional_kwargs` 没有保留它。
6. `DEEPSEEK_ANTHROPIC` 的 interleaved thinking 示例里，第一轮响应会同时包含 `thinking` 与 `tool_use`，第二轮 request 也会把 `thinking` block 一并回传。
