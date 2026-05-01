https://github.com/lobehub/lobehub/discussions/11480
    thinking content 分出来是标准做法。无论是 OpenAI 的 Response API ，还是 DeepSeek 亦或是 Anthropic 的 Message API ， thinking content 都是单独分离并且回传。拼接工作都是在各自的 provider 中实现的，所以默认会达到服务商提供的最佳性能。放在 content 字段里反而是非标且降低性能的做法
    Response API 默认会回传 thinking 字段 并正常消费，如果 vllm 支持 Response API 接口规范，直接用 Response API 即可。按标准 OpenAI Chat Completion 接口的规范默认就是要丢弃 thinking content 的。 像 DeepSeek / minimax 的 provider 我们是专门做了处理把这个 thinking_content 发回去的，这种魔改规范的东西只能定向 provider 支持。
MINIMAX相关参考
    https://platform.minimaxi.com/docs/api-reference/text-prompt-caching
    https://platform.minimaxi.com/docs/guides/text-m2-function-call
zhipu相关参考
    https://docs.bigmodel.cn/cn/guide/capabilities/thinking
    https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode
    https://docs.bigmodel.cn/cn/guide/capabilities/cache
deepseek相关参考
    https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
    https://api-docs.deepseek.com/zh-cn/guides/multi_round_chat
    https://api-docs.deepseek.com/zh-cn/guides/tool_calls
    https://api-docs.deepseek.com/zh-cn/guides/kv_cache
kimi相关参考（由于kimi目前还是使用chat completion而不是response api因此这里不讨论）
