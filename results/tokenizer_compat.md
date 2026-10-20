# Tokenizer compatibility: 1.5B vs. 7B rival

Computed by `scripts/check_tokenizer_compat.py`. Tokenizer files only; no 7B weights.

| | |
|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` @ `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Rival | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` @ `916b56a44061fd5cd7d6a8fb632557ed4f724f60` |
| Vocabulary size (len) | 151665 vs. 151665 |
| Pilot traces | 30 (prompt + output ids) |
| Distinct token ids checked | 2182 |
| Ids decoding differently | **0** |
| Traces whose full decode differs | **0** |
| Full token-to-id maps identical (beyond the pilot) | yes |

