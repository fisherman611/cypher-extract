# Token length analysis

`scripts/analyze_token_lengths.py` đo độ dài token thực tế mà `cutoff_len` và
context lúc inference phải phủ. Prompt được render bằng đúng
`PromptTemplates` và renderer (`render_qwen3_nothink` / `render_llama3`) mà
training và pipeline dùng, nên số đo giữ prompt parity by construction.

Chạy lại (số dưới đây đo trên `llamafactory_distractor_v3`):

```bash
python scripts/analyze_token_lengths.py --model-families all --local-files-only \
  --prepared-dir data/llamafactory_distractor_v3
```

Report ghi ra `data/token_length_analysis.json` (không được track vì `data/`
nằm trong `.gitignore`) — file này là bản chép lại kết quả để giữ lâu dài.

## Training (`data/llamafactory_distractor_v3/cypher_prepared_*.jsonl`)

`cutoff_len` truncate đúng `prompt + response`, nên nó phải `>= max_total`.
Prompt generator gồm `CANDIDATE SCHEMA` và `OTHER SCHEMA`, tức luôn chứa toàn
bộ schema, nên độ dài gần như không phụ thuộc mức nhiễu của generator.

| family | split | rows | max_prompt | max_total | max_response |
| --- | --- | --- | --- | --- | --- |
| qwen3 | train | 16580 | 860 | 963 | 113 |
| qwen3 | eval | 3414 | 842 | 934 | 109 |
| qwen2.5_coder | train | 16580 | 860 | 963 | 113 |
| qwen2.5_coder | eval | 3414 | 842 | 934 | 109 |
| llama3 | train | 16580 | 856 | 952 | 110 |
| llama3 | eval | 3414 | 844 | 933 | 108 |

`cutoff_len` hiện tại là `1024` cho mọi family: dư 61 token (qwen) và 72 token
(llama3). **Không có row nào bị truncate.**

## Inference (`*_inference_test.jsonl`)

Generator được prompt bằng *predicted* sub-schema, nên worst case là selector
đánh dấu toàn bộ schema unit của example là related — không phải gold
sub-schema. Vì prompt generator luôn chứa toàn bộ schema (phần không chọn nằm
trong `OTHER SCHEMA`), worst case và gold sub-schema gần như bằng nhau.

Số dưới đây của qwen3 / qwen2.5_coder (hai family chung tokenizer); llama3
lệch trong khoảng ±4 token.

| dataset | selector prompt | generator worst case | generator gold sub-schema |
| --- | --- | --- | --- |
| cypherbench | 543 | 999 | 1003 |
| mind_the_query | 604 | 1983 | 1987 |
| neo4j_text2cypher | 696 | 2253 | 2257 |

Cộng `max_new_tokens` (selector 16, generator 256):

| family | required `cutoff_len` | required inference context |
| --- | --- | --- |
| qwen3 | 963 | 2509 |
| qwen2.5_coder | 963 | 2509 |
| llama3 | 952 | 2511 |

## Hệ quả

- `cutoff_len: 1024` an toàn cho dữ liệu train/eval hiện tại; không cần nâng.
  Nếu đổi prompt hoặc thêm schema lớn hơn thì phải đo lại, vì biên chỉ còn
  khoảng 60 token.
- DistiLLM adaptive: rollout sinh tối đa `cutoff_len - rollout_context_length`
  token (`1024 - 797 = 227` cho qwen, `1024 - 810 = 214` cho llama3), sau đó
  response bị cắt còn `cutoff_len - len(prompt)`, tối thiểu `1024 - 860 = 164`
  token. Cả hai đều lớn hơn response dài nhất (113), nên rollout hợp lệ không bị
  cắt. `rollout_context_length` chỉ chia ngân sách token, không cắt prompt, nên
  prompt dài hơn 797 token vẫn được giữ nguyên.
- Generative eval trong lúc train dùng `max_new_tokens` riêng (selector 16,
  generator 256), không bị giới hạn bởi `cutoff_len`.
- Context inference cần khoảng `2.5k` token (neo4j_text2cypher), khoảng `2.5x`
  `cutoff_len`. Đường HF `generate` hiện tại không bị cắt, nhưng bất kỳ chỗ nào
  set `max_model_len` (ví dụ khi chuyển sang vLLM) phải dùng ít nhất `2511`,
  không phải `cutoff_len`.
- Response dài nhất trong training data là `113` token, nên
  `generator_max_new_tokens = 256` còn dư hơn `2x` headroom.
- Prompt generator lúc train dài nhất `860` token (schema cypherbench), còn lúc
  infer trên mind_the_query/neo4j có thể lên `2253`, do schema lớn hơn. Đây là
  distribution shift thật ở generator, không phải rủi ro truncation.
