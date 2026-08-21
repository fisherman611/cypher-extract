# cypher-extract

Pipeline tạo dữ liệu huấn luyện cho schema grounding trong Text-to-Cypher.
Nó chuẩn hóa schema của CypherBench, Mind-the-Query và Neo4j Text2Cypher về
cùng một graph representation, suy ngược gold sub-schema từ gold Cypher, rồi
xuất hai loại dữ liệu:

```text
Schema selection:  question + schema unit -> label 0/1
Cypher generation: question + gold sub-schema -> gold Cypher
```

## Requirements

- Python **3.10+**. Code dùng type union `X | Y`, không hỗ trợ Python 3.9 trở xuống.
- Không có package bên thứ ba: pipeline và test chỉ dùng Python standard library.
- Cần đặt các benchmark muốn xử lý dưới `benchmarks/` theo cấu trúc:

  ```text
  benchmarks/
  ├── Cypherbench/
  ├── Mind_the_query/
  └── Neo4j_Text2Cypher/
  ```

  Mặc định pipeline chạy cả ba benchmark. Có thể giới hạn bằng `--sources`.

Kiểm tra phiên bản Python:

```powershell
python --version
```

Tùy chọn tạo virtual environment (không cần cài dependency):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

`requirements.txt` được giữ để lệnh cài đặt thống nhất giữa các môi trường;
hiện tại file không khai báo package nào:

```powershell
python -m pip install -r requirements.txt
```

## Chạy tests

```powershell
python -m unittest discover -s tests -v
```

## Tạo dữ liệu

Xem toàn bộ tham số:

```bash
python scripts/build_schema_grounding_data.py --help
```

### 1. Smoke test (Kiểm tra nhanh)

Chạy thử 25 mẫu của **mỗi** benchmark (`cypherbench`, `mind_the_query`, `neo4j_text2cypher`):

```bash
python scripts/build_schema_grounding_data.py \
  --output-dir /tmp/schema-grounding-smoke \
  --max-examples 25 \
  --negative-ratio 4 \
  --overwrite
```

---

### 2. CypherBench (Train, Dev, Test)

CypherBench dùng các file prompt/response có schema nhúng: `train.jsonl`,
`dev.jsonl`, và `test.jsonl`. Tạo đủ ba split trong một output directory:

```powershell
python scripts\build_schema_grounding_data.py `
  --benchmarks-root benchmarks `
  --sources cypherbench `
  --splits train,dev,test `
  --output-dir data\cypherbench_schema_grounding_full `
  --overwrite
```

### 3. Neo4j Text2Cypher (Train, Test)

Neo4j Text2Cypher dùng `train.json` và `test.json`, vì các file `.jsonl` chỉ là
prompt/response export và không chứa schema cần cho sub-schema extraction.

```powershell
python scripts\build_schema_grounding_data.py `
  --benchmarks-root benchmarks `
  --sources neo4j_text2cypher `
  --splits train,test `
  --output-dir data\neo4j_text2cypher_schema_grounding_full `
  --overwrite
```

### 4. Mind-the-Query (Train, Test)

Mind-the-Query dùng `train_val.json` và `test.json` vì các JSONL tương ứng
không mang schema metadata.

```powershell
python scripts\build_schema_grounding_data.py `
  --benchmarks-root benchmarks `
  --sources mind_the_query `
  --splits train,test `
  --output-dir data\mind_the_query_schema_grounding_full `
  --overwrite
```

Chỉ thêm `--negative-ratio N` khi muốn giới hạn số negative schema unit cho mỗi
positive unit trong `selection_<split>.jsonl`; nó không thay đổi generation data.

### Lọc selector stage-1

Tạo tập selector nhỏ, cân bằng theo graph và nhãn, đồng thời cover mọi schema
unit × label quan sát được trong `selection_train.jsonl`:

```powershell
python scripts\filter_selector_stage1.py `
  --input-dir data\cypherbench_schema_grounding_full `
  --output-dir data\cypherbench_schema_grounding_full_final `
  --target-rows 3200 `
  --seed 42 `
  --overwrite
```

Script giữ nguyên generation data và selection dev/test; chỉ thay
`selection_train.jsonl` và cập nhật manifest với policy lọc.

### Chuẩn bị prompt multitask

Chuẩn bị `train.jsonl` và `eval.jsonl` sao cho batch có cả generator và
selector khi số row cho phép, không lặp lại bất kỳ row nguồn nào. Nếu task
selector ít hơn số batch (ví dụ `batch-size=2`), selector sẽ được rải đều vào
các mixed batch và batch còn lại chỉ chứa generator. Test generator và selector
được ghi thành hai file riêng. Không bật shuffle lại ở data loader vì thứ tự
trong train/eval đã được interleave theo batch.

Với CypherBench final hiện tại (`6,827` generator, `3,200` selector) và
`batch-size=2`, train có `3,200` mixed batch (`1 generator + 1 selector`) và
`1,814` batch chỉ generator; mọi mẫu nguồn chỉ xuất hiện một lần.

```powershell
python scripts\prepare_multitask_prompts.py `
  --input-dir data\cypherbench_schema_grounding_full_final `
  --output-dir data\prepared `
  --batch-size 2 `
  --overwrite
```

Chuyển bốn file prompt/response vừa tạo sang OpenAI chat JSONL và đăng ký tên
dataset local cho LlamaFactory:

```powershell
python scripts\prepare_llamafactory_data.py `
  --input-dir data\prepared `
  --output-dir data\llamafactory `
  --overwrite
```

Mỗi row đầu ra chỉ giữ `messages` theo thứ tự `system -> user -> assistant`.
Các config train dùng tên `cypher_prepared_train` và `cypher_prepared_eval`
trong `data/llamafactory/dataset_info.json`.

> **Lưu ý:**
> - Pipeline mặc định từ chối ghi đè lên thư mục đã có dữ liệu. Hãy thêm `--overwrite` khi chủ động muốn tạo lại từ đầu.
> - Tham số `--negative-ratio` chỉ tác động đến tập `selection_<split>.jsonl` (Task A) để cân bằng tỷ lệ nhãn `0` và `1`; tập `generation_<split>.jsonl` (Task B) luôn giữ nguyên gold sub-schema.

## Output format

`schemas.jsonl` lưu mỗi full schema duy nhất. Phần graph chuẩn hóa chỉ gồm
`nodes` và `relationships`:

```json
{
  "nodes": [
    {
      "label": "Person",
      "properties": {"name": "STRING"}
    }
  ],
  "relationships": [
    {
      "source": "Person",
      "type": "ACTED_IN",
      "target": "Movie",
      "properties": {}
    }
  ]
}
```

Output directory gồm:

- `selection_<split>.jsonl`: một dòng cho mỗi cặp `(question, schema unit)`,
  chứa `unit`, `unit_type`, và `label` nhị phân.
- `generation_<split>.jsonl`: `question`, `sub_schema` (chỉ gồm `nodes` và
  `relationships`) và gold `cypher`.
- `schemas.jsonl`: full canonical schema cho mỗi `schema_id`.
- `rejected_<split>.jsonl`: các mẫu không thể suy gold sub-schema chính xác,
  kèm diagnostics để audit.
- `normalization_issues_<split>.jsonl`: các schema không thể chuẩn hóa thành
  node/relation units. File giữ `schema_reference`, lỗi và, với Neo4j
  Text2Cypher, nguyên văn `raw_schema` để phân tích sau. Pipeline không tự
  suy đoán sửa những trường hợp này.
- `manifest.json`: cấu hình build và thống kê theo benchmark/split.

## Chất lượng chuẩn hóa

Canonical schema chuẩn hóa **cấu trúc** của ba nguồn về `nodes` và
`relationships`; nó không tự thêm property suy luận. Đặc biệt, property `name`
được gold Cypher dùng nhưng không xuất hiện trong metadata gốc của CypherBench
vẫn được giữ nguyên là thiếu. Điều này giữ corpus trung thực với benchmark gốc.

Adapter Neo4j hỗ trợ section chuẩn, relevant schema dạng prose, inline Cypher,
JSON inspection và Neo4j inspection repr. Những biến thể chưa hỗ trợ hoặc
không có topology sẽ nằm trong `normalization_issues_<split>.jsonl`, còn lỗi
coverage giữa gold Cypher và schema hợp lệ nằm trong `rejected_<split>.jsonl`.

Xem thêm về thiết kế và format tại [docs/data-preparation.md](docs/data-preparation.md).

## Distillation

Phần `src/distillation` tích hợp đầy đủ các baseline từ template: SFT,
FKL/RKL, SFKL/SRKL, CSD, AMID, HPD, BDL/DA-KD, FDD và adaptive DistiLLM.
Môi trường train yêu cầu Python 3.11+, Linux/WSL2, GPU NVIDIA hỗ trợ BF16 và
driver tương thích với PyTorch CUDA 12.8. Các lệnh train bên dưới chạy từ thư
mục gốc của repository.

### 1. Cài môi trường train

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-distillation.txt
```

DeepSpeed chỉ được bật trên Linux/WSL2. Có thể chạy SFT kiểm tra trên CPU hoặc
Windows bằng override `deepspeed=null bf16=false`, nhưng KD yêu cầu DeepSpeed
và BF16.

### 2. Tạo dữ liệu LlamaFactory

Nếu `data/prepared` đã tồn tại, chuyển nó sang OpenAI chat JSONL bằng:

```bash
python scripts/prepare_llamafactory_data.py \
  --input-dir data/prepared \
  --output-dir data/llamafactory \
  --overwrite
```

Lệnh tạo bốn dataset `cypher_prepared_train`, `cypher_prepared_eval`,
`cypher_prepared_test_generator`, `cypher_prepared_test_selector` và đăng ký
chúng trong `data/llamafactory/dataset_info.json`. Các YAML hiện tại đã trỏ
sẵn tới train/eval local này.

### 3. Train LoRA teacher Qwen

Train teacher `Qwen/Qwen3-4B-Instruct-2507`. Adapter được lưu tại
`results/qwen3/teacher_lora`:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/distillation/teacher_sft.yaml
```

Nếu chỉ dùng một GPU:

```bash
RUN_GPUS=0 bash scripts/train.sh configs/distillation/teacher_sft.yaml
```

### 4. Train SFT student làm baseline

Bước này không bắt buộc trước KD, nhưng dùng để lấy baseline student không có
teacher:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/distillation/student_sft.yaml
```

Output nằm tại `results/qwen3/student_sft`.

### 5. Train student bằng KD

Config project mặc định dùng FKL với `kd_ratio: 0.6`, tự nạp base teacher 4B
và LoRA adapter vừa train tại `results/qwen3/teacher_lora`:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/distillation/kd.yaml
```

Loss train là `0.4 * LM loss + 0.6 * FKL loss`; output mặc định nằm tại
`results/qwen3/fkl`.

Có thể thay adapter, dataset hoặc output bằng override:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/distillation/kd.yaml \
  ref_model_adapters=/path/to/teacher-adapter \
  dataset=cypher_prepared_train \
  eval_dataset=cypher_prepared_eval \
  output_dir=results/qwen3/my_fkl
```

### 6. Chạy các method khác

Các preset trong `configs/qwen` đã dùng dataset local và tự nạp adapter tại
`results/qwen3/teacher_lora`. Ví dụ AMID:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/qwen/amid.yaml
```

Tương tự có thể thay `amid.yaml` bằng `fkl.yaml`, `rkl.yaml`, `sfkl.yaml`,
`srkl.yaml`, `csd.yaml`, `hpd.yaml`, `da_kd.yaml`, các config `fdd_*` hoặc
`distillm_adaptive_*`.

Preset trong `configs/llama` dùng student Llama 3.2 1B và teacher Llama 3 8B.
Đăng nhập Hugging Face với tài khoản có quyền truy cập model gated, sau đó
train đúng LoRA teacher Llama:

```bash
export HF_TOKEN=your_huggingface_token
RUN_GPUS=0,1 bash scripts/train.sh configs/distillation/teacher_sft_llama.yaml
```

Adapter được lưu tại `results/llama3/teacher_lora` và mọi preset Llama tự nạp
đường dẫn này. Ví dụ:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/llama/amid.yaml
```

Không dùng adapter Qwen cho config Llama hoặc ngược lại.

### 7. GPU, resume và kiểm tra

Launcher mặc định dùng hai GPU. `RUN_GPUS` vừa chọn GPU vừa đặt số process;
có thể dùng `NPROC_PER_NODE` khi scheduler đã quản lý `CUDA_VISIBLE_DEVICES`:

```bash
RUN_GPUS=2,3 bash scripts/train.sh configs/distillation/kd.yaml
NPROC_PER_NODE=4 bash scripts/train.sh configs/distillation/kd.yaml
```

Resume từ checkpoint:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/distillation/kd.yaml \
  resume_from_checkpoint=results/qwen3/fkl/checkpoint-1000
```

Chạy test trước khi train dài:

```bash
python -m pytest
```
