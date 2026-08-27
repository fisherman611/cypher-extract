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

Train teacher trước, sau đó chạy tuần tự toàn bộ preset Qwen (SFT student và
tất cả KD baseline):

```bash
RUN_GPUS=0,1 bash scripts/train_all_qwen.sh
```

Mọi override phía sau script được chuyển cho từng run. Ví dụ smoke-test một epoch:

```bash
RUN_GPUS=0,1 bash scripts/train_all_qwen.sh num_train_epochs=1
```

Teacher là dependency bắt buộc nên nếu bước teacher lỗi, script luôn dừng. Sau
đó, mặc định script dừng ngay khi một method lỗi; đặt `CONTINUE_ON_ERROR=1` để
chạy các method còn lại. Log riêng của từng method nằm trong
`results/qwen3/run_all_logs/<timestamp>/`.

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

## Two-stage inference trên Linux

Pipeline inference thực hiện tuần tự hai task bằng cùng một LoRA adapter:

1. Chạy selector cho từng cặp `question + schema unit` để dự đoán `YES/NO`
   bằng greedy decoding (`do_sample=False`, tối đa một token).
2. Merge các unit `YES` thành predicted sub-schema, rồi đưa
   `question + predicted sub-schema` vào generator để sinh Cypher.

Prediction path không sử dụng gold selector label, gold sub-schema hoặc gold
Cypher. Các trường gold chỉ được đọc sau generation để tính metric.

### 1. Yêu cầu

- Linux x86-64.
- Python 3.11, khuyến nghị dùng virtual environment.
- GPU NVIDIA hỗ trợ BF16 và driver tương thích với CUDA 12.8.
- Đủ dung lượng Hugging Face cache cho base models và 13 LoRA adapters.
- Ba test dataset đã tồn tại:

  ```text
  data/cypherbench_schema_grounding_full_final/
  data/mind_the_query_schema_grounding_full/
  data/neo4j_text2cypher_schema_grounding_full/
  ```

Mỗi directory phải có `selection_test.jsonl` và `generation_test.jsonl`.

Từ repository root, tạo môi trường và cài dependency:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-distillation.txt
```

Đặt Hugging Face token nếu model repository yêu cầu authentication:

```bash
export HF_READ_TOKEN="hf_..."
```

Có thể đặt cache trên ổ đĩa riêng:

```bash
export HF_HOME="/mnt/models/huggingface"
mkdir -p "$HF_HOME"
```

Kiểm tra CLI trước khi chạy:

```bash
python scripts/infer_two_stage.py --help
```

### 2. Model được chạy

`--methods all` chạy 13 model sau:

```text
teacher_lora
sft
fkl
rkl
sfkl
srkl
csd
hpd
amid
fdd_sfkl
fdd_srkl
distillm_adaptive_sfkl
distillm_adaptive_srkl
```

`da_kd` không nằm trong inference suite. Với mỗi model, script truy vấn
`distillation-sql/nothing-extract/qwen3/<method>/checkpoint-N` và tự dùng
checkpoint có `N` lớn nhất. Script chỉ tải adapter/tokenizer files phục vụ
inference, không tải DeepSpeed optimizer states trong `global_step*`.

Base model được lấy từ `adapter_config.json`:

- `teacher_lora`: `Qwen/Qwen3-4B-Instruct-2507`;
- 12 student methods: `Qwen/Qwen3-0.6B`.

### 3. Chạy toàn bộ trên một GPU

Chạy đủ 13 model trên cả ba benchmark bằng GPU 0:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_all_qwen.sh \
  --selector-batch-size 128 \
  --generator-batch-size 16 \
  --dtype bfloat16 \
  --device cuda
```

Lệnh trên chạy model tuần tự để không giữ nhiều model trong VRAM. Trong mỗi
model, selector units và generator samples được infer theo batch. Nếu một batch
gây CUDA OOM, runner tự chia đôi batch cho đến khi chạy được hoặc chỉ còn một
sample, sau đó ghi nhớ batch size an toàn cho các batch có cùng token budget.

Để chỉ định Hugging Face cache qua CLI:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_all_qwen.sh \
  --cache-dir /mnt/models/huggingface \
  --selector-batch-size 128 \
  --generator-batch-size 16
```

### 4. Chạy một số method hoặc một benchmark

Ví dụ chạy teacher, SFT và FKL trên CypherBench:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_two_stage.py \
  --methods teacher_lora,sft,fkl \
  --datasets cypherbench \
  --selector-batch-size 64 \
  --generator-batch-size 8 \
  --dtype bfloat16 \
  --device cuda
```

Các giá trị hợp lệ của `--datasets`:

```text
cypherbench
mind_the_query
neo4j_text2cypher
```

Ví dụ chạy một method trên cả ba benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_two_stage.py \
  --methods hpd \
  --datasets cypherbench,mind_the_query,neo4j_text2cypher
```

### 5. Chạy song song trên nhiều GPU

Pipeline song song schema units bằng batching trên từng GPU. Để chạy nhiều
method đồng thời, khởi tạo một process cho mỗi GPU và đảm bảo danh sách method
không trùng nhau. Ví dụ với bốn GPU:

```bash
mkdir -p logs/inference

CUDA_VISIBLE_DEVICES=0 python scripts/infer_two_stage.py \
  --methods teacher_lora,sft,fkl \
  > logs/inference/gpu0.log 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/infer_two_stage.py \
  --methods rkl,sfkl,srkl,csd \
  > logs/inference/gpu1.log 2>&1 &
pid1=$!

CUDA_VISIBLE_DEVICES=2 python scripts/infer_two_stage.py \
  --methods hpd,amid,fdd_sfkl \
  > logs/inference/gpu2.log 2>&1 &
pid2=$!

CUDA_VISIBLE_DEVICES=3 python scripts/infer_two_stage.py \
  --methods fdd_srkl,distillm_adaptive_sfkl,distillm_adaptive_srkl \
  > logs/inference/gpu3.log 2>&1 &
pid3=$!

wait "$pid0" "$pid1" "$pid2" "$pid3"
```

Theo dõi log:

```bash
tail -f logs/inference/gpu0.log
```

Mỗi process nhìn GPU được gán qua `CUDA_VISIBLE_DEVICES` như `cuda:0`, vì vậy
giữ `--device cuda` hoặc bỏ tham số này.

### 6. Resume

Selector và generator ghi kết quả vào file `.partial`, sau đó mới publish file
JSONL hoàn chỉnh. Nếu process bị dừng, chạy lại chính xác command cũ:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_two_stage.py \
  --methods teacher_lora,sft,fkl \
  --datasets cypherbench \
  --selector-batch-size 64 \
  --generator-batch-size 8
```

Pipeline sẽ:

- tiếp tục selector/generator từ row cuối của file `.partial`;
- reuse stage đã hoàn thành;
- tính lại metrics và manifest sau khi đủ output.

`run_config.json` khóa immutable Hugging Face commit SHA, input paths, SHA-256
của input/prompt, ChatML template và inference options. Nếu `last_ckpt` trên Hugging Face thay đổi, command dùng
option khác, dataset được build lại hoặc prompt thay đổi, pipeline sẽ không
reuse output cũ. Khi đó chọn output directory mới:

```bash
python scripts/infer_two_stage.py \
  --methods sft \
  --output-dir results/inference/qwen3-run-2
```

### 7. Output

Kết quả mặc định nằm tại:

```text
results/inference/qwen3/seed<seed>/<method>/<dataset>/
├── run_config.json
├── selector_predictions.jsonl
├── predicted_subschemas.jsonl
├── generator_predictions.jsonl
├── metrics.json
└── manifest.json
```

- `selector_predictions.jsonl`: label và raw output của từng schema unit.
- `predicted_subschemas.jsonl`: sub-schema sau merge và các endpoint node được
  closure tự động.
- `generator_predictions.jsonl`: raw generator output, parsed Cypher, prompt
  length và reference dùng để đánh giá.
- `metrics.json`: selector accuracy/precision/recall/F1, Cypher exact
  match/ROUGE và schema diagnostics.
- `manifest.json`: checkpoint, options, thời gian và trạng thái từng stage.

Xem metric của một run:

```bash
jq . results/inference/qwen3/seed10/sft/cypherbench/metrics.json
```

Xem một prediction:

```bash
head -n 1 results/inference/qwen3/seed10/sft/cypherbench/generator_predictions.jsonl | jq .
```

### Chấm Cypher bằng Neo4j

Project có ba metric execution-based lấy từ pipeline CypherKD:

- `execution_accuracy`: so sánh kết quả thực thi của Cypher dự đoán và gold;
- `psjs`: Jaccard của provenance subgraph (các node được phần `MATCH` chạm tới);
- `executable`: kiểm tra Cypher dự đoán có chạy được hay không.

Cài Neo4j driver và chấm trực tiếp output của two-stage inference trên PowerShell.
Kết quả eval được tách khỏi inference và lưu dưới `results/evaluation/`:

```powershell
python -m pip install -e ".[evaluation]"

Copy-Item .env.example .env
# Sau đó sửa NEO4J_PASSWORD và các cấu hình Neo4j trong .env.

evaluate-cypher `
  --input results/inference/qwen3/seed10/sft/cypherbench/generator_predictions.jsonl `
  --name cypherbench-db `
  --graph nba
```

Eval toàn bộ 5 inference seed cho SFT/CypherBench graph `nba`:

```powershell
$seeds = 10, 42, 50, 100, 1234

foreach ($seed in $seeds) {
  evaluate-cypher `
    --input "results/inference/qwen3/seed$seed/sft/cypherbench/generator_predictions.jsonl" `
    --name cypherbench-db `
    --graph nba
}
```

Kết quả được ghi tự động vào các folder tương ứng:

```text
results/evaluation/qwen3/seed10/sft/cypherbench/nba/
results/evaluation/qwen3/seed42/sft/cypherbench/nba/
results/evaluation/qwen3/seed50/sft/cypherbench/nba/
results/evaluation/qwen3/seed100/sft/cypherbench/nba/
results/evaluation/qwen3/seed1234/sft/cypherbench/nba/
```

Eval và merge toàn bộ 7 graph CypherBench cho cả 5 seed:

```powershell
$seeds = 10, 42, 50, 100, 1234
$graphs = "company", "fictional_character", "flight_accident", "geography", "movie", "nba", "politics"

foreach ($seed in $seeds) {
  foreach ($graph in $graphs) {
    evaluate-cypher `
      --input "results/inference/qwen3/seed$seed/sft/cypherbench/generator_predictions.jsonl" `
      --name cypherbench-db `
      --graph $graph
  }

  merge-cypher-evaluations `
    --input-dir "results/evaluation/qwen3/seed$seed/sft/cypherbench"
}
```

Mỗi seed tạo thêm hai file gộp ở folder `cypherbench`:

```text
all_graphs_cypher_scores.jsonl
all_graphs_summary.json
```

`all_graphs_summary.json` chứa điểm `overall` trên toàn bộ sample và breakdown
riêng cho từng graph. Lệnh merge sẽ báo lỗi nếu thiếu graph, graph trong record
không khớp folder hoặc có ID trùng.

Có thể chọn metric với `--metrics execution_accuracy executable`. CLI tự load
`NEO4J_URI`, `NEO4J_USERNAME`, và `NEO4J_PASSWORD` từ `.env`. Database được chọn
bằng graph (`flight_accident` được đổi thành database `flight.accident`) đúng theo
CypherKD. `cypherbench-db` và `mind-the-query-db` là logical connector name, chọn
bằng `--name`; graph chọn bằng `--graph`, mặc định là `nba`. Có thể dùng
`--database` để override database thực tế. Với command trên, CLI ghi kết quả từng
mẫu vào `results/evaluation/qwen3/seed10/sft/cypherbench/nba/cypher_scores.jsonl` và
trung bình toàn bộ metric vào file `cypher_scores_summary.json` trong cùng folder.
Đường dẫn output được suy ra tự động từ `--input` và `--graph`; vẫn có thể truyền
`--output` nếu muốn ghi sang vị trí khác.

### 8. Điều chỉnh VRAM

Nếu GPU ít VRAM, giảm batch size. Với `teacher_lora` 4B có thể bắt đầu bằng:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_two_stage.py \
  --methods teacher_lora \
  --selector-batch-size 32 \
  --generator-batch-size 4 \
  --dtype bfloat16
```

Nếu GPU không hỗ trợ BF16, thử `--dtype float16`. Có thể giữ LoRA adapter chưa
merge bằng `--no-merge-adapter`, nhưng inference thường chậm hơn.

Generation mặc định dùng sampling theo CypherKD (`do_sample=True`,
`temperature=0.5`, `top_p=0.95`, `top_k=0`, `num_beams=1`) và ChatML
`qwen3_nothink` đúng format LlamaFactory, không chèn thẻ `<think>`. Mỗi method/dataset chạy lần lượt với các seed
`10,42,50,100,1234`, lưu vào folder `seed10`, `seed42`, ... Selector dùng greedy
decoding với nhãn một-token `YES/NO`; generator dùng tối đa 256 new tokens. Có thể override:

```bash
python scripts/infer_two_stage.py \
  --methods sft \
  --seeds 10,42,50,100,1234 \
  --temperature 0.5 \
  --top-p 0.95 \
  --top-k 0 \
  --num-beams 1 \
  --generator-max-new-tokens 256
```

Tài liệu chi tiết về merge policy và output schema nằm tại
[`docs/two-stage-inference.md`](docs/two-stage-inference.md).
