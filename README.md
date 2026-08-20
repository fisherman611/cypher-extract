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

Tạo đầy đủ cả 3 tập `train`, `dev`, `test` cho CypherBench với negative sampling (4:1):

```bash
python scripts/build_schema_grounding_data.py \
  --sources cypherbench \
  --splits train,dev,test \
  --output-dir data/cypherbench_grounding_4neg \
  --negative-ratio 4
```

Nếu muốn giữ **100% negative units** (Full - không sampling):

```bash
python scripts/build_schema_grounding_data.py \
  --sources cypherbench \
  --splits train,dev,test \
  --output-dir data/cypherbench_grounding_full
```

Hoặc tạo riêng từng tập theo nhu cầu:

```bash
# Chỉ tập Train
python scripts/build_schema_grounding_data.py \
  --sources cypherbench \
  --splits train \
  --output-dir data/cypherbench_train \
  --negative-ratio 4

# Chỉ tập Dev (Validation)
python scripts/build_schema_grounding_data.py \
  --sources cypherbench \
  --splits dev \
  --output-dir data/cypherbench_dev \
  --negative-ratio 4

# Chỉ tập Test (Offline Evaluation)
python scripts/build_schema_grounding_data.py \
  --sources cypherbench \
  --splits test \
  --output-dir data/cypherbench_test \
  --negative-ratio 4
```

---

### 3. Neo4j Text2Cypher (Train, Dev, Test)

Tạo dữ liệu riêng cho `neo4j_text2cypher` (khuyên dùng `--negative-ratio 4` do full schema có số lượng node/relation rất lớn):

```bash
# Tạo cả 3 tập train, dev, test
python scripts/build_schema_grounding_data.py \
  --sources neo4j_text2cypher \
  --splits train,dev,test \
  --output-dir data/neo4j_grounding_4neg \
  --negative-ratio 4

# Chỉ tập Train
python scripts/build_schema_grounding_data.py \
  --sources neo4j_text2cypher \
  --splits train \
  --output-dir data/neo4j_train_4neg \
  --negative-ratio 4

# Chỉ tập Dev (Validation)
python scripts/build_schema_grounding_data.py \
  --sources neo4j_text2cypher \
  --splits dev \
  --output-dir data/neo4j_dev_4neg \
  --negative-ratio 4

# Chỉ tập Test
python scripts/build_schema_grounding_data.py \
  --sources neo4j_text2cypher \
  --splits test \
  --output-dir data/neo4j_test_4neg \
  --negative-ratio 4
```

---

### 4. Mind-the-Query (Train, Dev, Test)

```bash
# Tạo cả 3 tập train, dev, test
python scripts/build_schema_grounding_data.py \
  --sources mind_the_query \
  --splits train,dev,test \
  --output-dir data/mind_the_query_grounding_4neg \
  --negative-ratio 4
```

---

### 5. Tạo toàn bộ 3 Benchmarks cùng lúc

```bash
# Tạo cả 3 nguồn (cypherbench, mind_the_query, neo4j_text2cypher)
python scripts/build_schema_grounding_data.py \
  --splits train,dev,test \
  --output-dir data/all_benchmarks_grounding_4neg \
  --negative-ratio 4
```

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
