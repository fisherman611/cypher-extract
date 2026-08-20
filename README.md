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
