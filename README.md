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

```powershell
python scripts\build_schema_grounding_data.py --help
```

Smoke test với 25 mẫu train của **mỗi** benchmark. Dùng `--negative-ratio 4`
để giới hạn tối đa bốn negative unit cho mỗi positive unit:

```powershell
python scripts\build_schema_grounding_data.py `
  --output-dir $env:TEMP\schema-grounding-smoke `
  --max-examples 25 `
  --negative-ratio 4
```

Tạo corpus train đầy đủ từ cả ba benchmark, giữ mọi schema unit negative:

```powershell
python scripts\build_schema_grounding_data.py `
  --output-dir data\schema_grounding
```

Nếu chỉ cần corpus CypherBench để train, chỉ định nguồn một cách tường minh:

```powershell
python scripts\build_schema_grounding_data.py `
  --sources cypherbench `
  --output-dir data\cypherbench_schema_grounding
```

Tạo corpus train với negative sampling, phù hợp khi full schema lớn:

```powershell
python scripts\build_schema_grounding_data.py `
  --output-dir data\schema_grounding_4neg `
  --negative-ratio 4
```

Tạo dữ liệu test để đánh giá offline selector/generator. Không trộn output này
vào dữ liệu SFT train:

```powershell
python scripts\build_schema_grounding_data.py `
  --splits test `
  --output-dir data\schema_grounding_test
```

Lệnh từ chối ghi đè output cũ. Chỉ thêm `--overwrite` khi chủ động muốn tạo lại
toàn bộ các file trong output directory.

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
