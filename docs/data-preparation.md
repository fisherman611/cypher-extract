# Kế hoạch và pipeline tạo dữ liệu schema grounding

## Mục tiêu

Từ mỗi bản ghi `(question, full schema, gold Cypher)`, pipeline tạo hai corpora:

```text
Schema selection:  question + schema unit -> label 0/1
Cypher generation: question + gold sub-schema -> gold Cypher
```

Pipeline chỉ dùng gold Cypher để tạo nhãn offline.  Gold Cypher tuyệt đối không
được dùng trong inference hoặc để tạo sub-schema cho test-time prediction.

## Kế hoạch hiện thực

1. **Đọc benchmark qua adapter riêng.** CypherBench, Mind-the-Query và Neo4j
   Text2Cypher có định dạng schema khác nhau, nhưng đều được đưa vào một model
   chung trước khi tạo data.
2. **Chuẩn hóa schema.** Mỗi full schema được biểu diễn bằng node unit
   `(label, properties)` và relation unit `(source, type, target, properties)`.
   Kiểu dữ liệu thường gặp được chuẩn hóa, ví dụ `str -> STRING`, `int ->
   INTEGER`, `double -> FLOAT`.
3. **Trích gold sub-schema.** Parser schema-guided nhận diện node labels,
   relation types, hướng cạnh và variable reuse trong Cypher rồi ánh xạ chúng về
   ID của unit trong full schema.
4. **Kiểm soát chất lượng.** Mặc định strict: một query có type không có trong
   schema, node không xác định được label, hoặc relation mơ hồ sẽ không được đưa
   vào training data. Nó được ghi sang `rejected_<split>.jsonl` cùng diagnostics.
   Nếu schema nguồn không chuẩn hóa được thành node/relation units, record được
   ghi riêng vào `normalization_issues_<split>.jsonl`; pipeline không tự suy
   đoán để sửa schema gốc.
5. **Xuất JSONL streaming.** Việc ghi theo dòng tránh phải giữ toàn bộ 68k+ mẫu
   vào bộ nhớ. `--negative-ratio` hỗ trợ subsample negative units nếu file
   selection quá lớn.

## Canonical schema

```json
{
  "schema_id": "mind_the_query:covid:<hash>",
  "benchmark": "mind_the_query",
  "graph": "covid",
  "nodes": [
    {
      "label": "Person",
      "properties": {"id": "STRING", "name": "STRING"}
    }
  ],
  "relationships": [
    {
      "source": "Person",
      "type": "VISITS",
      "target": "Place",
      "properties": {"duration": "DURATION"}
    }
  ]
}
```

Canonical schema chỉ mô tả graph. `unit_id`, `kind` và text prompt chỉ xuất hiện
trong `selection_<split>.jsonl`. Sub-schema cho task generation chỉ có `nodes`
và `relationships`. Relation unit ID bao gồm cả hai endpoint, nên hai
relationship trùng type ở các vị trí khác nhau không bị gộp nhầm.

## Đầu ra

`schemas.jsonl`
: Một canonical full schema cho mỗi `schema_id` duy nhất.

`selection_<split>.jsonl`
: Một dòng cho mỗi `(question, schema unit)`, có `question`, `unit`,
  `unit_type`, và binary `label`.

`generation_<split>.jsonl`
: Một dòng cho mỗi example hợp lệ, có `question`, `sub_schema` chỉ gồm
  `nodes` và `relationships`, và `cypher` vàng.

`rejected_<split>.jsonl`
: Các mẫu không đủ coverage, giữ lại để audit/cải thiện parser thay vì bị mất
  im lặng.

`normalization_issues_<split>.jsonl`
: Các schema không tạo được canonical units. Mỗi dòng có source, split, graph,
  `schema_reference`, lỗi chuẩn hóa và `raw_schema` khi nguồn Neo4j cung cấp
  schema trực tiếp trong record. File này là backlog để phân tích các format lỗi
  hoặc chưa được adapter hỗ trợ; schema không bị chỉnh sửa hay bổ sung property.

`manifest.json`
: Version format, cấu hình build, số lượng positive/negative, số record bị
  loại và các chỉ số coverage theo benchmark/split.

## Chạy pipeline

Smoke test, ba benchmark, 25 mẫu train cho mỗi benchmark:

```powershell
python scripts/build_schema_grounding_data.py `
  --output-dir $env:TEMP/schema-grounding-smoke `
  --max-examples 25
```

Tạo corpus train đầy đủ và giữ mọi negative unit:

```powershell
python scripts/build_schema_grounding_data.py `
  --output-dir data/schema_grounding
```

Giới hạn tối đa 4 negative units trên mỗi positive unit:

```powershell
python scripts/build_schema_grounding_data.py `
  --output-dir data/schema_grounding_4neg `
  --negative-ratio 4
```

Output cũ không bị ghi đè trừ khi truyền `--overwrite`. Test split được hỗ trợ
để đánh giá selector offline, nhưng không nên được trộn vào SFT training data.
