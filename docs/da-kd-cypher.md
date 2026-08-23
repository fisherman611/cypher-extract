# DA-KD cho Text-to-Cypher

Implementation trong `src/distillation` giữ ba thành phần của DA-KD:

- DDS: `student CE / teacher CE`, tính trên response token của từng mẫu.
- DiffUp/SDU: epoch đầu dùng toàn bộ dữ liệu; các epoch sau chấm lại toàn bộ
  dataset, sort DDS và lấy mẫu từ cả high-DDS lẫn low-DDS partition.
- BDL: `KL((1-lambda)p + lambda*q || lambda*p + (1-lambda)q)` với
  `lambda=0.9`.

Hai preset Qwen và Llama dùng 5 epochs. Với cosine schedule zero-based, tỷ lệ
dữ liệu lần lượt xấp xỉ `1.0000, 0.9045, 0.6545, 0.3455, 0.0955`. Cách đánh
chỉ số này khớp tỷ lệ iteration thực nghiệm được báo trong paper. Nếu low-DDS
partition không đủ số mẫu theo `tau`, implementation lấy toàn bộ low partition
rồi bù phần còn thiếu từ high partition, không duplicate mẫu.

`bdl_lambda=0.5` bị từ chối vì khi đó hai mixture giống hệt nhau và BDL luôn
bằng 0.

## Audit dữ liệu Cypher

Trước mỗi epoch sau epoch đầu, rank 0 ghi một file:

```text
<output_dir>/da_kd_audit/epoch_XXX.json
```

Mỗi file chứa:

- DDS min, p10, median, p90, max và mean;
- CE trung bình của student và teacher;
- tỷ lệ mẫu teacher có CE thấp hơn student;
- số mẫu thực tế lấy từ high/low partition;
- các mẫu DDS cao nhất và thấp nhất, gồm dataset index, DDS, student/teacher
  CE, prompt và gold Cypher đã decode.

`da_kd_audit_samples` điều khiển số mẫu ở mỗi đầu phân phối; đặt `0` để tắt ghi
audit. Khi kiểm tra audit nên chú ý mẫu có teacher CE cao, gold Cypher không
canonical, hoặc nhiều Cypher tương đương về ngữ nghĩa. Những trường hợp đó có
thể làm DDS phản ánh độ khớp chuỗi thay vì chất lượng truyền kiến thức.
