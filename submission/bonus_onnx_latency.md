# ⭐ Bài tập về nhà 3 — Export YOLO26n sang ONNX, đo latency trên CPU

Notebook đã chạy: [submission/bonus_onnx_latency.ipynb](https://github.com/huybla166/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/submission/bonus_onnx_latency.ipynb)

Môi trường: Colab runtime CPU (AMD EPYC 7B12, 2 vCPU, PyTorch dùng 1 luồng), không GPU.
Phiên bản: `torch 2.11.0+cpu`, `ultralytics 8.4.171`, `onnxruntime 1.30.0` với `CPUExecutionProvider`.

## Cách làm

- **Export.** Gọi `YOLO("yolo26n.pt").export(format="onnx")`, opset 18, mỗi file 9.9 MB.
  - Mặc định `nms=None` giữ head **one-to-many**: output `(1, 84, 8400)`, NMS chạy ngoài graph lúc suy luận.
  - Thêm `nms=False` thì chọn head **one-to-one**: output `(1, 300, 6)`, không cần NMS.
- **Đo.** Đo trên `bus.jpg`, lấy thời gian preprocess / inference / postprocess từ `Results.speed`.
  - Lần 1 đo giống ô `bench` ở mục 1C: warm-up 1 lần, trung bình 30 lần.
  - Lần 2 đo chắc hơn: warm-up 3 lần, chạy 3 vòng xen kẽ qua cả 8 cấu hình, mỗi vòng 30 lần (90 mẫu), lấy trung vị.
  - Đo cả PyTorch trên cùng CPU để làm mốc.

## Kết quả (trung vị của 90 lần, ms)

| Backend | Head | conf | preprocess | inference | postprocess | tổng | số box |
|---|---|---:|---:|---:|---:|---:|---:|
| ONNX | one-to-many + NMS | 0.25 | 3.48 | 91.99 | **1.40** | 97.88 | 5 |
| ONNX | one-to-many + NMS | 0.001 | 3.32 | 87.15 | **1.82** | 92.40 | 186 |
| ONNX | one-to-one NMS-free | 0.25 | 3.35 | 87.82 | **0.47** | 91.70 | 5 |
| ONNX | one-to-one NMS-free | 0.001 | 3.34 | 87.81 | **0.50** | 91.82 | 177 |
| PyTorch | one-to-many + NMS | 0.25 | 2.93 | 89.62 | 1.10 | 93.82 | 5 |
| PyTorch | one-to-many + NMS | 0.001 | 2.81 | 87.17 | 1.57 | 91.81 | 203 |
| PyTorch | one-to-one NMS-free | 0.25 | 2.85 | 87.94 | 0.29 | 91.28 | 5 |
| PyTorch | one-to-one NMS-free | 0.001 | 2.87 | 88.91 | 0.31 | 92.20 | 204 |

## Nhận xét

1. **NMS chỉ làm khác cột postprocess.** Đây là cột duy nhất khác biệt có hệ thống giữa hai head.
   - Với ONNX, one-to-many + NMS mất 1.40 ms ở conf 0.25 và 1.82 ms ở conf 0.001. One-to-one chỉ mất 0.47 và 0.50 ms, tức nhanh hơn 3.0–3.6 lần. Với PyTorch, con số này là 3.8–5.1 lần.
   - NMS tốn thêm 0.93 ms ở conf 0.25 và 1.32 ms ở conf 0.001 (ONNX). Conf càng thấp thì càng nhiều box ứng viên phải qua NMS, sau NMS vẫn còn 186–203 box. Trong khi đó, head one-to-one gần như không đổi.
2. **Trên CPU, NMS chỉ chiếm khoảng 0.9–1.4% tổng thời gian.**
   - Inference mất khoảng 87–92 ms, chiếm khoảng 95% pipeline. Lượng NMS thêm vào (0.8–1.3 ms) còn nhỏ hơn dao động giữa các lần đo: p90 của tổng thời gian lên tới 95–127 ms.
   - Ở lần đo 1 (trung bình 30 lần liên tiếp), hiệu số tổng "one-to-many − one-to-one" nhảy từ −7.6 đến +11.9 ms. Vì vậy phải lấy trung vị trên nhiều vòng xen kẽ mới so sánh được.
   - Đối chiếu với GPU T4 ở notebook chính: NMS tốn số ms gần như nhau (postprocess 1.24–1.33 ms so với 0.41–0.43 ms). Nhưng ở đó inference chỉ khoảng 9–10 ms, nên NMS chiếm tỉ trọng lớn hơn nhiều.
   - Kết luận: với cảnh ít object như `bus.jpg`, bỏ NMS không làm pipeline trên CPU nhanh hơn đáng kể. Phần lợi tăng theo số box ứng viên, tức là ở cảnh đông hoặc khi conf thấp.
   - Bài này không đo NPU. Ở đó lợi ích nằm ở chỗ graph one-to-one `(1, 300, 6)` export trọn vẹn, không cần op NMS động chạy trên CPU host.
3. **ONNX Runtime và PyTorch có inference gần bằng nhau** (trung vị 87–92 ms so với 87–90 ms).
   - Lưu ý là file ONNX có input cố định 640×640 (8400 anchor). PyTorch letterbox `bus.jpg` thành (640, 480) (6300 anchor), nên ONNX Runtime xử lý nhiều hơn 33% pixel trong cùng khoảng thời gian.
   - Input khác nhau cũng giải thích vì sao ở conf 0.001 số box khác nhau (177–186 so với 203–204).
   - Muốn so sánh công bằng hơn thì export với `dynamic=True` hoặc `imgsz=(640, 480)`.
