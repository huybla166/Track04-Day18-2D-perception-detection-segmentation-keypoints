# Báo cáo Lab Ngày 18 — 2D Perception: Detection · Segmentation · Keypoints

## Notebook đã chạy

- GitHub (còn nguyên output): [lab_2d_perception_student.ipynb](https://github.com/huybla166/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb)
- Mở bản đã chạy trên Colab: [Open in Colab](https://colab.research.google.com/github/huybla166/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb)
- Bản gốc trên Google Drive (Colab): <https://colab.research.google.com/drive/1WSiwYtdPQiwu83ZeCCJsByOoxAH34AsK>

Môi trường: Google Colab, GPU Tesla T4 · `torch 2.11.0+cu130` · `ultralytics 8.4.171`.
Notebook chạy `Run all` trên runtime mới, không ô nào lỗi; ô 3A được chạy lại với `KP_THR = -100` theo yêu cầu của Q7.

## Kết quả chính

| Mục | Kết quả |
|---|---|
| 1B | `box_iou`, `nms`, `batched_nms` ✅ · NMS của mình giữ 5 box, khớp Ultralytics |
| 1C | 51 → 5 → 5 box · camera đếm 4 người, 1 xe buýt · bảng latency đủ 4 cấu hình |
| 1D ⭐ | `average_precision` ✅ · AP = 0.535 trên ví dụ slide |
| 2B | `mask_iou`, `polygon_to_mask`, `mask_to_yolo_seg` ✅ · mask IoU YOLO26n-seg ↔ Mask R-CNN 0.84–0.94 |
| 2C | `autolabel/bus.txt` hợp lệ: 5 object, round-trip IoU 0.967–0.983 |
| 3B, 3C | `oks`, `joint_angle` ✅ · lệch 8 px: mắt 0.53, hông 0.97 |
| 4A | `FLIP_IDX = [0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]` ✅ |
| 4B | 40 epoch, imgsz 640, T4, 4.4 phút · Pose mAP50 0.995 · Pose mAP50-95 0.457 · Box mAP50-95 0.930 |

## ⭐ 4C — Val lật gương: metric nào đã che lỗi `flip_idx`?

Hai model YOLO26n-pose train giống hệt nhau (40 epoch, imgsz 640, seed 0), chỉ khác `flip_idx`.
Tập val lật gương: 53 ảnh val lật ngang, nhãn đổi chỗ theo quy ước giải phẫu, giả lập hổ quay trái lúc triển khai.

| Model | Val gốc: Pose mAP50 | Val gốc: Pose mAP50-95 | Val lật gương: Pose mAP50 | Val lật gương: Pose mAP50-95 |
|---|---:|---:|---:|---:|
| `flip_idx` giải phẫu | 0.995 | 0.457 | 0.995 | 0.439 |
| `flip_idx` đồng nhất | 0.995 | 0.417 | 0.878 | 0.298 |

Box mAP50 của cả hai model đều 0.995 trên cả hai tập.

**Metric nào đã che lỗi.** Trên val gốc, mọi metric đều che lỗi: Box mAP và Pose mAP50 của hai model bằng nhau (0.995). Pose mAP50-95 chỉ chênh 0.04, chưa đủ để nghi ngờ.
Lý do là dữ liệu: cả 210 ảnh train và 53 ảnh val đều có hổ quay phải. Với `flip_idx` đồng nhất, model học quy tắc "chân phía camera là `right_*`". Trên val, quy tắc này luôn trùng với nhãn giải phẫu nên không bị phạt.
Box metric không bao giờ thấy được lỗi, vì box không có khái niệm trái/phải. Pose mAP50 với ngưỡng OKS 0.5 lại khá dễ dãi.

Chỉ trên val lật gương lỗi mới lộ ra. Model đồng nhất tụt còn Pose mAP50 0.878 và mAP50-95 0.298 (−29% so với val gốc), vì nó gọi chân trái là chân phải. Model giải phẫu gần như giữ nguyên (0.439).

**Thiết kế tập val.** Tập val phải bao phủ các biến đổi sẽ gặp lúc triển khai, chứ không chỉ cùng phân phối với train:

- có hổ quay cả hai hướng, lấy từ video hoặc camera khác (đừng cắt train/val từ cùng một video), hoặc ít nhất thêm một bản lật gương có nhãn giải phẫu như ở trên;
- báo cáo metric theo từng nhóm (quay trái / quay phải, từng keypoint) thay vì một con số trung bình;
- thêm một kiểm tra riêng cho lỗi đảo trái/phải: tỉ lệ ảnh mà đổi nhãn trái↔phải lại cho OKS cao hơn (ở 4B là 4/53 ảnh);
- chấm bằng σ riêng cho từng keypoint, ước lượng từ độ lệch giữa những người gán nhãn, thay vì σ = 1/12 cho mọi điểm.
