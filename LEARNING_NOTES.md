# Ghi chú học tập: Anomaly Detection

**Dự án**: Phát hiện lỗi trong sản xuất công nghiệp  
**Thời gian**: 14 ngày (Jun 9-23, 2026)  
**Hardware**: RTX 3050  

---

## 1. KHÁI NIỆM CƠ BẢN

### 1.1 Anomaly Detection là gì?

**Định nghĩa đơn giản:**
```
Anomaly = Tìm những thứ KHÁC BIỆT so với bình thường

Ví dụ:
- Bình thường: Bàn chải với lông nguyên vẹn
- Lỗi: Bàn chải với lông bị gãy
→ Hệ thống học "cái bình thường trông như thế nào"
→ Thấy cái khác → "LỖI!"
```

### 1.2 Unsupervised Approach (Cách tiếp cận của chúng ta)

```
- Training: Chỉ dùng hình BÌNH THƯỜNG
- Model học: "Đây là bình thường"
- Testing: 
  * Nếu hình mới giống bình thường → OK
  * Nếu hình mới khác → NG (Lỗi)
  
Lợi ích:
- Không cần label lỗi phức tạp
- Phát hiện các lỗi chưa thấy trước
```

---

## 2. PIPELINE CHI TIẾT

### Bước 1: Feature Extraction (Trích xuất đặc trưng)

```
Đầu vào: Ảnh (224×224×3)
     ↓
Backbone (ResNet50 / DINO / DINOv2)
     ↓
Đầu ra: Vector đặc trưng (1536 chiều)

Tại sao Feature?
- Pixel: 224×224×3 = 150K chiều (quá nhiều)
- Features: 1536 chiều (đủ thông tin, nhỏ gọn)
- Features = "cái model thấy" (semantic information)
```

### Bước 2: Memory Bank (Kho tham chiếu)

```
Công dụng: Lưu features từ ảnh BÌNH THƯỜNG
Giảm kích thước: k-center greedy (giữ 10% điểm xa nhất)
Lợi ích:
- Tránh lưu cả triệu vector
- Tăng tốc độ so sánh
```

### Bước 3: Anomaly Scoring (Tính điểm lỗi)

```
Cho ảnh test:
1. Trích features
2. Normalize (L2)
3. Tính distance đến mỗi point trong memory bank
4. Lấy distance nhỏ nhất (nearest neighbor)
5. Smooth spatial (Gaussian) nếu cần
6. Aggregate thành 1 số (max hoặc mean-top-k)

Kết quả: 1 số = "bao xa so với bình thường"
```

### Bước 4: Threshold (Ngưỡng quyết định)

```
Cách đặt (leakage-free):
- Dùng 20% ảnh bình thường "bị giữ lại" (held-out)
- Tính: mean + 2×std của scores này
- Threshold = ~2.3% false positive rate (Gaussian)

Quyết định:
- score > threshold → NG (Lỗi)
- score ≤ threshold → OK (Bình thường)
```

---

## 3. BA BACKBONE SO SÁNH

### ResNet50 (Baseline)

```
- CNN tập trung lớp
- Huấn luyện supervised (ImageNet)
- Input: 224×224
- Features: 1536 chiều
- Công cụ: layer2 + layer3
```

**Hiệu suất**: AUROC 97.65%, F1 91.55%

### DINO ViT-S/8 (Self-supervised)

```
- Vision Transformer (Attention-based)
- Tự học (DINO) không cần label
- Input: 224×224
- Features: 384 chiều
- Global attention → nhìn toàn bộ ảnh
```

**Hiệu suất**: AUROC 98.55%, F1 94.72% (+0.9%)

### DINOv2 ViT-S/14 (State-of-the-art)

```
- Vision Transformer cải tiến
- Học tốt hơn DINO
- Input: 518×518 (gấp 2× độ phân giải)
- Features: 384 chiều
- Patch lớn (14×14) nhưng ảnh to → chi tiết hơn
```

**Hiệu suất**: AUROC 99.30%, F1 96.15% (+1.7%)

---

## 4. THỐNG NHẤT KẾT QUẢ

### Tính chất Monotonic (Tuyến tính)

```
ResNet → DINO → DINOv2
97.65% → 98.55% → 99.30% (AUROC)

Kết luận:
"Feature representation quality là driver chính"
→ Backbone tốt hơn = Kết quả tốt hơn
→ Không cần pipeline phức tạp
```

### Per-Category Insights

**Loại 1: Dễ (bottle, hazelnut, leather)**
```
- Lỗi to, rõ ràng
- Tất cả backbone > 99%
- Chênh lệch: < 1%
→ Feature cơ bản đủ
```

**Loại 2: Khó (screw, toothbrush, transistor)**
```
- Lỗi nhỏ, tinh tế
- ResNet: 89-94%
- DINOv2: 96-99%
- Chênh lệch: +5-6%
→ Độ phân giải cao cần thiết
```

---

## 5. CÁC QUYẾT ĐỊNH THIẾT KẾ

### Leakage-Free Evaluation

```
❌ Sai (Data leakage):
- Dùng test labels để chọn threshold
- F1 cao nhưng không thật

✅ Đúng (Leakage-free):
- Split train: 80% (memory) / 20% (calibration)
- Chọn threshold từ 20% này
- Test set chỉ dùng cuối cùng
→ Công bằng, tái lập được
```

### One Pipeline, Swap Backbone

```
Pipeline = Fixed
├─ Memory bank
├─ k-center selection
├─ NN distance
├─ Gaussian smooth
└─ Threshold μ+2σ

Chỉ thay: Feature extractor
→ Cách duy nhất để biết backbone tác động bao nhiêu
```

### Minimal Design

```
Không cần:
- Local aggregation
- Score reweighting
- Complex post-processing

Tại sao?
- Dễ giải thích
- Dễ bảo trì
- Dễ compare
```

---

## 6. GIỚI HẠN

1. **Chỉ image-level**: Không biết lỗi ở đâu (no localization)
2. **Một lần chạy**: Không có mean±std qua nhiều lần
3. **Không novel**: DINOv2 + Memory bank đã tồn tại trước
4. **Chỉ MVTec-AD**: Không test trên dataset khác

---

## 7. GỌI TÓM LẠI

```
Câu hỏi: Pipeline fixed, backbone tốt thêm bao nhiêu?
Câu trả lời: +1.7% AUROC (ResNet→DINOv2)
             +4.6% F1 (91.5%→96.15%)

Suy luận: Feature quality = Primary driver
         Đầu tư vào backbone > đầu tư vào pipeline phức tạp
```

---

**Phương châm**: Simplify, isolate, measure.
