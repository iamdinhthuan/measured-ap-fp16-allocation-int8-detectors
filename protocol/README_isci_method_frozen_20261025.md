# Đề xuất method cho Information Sciences (25-10-2026)

> Bối cảnh: bài *Capacity-Stratified Analysis of INT8/FP8…* bị IMAVIS desk reject. Editor nêu 3 lý do: (1) thí nghiệm chỉ ở mức baseline, không so với các công trình SCI/WoS 2022–2026; (2) đóng góp mang tính gia tăng vì chỉ ghép các kỹ thuật đã có; (3) thiếu so sánh tài liệu đầy đủ và thiếu đoạn nói rõ bài kế thừa và khác biệt ở đâu (kể cả so với bài trước của chính tác giả).
> Anh quyết định nâng bài thành **bài method** để nộp *Information Sciences* (INS).
> File này là **đề xuất để anh duyệt phạm vi** trước khi em chạy grid GPU lớn. Mọi số liệu dưới đây là thật, đã chạy xong, và có đường dẫn tới file gốc.

---

## 1. Ý tưởng một câu

**Group-robust selective-precision PTQ:** cho trước một detector đã lượng tử hoá INT8 và một ngân sách tính toán (tỉ lệ MAC hoặc độ trễ được phép chạy FP16), chọn tập block cần giữ FP16. Mục tiêu không phải tối đa mAP trung bình mà là **tối thiểu hoá tổn thất tương đối của nhóm bị thiệt nhất** (small/medium/large; về sau có thể là ngày/đêm, lớp hiếm…).
- Độ nhạy được đo **trực tiếp trên engine TensorRT thật** (engine-in-the-loop), trên tập tuning lấy từ **train**.
- Bài toán chọn block được giải **chính xác bằng MILP**.

### Vì sao không phải "ghép kỹ thuật cũ"

Đây là cách trả lời lý do (2) của editor:

- Các phương pháp mixed-precision hiện có tối ưu **loss trung bình** hoặc một proxy trung bình: HAWQ dùng Hessian trace, ModelOpt `auto_quantize` dùng loss trung bình, Qualcomm 2302.05397 dùng SQNR.
- Dữ liệu của bài cũ cho thấy loss trung bình **che mất** chênh lệch giữa các nhóm. Trong 13/15 model, sau entropy-INT8 tổn thất tương đối của nhóm small cao gấp **1.5–2.5 lần** nhóm large (bảng ở §3.1).
- Vì vậy, chọn block theo độ nhạy trung bình và chọn theo minimax có thể ra **hai tập block khác nhau** ở cùng ngân sách. Điều này đã thấy trên yolo11s (§3.3).
- Đây là phần mở rộng có lý do từ chính kết quả của bài cũ (DIFF đổi dấu theo capacity), không phải ghép kỹ thuật cho đủ.

---

## 2. Đóng góp dự kiến (bản nháp, chỉnh theo kết quả)

1. **Bài toán.** Phát biểu *group-robust mixed-precision allocation* cho detector: minimax tổn thất tương đối theo nhóm, ràng buộc ngân sách MAC/độ trễ. Bài toán viết được dưới dạng MILP nhỏ (khoảng 30–60 biến nhị phân) và giải chính xác trong mili-giây.
2. **Đo độ nhạy trên engine thật.** Restore từng block về FP16 bằng phẫu thuật đồ thị Q/DQ (`qdq_restore.py`), giữ nguyên toàn bộ scale của các block khác, build engine TRT 11.1 strongly-typed. Không dùng fake-quant cho phần chọn, nên độ trễ đo được là độ trễ thật.
3. **Lý thuyết** (phần INS cần):
   - (a) Cận sai số giữa worst-group gap dự đoán (giả định cộng tính) và gap thật, tách thành hai thành phần: tương tác giữa các block và sai số ước lượng do tập tuning hữu hạn.
   - (b) Phiên bản *scenario-robust*: minimax trên cả nhóm lẫn bootstrap resample, kèm đảm bảo xác suất.
   - (c) Điều kiện để lời giải minimax khác lời giải trung bình.
4. **Thực nghiệm** trên YOLO11/YOLOv8/YOLO26 (m/l/x là chính), COCO và ít nhất một bộ dữ liệu với kiểu nhóm khác. Có độ trễ thật trên Jetson Orin (không có FP8) và RTX 5090. So với baseline 2022–2026 (§5).

---

## 3. Bằng chứng sơ bộ (đã chạy, ngày 25-10)

### 3.1. Small objects chịu tổn thất tương đối lớn hơn

Số liệu canonical lấy từ `metrics/coco_5090/`, COCO val2017. Tổn thất tương đối tính bằng (AP_g(FP32) − AP_g(q)) / AP_g(FP32).

| model | recipe | S % | M % | L % |
|---|---|---|---|---|
| yolo11s | int8 max | 12.0 | 6.3 | 5.9 |
| yolo11s | int8 entropy | 4.9 | 4.1 | 3.6 |
| yolo11l | int8 entropy | 5.2 | 3.4 | 3.0 |
| yolov8l | int8 entropy | 6.7 | 2.1 | 2.2 |
| yolo26s | int8 entropy | 16.4 | 7.9 | 6.4 |
| yolo26l | int8 entropy | 12.6 | 8.7 | 7.9 |
| yolo26x | int8 entropy | 8.3 | 5.3 | 3.5 |

- Với entropy-INT8 (recipe uniform mạnh nhất), tổn thất tương đối của S lớn hơn L ở 13/15 model. Hai ngoại lệ là yolo11n và yolo26n.
- Khoảng cách từ entropy-INT8 lên FP8 khá lớn: khoảng 1 AP ở YOLO11/v8 và 2.4–5 AP ở YOLO26. **YOLO26 là nơi method có nhiều dư địa nhất.**

### 3.2. Tổn thất INT8 nằm chủ yếu ở backbone/neck, không phải ở đầu ra

Model yolo11s, val2017, chỉ dùng để thăm dò. File: `server:isci_method_20261025/logs/gate_yolo11s.out`.

| biến thể | AP | AP_S | AP_M | AP_L |
|---|---|---|---|---|
| FP32 | 46.02 | 28.46 | 50.53 | 62.99 |
| INT8 max | 43.08 | 25.05 | 47.35 | 59.30 |
| INT8 entropy | 44.34 | 27.07 | 48.46 | 60.71 |
| INT8 max, conv đầu ra head giữ FP16 | 43.40 | 25.77 | 47.68 | 59.46 |
| INT8 entropy, conv đầu ra head giữ FP16 | 44.88 | 27.77 | 49.20 | 61.26 |
| INT8 entropy, cả head giữ FP16 | 44.96 | 27.63 | 49.47 | 61.64 |

- Pipeline mới tái lập đúng số canonical: int8max 43.08, FP32 lệch không quá 0.01.
- Giữ FP16 cho conv đầu ra của head chỉ lấy lại khoảng 0.3–0.5 AP. Phần lớn tổn thất nằm ở thân mạng, nên chọn block là việc có ý nghĩa.

### 3.3. Độ nhạy từng block, đo trên engine thật

Setup: yolo11s, base int8max, tập tuning 2000 ảnh train.
- 30 block. Mỗi block được restore riêng về FP16, rồi build và đánh giá engine.
- Gap từ INT8 lên FP16 trên tập tuning: AP 3.26; S 3.17 / M 4.27 / L 3.48.
- Chỉ riêng **m2** đã lấy lại 1.42 AP, tức 44% gap, với 6.1% MAC.
- **head.dfl** lấy lại 0.43 AP với chi phí MAC gần bằng 0.
- Phần lớn các block khác cho kết quả trong khoảng ±0.3, tức ngang mức nhiễu.
- File: `server:isci_method_20261025/metrics/sens_yolo11s_int8max.json`.

Kết quả của solver (`allocate.py`; đây là **dự đoán** theo giả định cộng tính, chưa phải đo thật):

| ngân sách MAC | mục tiêu | block chọn | gap AP dự đoán | gap S/M/L dự đoán | worst-group tương đối |
|---|---|---|---|---|---|
| 10% | mean | m2, m10, dfl | 1.24 | 1.08/1.19/1.01 | 3.48% |
| 10% | minimax | m0, m2, m9, dfl | 1.44 | 0.61/1.10/1.36 | 1.99% |
| 20% | mean | m1, m2, m6, m10, cv3.2, dfl | 0.65 | 0.98/0.00/−0.75 | 3.16% |
| 20% | minimax | m0, m2, m6, m7, m9, m11, m14, cv3.2, dfl | 1.23 | 0.11/0.24/0.08 | 0.40% |

### 3.4. Đo thật các allocation

Mỗi allocation ở trên được build thành **một** engine (mọi block được chọn restore cùng lúc), rồi chạy trên tune2000 và trên val2017. **val2017 không được dùng để chọn block hay budget.** File: `isci_method_20261025/results/yolo11s_int8max/` (có hash prediction; file `.pred.json.gz` nằm trên server).

**val2017 held-out.** FP32 = 46.02 / 28.46 / 50.53 / 62.99. "Worst %" là tổn thất tương đối lớn nhất trong ba nhóm.

| cấu hình | %MAC FP16 | AP | AP_S | AP_M | AP_L | worst % |
|---|---|---|---|---|---|---|
| INT8 max (uniform) | 0 | 43.08 | 25.05 | 47.35 | 59.30 | 12.0 |
| INT8 entropy (uniform) | 0 | 44.34 | 27.07 | 48.46 | 60.71 | 4.9 |
| **INT8 entropy + conv đầu ra head FP16** (baseline rẻ) | **1.1** | **44.88** | 27.77 | 49.20 | 61.26 | **2.75** |
| INT8 entropy + cả head FP16 | 14.9 | 44.96 | 27.63 | 49.47 | 61.64 | 2.92 |
| mean @10% | 9.8 | **44.64** | 27.03 | 49.05 | 61.20 | 5.03 |
| **minimax @10%** | 9.4 | 44.57 | **27.65** | 48.95 | 61.42 | **3.12** |
| mean @15% | 14.9 | **44.87** | 26.84 | 49.20 | 61.66 | 5.70 |
| **minimax @15%** | 14.1 | 44.66 | **27.57** | 49.54 | 61.87 | **3.13** |
| mean @20% | 19.7 | **45.13** | 27.45 | 49.74 | 62.34 | 3.54 |
| **minimax @20%** | 19.3 | 44.72 | **28.01** | 49.52 | 61.67 | **2.09** |

> **Kết luận thẳng: hiện tại method CHƯA thắng baseline rẻ.**
> - "Entropy + conv đầu ra head FP16" chỉ tốn 1.1% MAC, nhưng đã đạt AP 44.88 và worst 2.75%. Mức này **tốt hơn minimax @10% và @15% ở cả AP lẫn worst-group**.
> - Minimax @20% chỉ thắng về worst-group (2.09 so với 2.75), và phải trả 0.16 AP cùng gấp khoảng 17 lần lượng MAC chạy FP16.
> - Lý do: so sánh ở trên dùng base **int8max** (yếu), trong khi baseline rẻ dùng **entropy**.
> - Gate thật sự là G2: chạy method trên base entropy, có ép sẵn conv đầu ra head vào FP16, rồi so ở cùng MAC. **Nếu G2 thua, nên giữ bài ở dạng phân tích và chỉ bổ sung baseline, không nên làm bài method.**

So sánh hàm mục tiêu (ablation trên base int8max, chưa phải so với baseline mạnh):
- Ở cả 3 budget, trên val, minimax giảm worst-group relative loss so với mean: từ 5.0 xuống 3.1, từ 5.7 xuống 3.1, và từ 3.5 xuống 2.1. AP_small tăng thêm 0.56 đến 0.73.
- Đổi lại, AP chung giảm 0.07 đến 0.41. Đây là trade-off theo đúng thiết kế, không phải thắng ở mọi chỉ số.
- Mọi allocation minimax đều chứa **m0 (stem)**. Kinh nghiệm phổ biến là "giữ lớp đầu và lớp cuối ở precision cao", nên reviewer sẽ hỏi MILP có phải chỉ đang tìm lại đúng mẹo đó không. Baseline "stem + conv đầu ra head FP16" đã được xếp hàng chạy sau G2.
- **Paired bootstrap trên val** (500 resample theo ảnh, `bootstrap_pair.py`; point estimate tái lập đúng từng chữ số). Bảng là hiệu minimax − mean:

  | budget | ΔAP [CI95] | ΔAP_S [CI95] | Δworst (điểm %) [CI95] |
  |---|---|---|---|
  | 10% | −0.06 [−0.30, +0.18] | +0.31 [−0.59, +1.24] | +0.77 [−2.0, +3.4] |
  | 15% | −0.22 [−0.46, +0.01] | +0.34 [−0.55, +1.14] | +1.14 [−1.7, +3.9] |
  | 20% | **−0.42 [−0.67, −0.16]** | +0.20 [−0.80, +1.14] | +0.29 [−2.8, +3.4] |

  **Ưu thế về worst-group và AP_small của minimax trên yolo11s CHƯA vượt mức nhiễu**, trong khi phần mất AP ở budget 20% thì có ý nghĩa. Trung bình bootstrap của ΔAP_S chỉ bằng khoảng nửa point estimate: AP_small của COCO tính trung bình trên các lớp, nhiều lớp có rất ít object nhỏ, nên chỉ số này dao động mạnh.
- Cách giải thích hợp lý: dưới entropy INT8, yolo11s gần như không có chênh lệch giữa các nhóm (4.9 / 4.1 / 3.6%), nên minimax không có gì để sửa. **Phép thử đúng phải làm trên model có chênh lệch lớn** như YOLO26 (yolo26s entropy: 16.4 / 7.9 / 6.4%). Ngoài ra cần một chỉ số nhóm ổn định hơn AP_small, ví dụ AP trung bình theo lớp trên ít nhất 5 lớp, hoặc nhóm theo supercategory hay theo miền.

**Tune2000: dự đoán cộng tính so với đo thật.**

| cấu hình | gap AP dự đoán / đo | worst % dự đoán / đo |
|---|---|---|
| mean @10% | 1.24 / 1.29 | 3.48 / 4.41 |
| minimax @10% | 1.44 / 1.30 | 1.99 / 3.20 |
| mean @15% | 0.93 / 1.04 | 2.94 / 4.59 |
| minimax @15% | 1.54 / 1.25 | 1.28 / 3.85 |
| mean @20% | 0.65 / 0.99 | 3.16 / 2.46 |
| minimax @20% | 1.23 / 1.04 | 0.40 / 2.69 |

- Giả định cộng tính dự đoán **gap AP chung khá tốt** (sai ≤ 0.34). Nhưng nó **lạc quan quá mức với gap từng nhóm**, và sai số tăng theo số block được chọn: ở minimax @20%, dự đoán 0.40% nhưng đo được 2.69%.
- Ở @20%, đo trên tune thì minimax **không** hơn mean (2.69 so với 2.46), dù trên val minimax vẫn hơn. Nghĩa là bộ chọn hiện tại đang **thắng nhờ hướng đi đúng** (ưu tiên các block độ phân giải cao như m0, m9), **chưa phải nhờ dự đoán chính xác**.
- Đây chính là chỗ cần cải tiến method (§3.5), và cũng là chỗ có nội dung cho phần lý thuyết: cận sai số do tương tác giữa các block.

### 3.4b. G2: base entropy, conv đầu ra head luôn FP16 (yolo11s, val2017 held-out)

File: `isci_method_20261025/results/yolo11s_int8ent/`.

| cấu hình | %MAC FP16 | AP | AP_S | AP_M | AP_L | worst % |
|---|---|---|---|---|---|---|
| entropy + head-out (baseline rẻ) | 1.1 | 44.88 | 27.77 | 49.20 | 61.25 | 2.76 |
| entropy + stem + head-out (mẹo "lớp đầu/cuối") | 1.9 | 44.94 | 27.61 | 49.55 | 61.21 | 2.99 |
| mean @5% | 4.5 | 45.01 | 27.49 | 49.63 | 61.61 | 3.42 |
| minimax @5% | 3.6 | 44.99 | 27.46 | 49.58 | 61.54 | 3.51 |
| **mean @10%** | 9.7 | **45.59** | **28.40** | **50.11** | 62.65 | **0.83** |
| minimax @10% | 9.7 | 45.57 | 28.24 | 50.01 | 62.65 | 1.03 |
| **mean @15%** | 14.1 | **45.78** | 28.58 | 50.17 | 62.98 | 0.71 |
| minimax @15% | 14.0 | 45.46 | 28.23 | 49.94 | 62.38 | 1.17 |
| **mean @20%** | 19.2 | **45.79** | 28.83 | 50.35 | 62.91 | **0.36** |
| minimax @20% | 19.5 | 45.45 | 28.15 | 50.00 | 62.29 | 1.10 |
| FP32 | 100 | 46.02 | 28.46 | 50.53 | 62.99 | 0 |

**Kết luận G2 trên yolo11s:**
1. **Khung chọn precision dựa trên engine thắng baseline rẻ từ budget 10% trở lên.** Mean @10% thêm +0.71 AP so với head-out, và worst giảm từ 2.76 xuống 0.83%. Mean @20% gần như về lại FP32 (thiếu 0.23 AP). Mẹo stem + head-out gần như không giúp gì (+0.06 AP).
2. **Minimax không hơn mean** ở mọi budget. Khi tổng gap đã nhỏ thì mọi nhóm cùng nhỏ, nên chọn theo mean đủ để giảm worst-group. Ở budget 15–20%, minimax còn tệ hơn vì bám vào các block có độ nhạy nhiễu.
3. Vì vậy, với yolo11s, phần mới (group-robust) **không có tác dụng**. Phần có tác dụng (chọn block theo độ nhạy đo trên engine) lại là ý tưởng mixed-precision quen thuộc. Phần này phải thắng ModelOpt `auto_quantize` ở cùng chi phí thì mới thành đóng góp.
4. Còn lại một giả thuyết, sẽ kiểm tra trên **yolo26s** (sweep đang chạy). Minimax chỉ có ích khi chênh lệch giữa các nhóm lớn: yolo26s entropy có tổn thất 16.4 / 7.9 / 6.4%. Nếu đúng, bài có được câu chuyện có cơ sở: *minimax trùng với mean khi chênh lệch nhỏ và khác mean khi chênh lệch lớn*, cộng thêm điều kiện lý thuyết (đóng góp 3c). Nếu sai, đóng góp group-robust phải bỏ.

### 3.4c. G4: so với ModelOpt `auto_quantize` (yolo11s, base entropy, val2017 held-out)

Cả hai bên dùng cùng graph Q/DQ entropy, cùng engine TensorRT, cùng tập tune 512–2000 ảnh train, cùng head-out bắt buộc. Chỉ khác **tiêu chí chọn**:
- **engine (của mình):** độ nhạy đo trên engine TensorRT thật, sau đó giải MILP mean.
- **Fisher:** score gradient/Fisher của `auto_quantize` (loss train của YOLO, 512 ảnh tune), giải cùng bài toán MILP theo MAC.
- **auto_quantize gốc:** ràng buộc effective-bits theo số trọng số, không ép head-out.

| % MAC FP16 | engine mean | Fisher | auto_quantize gốc (% MAC thật) |
|---|---|---|---|
| ~5 | 45.01 | 45.04 | 44.57 (6.2) |
| ~10 | **45.59** | 45.15 | 44.70 (10.3) |
| ~15 | **45.78** | 45.25 | 44.69 (12.9), 44.74 (17.9) |
| ~20 | **45.79** | 45.22 | 44.75 (21.7) |

Tham chiếu: int8ent 44.34, head-out 1.1% MAC 44.88, FP32 46.02.

Đọc kết quả:
1. Từ 10% MAC trở lên, chọn theo engine hơn Fisher khoảng +0.45 đến +0.57 AP. Ở 5% thì hoà. Khoảng tin cậy bootstrap đang chạy.
2. `auto_quantize` gốc còn **thua head-out 1.1% MAC**. Lý do: cost tính theo số trọng số nên nó ưu tiên giữ các conv đầu, vốn tốn nhiều MAC mà ít trọng số. Nó cũng không biết head-out là phần nhạy nhất trên engine.
3. Fisher bão hoà quanh 45.2 sau 10%. Engine tiếp tục tăng tới 45.8 và chỉ còn cách FP32 0.23 AP.
4. Hệ quả cho bài: đóng góp *đo độ nhạy trên engine thật* có bằng chứng thắng baseline công nghiệp ở cùng chi phí. Cần lặp lại trên yolo26s (đang chạy) và ít nhất một model m/l.

### 3.4d. G2 trên yolo26s: mean so với minimax (base entropy, head-out bắt buộc 0.84% MAC, val2017 held-out)

FP32: AP 47.71, S/M/L 30.24 / 52.35 / 64.22. Worst-group = tổn thất tương đối lớn nhất trong S/M/L so với FP32.

| cấu hình | % MAC | AP | AP_S | AP_M | AP_L | worst-group % | dự đoán worst (tune) |
|---|---|---|---|---|---|---|---|
| int8ent | 0 | 44.03 | 25.28 | 48.21 | 60.12 | 16.40 | |
| head-out | 0.84 | 44.84 | 26.83 | 48.81 | 60.12 | 11.28 | |
| stem + head-out | 1.7 | 44.82 | 26.36 | 48.63 | 60.82 | 12.83 | |
| cả head FP16 | 8.9 | 46.47 | 28.25 | 50.89 | 62.46 | 6.58 | |
| mean@3% | 2.8 | 45.96 | 26.93 | 50.38 | 62.62 | 10.95 | 10.85 |
| minimax@3% | 2.8 | 45.89 | 26.85 | 50.26 | 62.44 | 11.21 | 10.67 |
| mean@5% | 4.1 | **46.10** | 27.03 | 50.72 | 62.68 | 10.62 | 10.30 |
| minimax@5% | 5.0 | 45.95 | 27.82 | 50.42 | 61.85 | **8.00** | 8.08 |
| mean@10% | 9.7 | 46.54 | 28.33 | 51.04 | 62.69 | 6.32 | 5.55 |
| minimax@10% | 9.7 | 46.48 | 28.19 | 50.74 | 62.45 | 6.78 | 5.38 |
| mean@15% | 13.6 | 46.55 | 28.34 | 50.87 | 62.51 | 6.28 | 6.06 |
| minimax@15% | 14.8 | **46.69** | 28.76 | 51.03 | 63.10 | **4.89** | 4.52 |
| mean@20% (= minimax) | 19.9 | 47.03 | 28.56 | 51.70 | 63.29 | 5.56 | 3.58 |

Đọc kết quả (chưa có bootstrap, đang chạy):
1. **Minimax có tác dụng ở đúng những chỗ solver dự đoán nó có tác dụng.** Ở 5% và 15%, solver dự đoán worst-group giảm 2.2 và 1.5 điểm; trên val giảm thật 2.6 và 1.4 điểm. Ở 15%, AP còn tăng 0.14; ở 5%, AP giảm 0.15. Ở 3% và 10%, solver chỉ dự đoán chênh 0.2 điểm, và trên val minimax hơi tệ hơn (nằm trong nhiễu). So với yolo11s (chênh lệch giữa các nhóm nhỏ, minimax không bao giờ giúp), điều này khớp giả thuyết *minimax chỉ có ích khi chênh lệch giữa các nhóm lớn*.
2. **Baseline "cả head FP16" rất mạnh trên yolo26s.** Ở 8.9% MAC nó đạt 46.47, gần bằng mean@10% (46.54 ở 9.7%). Tức là trên yolo26s, ở ngân sách khoảng 10%, cách chọn theo engine chỉ tìm lại đúng "giữ cả head". Lợi thế chỉ rõ ở 4–5% (46.10 ở 4.1% so với head-out 44.84) và ở 15–20%. Trên yolo11s thì khác: cả head tốn 14.9% MAC chỉ được 44.96, trong khi mean@15% đạt 45.78.
3. Kết luận tạm thời cho gate G2: **chưa bác bỏ** phần group-robust, nhưng bằng chứng mới ở 2/4 budget trên một model. Cần bootstrap, và cần thêm ít nhất một model có chênh lệch giữa các nhóm lớn (yolo26l: 12.6 / 8.7 / 7.9%) trước khi giữ nó là đóng góp chính.

### 3.5. Cải tiến method từ kết quả trên (đề xuất)

1. **Re-linearization lặp, có engine trong vòng lặp** (successive MILP / trust region).
   - Sau khi giải, build cấu hình được chọn S_t, rồi đo lại độ nhạy *biên* của các block ứng viên ngay trong ngữ cảnh S_t (thêm hoặc bớt một block). Sau đó giải lại, giới hạn số block được đổi mỗi vòng.
   - Mỗi vòng tốn khoảng 10–20 lần build. Cách này xử lý trực tiếp lỗi do tương tác, đổi lại số engine phải build tăng.
2. **Scenario-robust:** minimax đồng thời trên nhóm và trên B bootstrap resample của tập tuning (prediction đã được lưu). Mục đích là tránh việc chọn các block mà độ nhạy chỉ có được nhờ nhiễu.
3. **Tập tuning lớn hơn** (5000 ảnh train) nếu (1) và (2) chưa đủ. Chỉ tốn thời gian chạy, không tốn nhãn.

---

## 4. Rủi ro (nói thẳng)

1. **Nhiễu ước lượng.** Phần lớn độ nhạy theo nhóm có độ lớn ±0.2–0.4 AP. Minimax thường bám vào chính các block nhiễu này (ví dụ m0: AP −0.11 nhưng S +0.26).
   - Nếu kết quả đo thật trên val không giữ được ưu thế, phải chuyển sang phiên bản scenario-robust (bootstrap) hoặc dùng tập tuning lớn hơn.
   - Từ sweep int8ent, em đã lưu prediction để bootstrap được.
2. **Baseline mạnh.** Riêng việc đổi calibration sang entropy đã lấy lại 1.26 AP, và để thêm head ở FP16 lên tới 44.96. Method phải thắng *entropy + selective FP16* ở cùng ngân sách, và thắng ModelOpt `auto_quantize`. Nếu không thắng, em sẽ báo đúng như vậy.
3. **Độ trễ trên RTX 5090.**
   - FP8 nhanh hơn INT8 ở 9/9 cấu hình, nên trên Blackwell chạy toàn bộ FP8 là lựa chọn tốt hơn.
   - Với yolo11s, FP16 (0.47 ms) còn nhanh hơn INT8 (0.54 ms).
   - Do đó method phải đặt vào bối cảnh **phần cứng không có FP8** (Orin, Ampere, edge) và model **m/l/x**, nơi INT8 thực sự nhanh hơn FP16. Số độ trễ chính phải đo trên Orin. Không được claim độ trễ từ fake-quant, và không gọi FP8 là trung tính về kích thước.
4. **Trùng ý tưởng (prior art).**

   | công trình | điểm trùng | điểm khác |
   |---|---|---|
   | MLFQ, *Mask-Guided Local Feature Quantization for Small Object Detection* (IET Image Processing 2026) | Quan tâm small object trong PTQ | Họ thêm trọng số không gian vào reconstruction loss, chỉ có lợi ở W4A4/W3A3. Mình chọn precision theo minimax nhóm ở W8 trên engine thật. |
   | FairQuant (arXiv 2602.23192) | Group-conditioned importance và phân bổ bit theo ngân sách | Bài cho phân loại ảnh y tế. Mình làm detection, nhóm theo kích thước/miền, đo trên engine, dùng MILP chính xác. |
   | FLIQS, Flexible 8-bit formats (2310.13513) | Tìm format INT/FP theo layer | Mục tiêu trung bình, không có nhóm |
   | HAWQ-v3, Qualcomm 2302.05397, ModelOpt `auto_quantize` | Phân bổ mixed-precision | Mục tiêu trung bình, nên làm baseline |
   | Q-YOLO (ACPR 2023), Reg-PTQ (CVPR 2024), MSQuant (2025), QRT-DETR (2026), DetPTQ (2304.09785) | PTQ cho detector | Calibration/reconstruction. Có thể chạy kết hợp với method (trực giao). |

5. **Phạm vi INS.** INS cần tính tổng quát và có lý thuyết. Một bài chỉ về YOLO trên COCO dễ bị coi là ứng dụng hẹp. Vì vậy cần **ít nhất hai kiểu nhóm** (kích thước, và một kiểu khác như ngày/đêm trên BDD100K hoặc ExDark, nối với hướng nighttime của anh) và phần lý thuyết ở §2.3.

---

## 5. Baseline bắt buộc (trả lời lý do (1) của editor)

- **Uniform:** INT8 max, entropy, percentile; FP8 làm tham chiếu (oracle, không phải đối thủ cùng phần cứng).
- **Mixed-precision:**
  - ModelOpt `mtq.auto_quantize`, dùng `loss_func` và `constraints` đúng ngân sách.
  - HAWQ-style (Hessian trace).
  - Qualcomm SQNR-style.
  - Chọn theo độ nhạy trung bình. Đây là ablation chính: cùng pipeline, chỉ khác hàm mục tiêu.
  - Greedy, và chọn ngẫu nhiên ở cùng chi phí.
- **PTQ reconstruction:** một phương pháp có code chính thức (QDrop hoặc BRECQ), ghi rõ là fake-quant. Thêm phiên bản kết hợp "reconstruction + allocation".
- **Bảng so sánh với số đã công bố (2022–2026):** Q-YOLO, Reg-PTQ, MSQuant, MLFQ, QRT-DETR, QRID. Ghi rõ khác setting ở đâu.

---

## 6. Giao thức chống rò rỉ dữ liệu

- Calibration: 512 ảnh train (đã có). Tuning: 2000 ảnh train khác, seed 20261025, có manifest và hash (`isci_method_20261025/make_tuning_split.py`).
- Chọn block, budget và mọi siêu tham số chỉ dựa trên tuning. val2017 chỉ dùng để báo cáo, mỗi cấu hình chạy đúng một lần.
- FP32 trên tập tuning bị thổi phồng (AP 54.56) vì model đã train trên chính các ảnh này. Gap được tính so với FP16 trên cùng tập nên phần thiên lệch này phần lớn tự triệt tiêu. Phương án dự phòng là dùng prediction FP32 làm pseudo-GT, không cần nhãn.
- Mỗi run có ID mới, không ghi đè run cũ, có hash prediction, và kiểm định paired bootstrap trên val.

---

## 7. Kế hoạch và điểm quyết định

| bước | việc | chi phí | gate |
|---|---|---|---|
| G1 | Ablation hàm mục tiêu (minimax so với mean) trên base int8max, yolo11s | **xong** | Minimax giảm worst-group trên val ở 3/3 budget: đạt (chờ bootstrap). Cộng tính theo nhóm: **không đạt**. So với baseline rẻ (entropy + head-out): **thua ở budget ≤15%**. |
| G2 | Sweep int8ent cho yolo11s (**đang chạy**), sau đó baseline stem + head-out. Rồi sweep yolo26s/m | khoảng 1.5 giờ mỗi model | **Gate quyết định:** minimax trên base entropy, có ép head-out vào FP16, phải thắng "entropy + head-out" và "stem + head-out" ở cùng MAC |
| G3 | Scenario-robust (bootstrap) nếu G1 nhiễu | CPU | |
| G4 | Baseline `auto_quantize` (**đang chạy**, yolo11s) và HAWQ trên 2 model | khoảng nửa ngày | Method không thua ở cùng chi phí |
| G5 | Grid chính: YOLO11/v8/YOLO26 m/l/x, COCO và ExDark; độ trễ Orin | khoảng 2–3 ngày | Anh đã duyệt; chỉ chạy sau khi G2 và G4 xong |
| G6 | Viết lại bài (bảng tài liệu 2022–2026, đoạn khác biệt so với bài cũ của tác giả, lý thuyết) | | |

**Anh đã quyết (25-10):**
1. Đồng ý đổi sang bài method. Em hiểu là **có điều kiện**: phần group-robust chỉ được giữ nếu qua gate G2 trên yolo26s. Nếu không qua, bài vẫn là bài method nhưng đóng góp chính là *phân bổ độ chính xác dựa trên độ nhạy đo trên engine thật*, và phải thắng `auto_quantize` (G4).
2. Bộ thứ hai do em chọn: **ExDark** (§8).
3. Được dùng Orin để đo độ trễ chính.

---

## 8. Bộ dữ liệu thứ hai: ExDark (đã chọn)

**Vì sao chọn ExDark:**
- 12 lớp của ExDark đều có trong COCO, nên model COCO-pretrained chạy zero-shot, không phải train lại. Calibration và tuning vẫn chỉ dùng ảnh train.
- Mỗi ảnh có nhãn **kiểu chiếu sáng** (10 loại) và indoor/outdoor. Đây là trục nhóm thứ hai, khác hẳn kích thước: nhóm theo điều kiện chụp, không theo object. Nó cũng nối với hướng nighttime của anh mà không đụng các bài CNTSSS về biển báo ban đêm.
- Lý do loại các bộ khác: BDD100K và VisDrone đều phải train lại, mà VisDrone lại là sân nhà của MLFQ.
- Giấy phép: dùng cho nghiên cứu học thuật phi thương mại thì được. Nếu có tài trợ từ công ty vì lợi nhuận thì phải xin phép bằng văn bản. **Anh kiểm tra giúp em mục funding của bài.**

**Dữ liệu** (`make_exdark_coco.py`, split chính thức, không bỏ ảnh nào):

| split | ảnh | object | S / M / L |
|---|---|---|---|
| train | 3000 | 10381 | 712 / 4163 / 5506 |
| val | 1800 | 6212 | |
| test | 2563 | 7117 | 264 / 1981 / 4872 |

**yolo11s FP32 zero-shot trên test:** AP 34.73. Theo kiểu chiếu sáng:

| Ambient | Low | Object | Screen | Shadow | Single | Strong | Twilight | Weak | Window |
|---|---|---|---|---|---|---|---|---|---|
| 41.07 | 30.49 | 37.30 | 38.56 | 38.86 | 35.63 | 30.11 | 31.36 | 29.75 | 37.39 |

Indoor 38.61, outdoor 31.61.

**Định nghĩa nhóm, chốt trước khi có bất kỳ kết quả lượng tử nào trên ExDark:**
- Mục tiêu nhóm dùng **8 kiểu chiếu sáng có ít nhất 100 ảnh test**. Screen (28 ảnh) và Shadow (14 ảnh) chỉ tính vào AP tổng.
- Không dùng nhóm S/M/L trên ExDark làm mục tiêu: test chỉ có 264 object nhỏ, quá nhiễu.
- Độ nhạy và lựa chọn chỉ đo trên ExDark train (3000 ảnh). ExDark val dùng để chọn siêu tham số nếu cần; test chỉ dùng để báo cáo, mỗi cấu hình chạy đúng một lần.
- Box area thay cho mask area, sẽ ghi rõ trong bài.

**Trục nhóm phụ không tốn thêm chi phí:** 12 supercategory của COCO (`group_eval.py`). Trên yolo11s @10%, tổn thất khá đều giữa các nhóm.

---

## 9. Trạng thái baseline và lỗi đã sửa (25-10)

- **ModelOpt `auto_quantize`:** bản đầu dùng loss MSE so với output FP32 nên mọi score đều bằng 0. Lý do: auto_quantize lấy gradient tại điểm chưa lượng tử, mà ở đó loss này và gradient của nó bằng đúng 0. Em đã sửa sang dùng loss train của ultralytics (box + cls + dfl) với nhãn GT của tập tune; tập này là ảnh train2017 nên không rò rỉ. Có hai biến thể:
  - (a) `auto_quantize` gốc: cost tính theo số trọng số; ghi lại phần trăm MAC tương ứng.
  - (b) Fisher score của `auto_quantize` giải cùng bài toán knapsack theo MAC như method, có ép head-out. Biến thể này cô lập đúng một khác biệt là *tiêu chí độ nhạy* (gradient/Fisher trong torch so với đo trên engine TensorRT thật).
  - Smoke test 32 ảnh: 8.4 bit giữ 28 conv ở FP16, tương đương 28% MAC. Lý do là các lớp đầu có ít trọng số nhưng nhiều MAC. Đang chạy bản đủ 512 ảnh.
- **yolo26s, các dòng head-out trong bảng gate trước đây không hợp lệ:** regex không khớp nhánh `one2one_` của head YOLO26, nên các dòng đó thực chất giống hệt base. Em đã sửa regex (`configs/manual_yolo26s_*.json`) và đo lại.
- `eval_alloc.py` bị lỗi khi config thủ công không có `cost`. Đã sửa: tự tính phần trăm MAC từ regex.

---

## 10. Giao thức confirmatory (chốt 25-10-2026 22:00, TRƯỚC khi chạy yolo26l / yolo11l và ExDark-allocation)

yolo11s và yolo26s là **exploratory**: em đã nhìn val của chúng khi sửa method. Các model mới (yolo26l, yolo11l) và ExDark-allocation là **confirmatory**: chạy đúng một lần theo giao thức dưới đây, không chỉnh method sau khi xem kết quả.

1. **Base:** INT8 entropy, ModelOpt 0.45, 512 ảnh calibration train2017, Sigmoid giữ FP16. TensorRT 11.1 strongly-typed trên RTX 5090.
2. **Tập bắt buộc FP16 (head-out):** các conv cuối của mỗi nhánh box/cls trong head và DFL (regex trong `configs/manual_*`). Chi phí MAC của tập này được trừ khỏi mọi budget.
3. **Budget:** 3 / 5 / 10 / 15 / 20 % MAC conv.
4. **Phương pháp so sánh:**
   - engine-mean
   - engine-minimax (S/M/L trên COCO; 8 kiểu chiếu sáng trên ExDark)
   - mean ép tiêu cùng MAC với minimax (`mean_matched`)
   - Fisher (`auto_quantize`, **2000** ảnh tune, cùng knapsack MAC, cùng head-out)
   - `auto_quantize` gốc (lưới effective-bits 8.02–8.3)
   - random cùng MAC (5 seed, budget 5 / 10 / 20)
   - head-out, stem + head-out, cả head FP16
5. **So sánh minimax và mean phải ở cùng MAC thật:** báo cáo đường Pareto AP–MAC và worst-group–MAC, cộng `mean_matched`.
6. **Thống kê:** paired bootstrap theo ảnh, 1000 lần lặp, seed 20261025. Khác biệt chỉ được gọi là có ý nghĩa khi CI95 không chứa 0.
7. **Tập báo cáo:** COCO val2017; ExDark test (2563 ảnh). ExDark val không dùng.
8. **Quy tắc framing (chốt trước):**
   - minimax chỉ là đóng góp đồng hạng nếu nó giảm worst-group ở cùng MAC, có CI95 không chứa 0, trên **≥ 2 model có chênh lệch nhóm lớn** (yolo26s, yolo26l), hoặc trên ExDark.
   - Nếu không, bài dẫn bằng *phân bổ dựa trên độ nhạy đo trên engine* so với Fisher/`auto_quantize`, còn minimax thành một mục phân tích ("hàm mục tiêu chỉ quan trọng khi chênh lệch lớn").
9. **Chi phí tìm kiếm:** ghi wall-clock của mỗi phương pháp (sweep engine so với Fisher), báo cáo trung thực.
10. **Export model mới:** tải trọng số chính thức từ Ultralytics, ghi sha256. ONNX opset 17, simplify, batch 1, static, nms False, ghi version torch.

## Phụ lục: code mới (`isci_method_20261025/`, có mirror trên server)

- `qdq_restore.py`: restore các node khớp regex về FP16, giữ nguyên Q/DQ ở các node khác (có test: `tests/test_qdq_restore.py`, 3 test pass).
- `make_tuning_split.py`: tạo tập tuning 2000 ảnh train.
- `infer_eval.py`: chạy engine và tính COCO AP trên GT bất kỳ, có thể lưu prediction.
- `block_sensitivity.py`: đo độ nhạy từng block trên engine thật, kèm độ trễ.
- `allocate.py`: giải MILP mean/minimax theo ngân sách MAC.
- `eval_alloc.py`: build và đo thật các allocation trên tune và val.
- `gate_headroom.py`: đo dư địa ban đầu của các recipe uniform.
- `bootstrap_pair.py`: paired bootstrap theo ảnh, tính lại COCOeval trên mẫu lặp.
- `group_eval.py`: AP theo nhóm: size, supercategory, thuộc tính ảnh (`attr:light`).
- `make_exdark_coco.py`: chuyển ExDark sang GT dạng COCO theo split chính thức.
- `autoquant_select.py`: baseline ModelOpt `auto_quantize` (bản gốc và bản Fisher-knapsack theo MAC).
- `configs/`: các cấu hình thủ công (head-out, stem + head-out) cho yolo11s/yolo26s.
