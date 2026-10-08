# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** 1 người **Thành viên:** Bùi Đăng Khoa - 2A202602617

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

## 1. Cấu hình đã chọn

Mỗi video: tracker bạn nộp, `conf`, `iou`, điều bạn **nhìn thấy** trên video, và một cấu hình đã thử rồi loại.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.3 | 0.5 | Quảng trường ngoài trời, camera tĩnh, ban ngày. Người đi bộ đan xen nhau; BoTSORT kết hợp Re-ID và GMC giúp giữ ID rất tốt khi hai người cắt nhau, ít bị nhảy ID. | bytetrack (conf 0.3, iou 0.5): HOTA đạt 26.912 và IDF1 đạt 25.713, thấp hơn BoTSORT (HOTA 29.460, IDF1 29.354). |
| video_2 (phố đêm, tĩnh, rất đông) | bytetrack | 0.25 | 0.5 | Camera tĩnh trên cao, ban đêm, mật độ người cực kỳ đông và che khuất liên tục. ByteTrack 2-stage association liên kết tốt các detection bị che khuất một phần (conf 0.25), không bị phụ thuộc vào Re-ID khi màu sắc quần áo trong đêm bị tối và bão hòa. | botsort (conf 0.3, iou 0.5): Ánh sáng ban đêm yếu khiến vector Re-ID bị nhiễu, dễ gán nhầm ID cho người gần cạnh và FPS xử lý giảm mạnh. |
| video_3 (camera di động, ảnh nhỏ) | botsort | 0.3 | 0.5 | Camera di chuyển, độ phân giải thấp, khung hình chậm khiến chuyển động của vật thể trên ảnh bị giật. BoTSORT có bù chuyển động camera (GMC) kết hợp Re-ID giúp duy trì liên kết track tốt hơn hẳn chuyển động thuần. | ocsort (conf 0.3, iou 0.5): Do FPS thấp và camera di chuyển, thiếu GMC và Re-ID khiến quỹ đạo bị đứt đoạn, sinh ra nhiều ID mới. |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.35 | 0.5 | Trong nhà, camera tiến về phía trước, nhiều bề mặt kính phản chiếu bóng người. Đặt conf=0.35 giúp lọc triệt để các hộp bóng ảo; Re-ID của BoTSORT giữ vững ID khi kích thước người phóng to dần lúc lại gần camera. | bytetrack (conf 0.25, iou 0.5): Ngưỡng conf thấp bắt phải nhiều bóng phản chiếu trên cửa kính, tạo ra các track ảo (false positive tracks). |
| video_5 (trên xe bus, giao lộ đông) | botsort | 0.3 | 0.5 | Góc quay từ xe bus bị rung lắc mạnh, giao lộ đông đúc hỗn loạn. BoTSORT triệt tiêu rung lắc camera nhờ GMC và dùng Re-ID để nhận lại đúng đối tượng sau các cú xóc nảy. | bytetrack (conf 0.3, iou 0.5): Rung giật mạnh làm khoảng cách giữa 2 frame liên tiếp nhảy vọt, mô hình vận tốc tuyến tính của ByteTrack bị trượt và cấp phát ID mới. |

## 2. Số liệu video_1

Dán bảng HOTA / MOTA / IDF1 do `scripts/evaluate_practice.py` in ra.

```
HOTA: nop_bai_video1-pedestrian    HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)
video_1                            29.46     18.095    48.223    18.596    78.888    51.153    82.992    83.666    29.899    35.813    78.45     28.095    
COMBINED                           29.46     18.095    48.223    18.596    78.888    51.153    82.992    83.666    29.899    35.813    78.45     28.095    

CLEAR: nop_bai_video1-pedestrian   MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag      
video_1                            19.811    81.471    19.945    21.759    92.306    12.903    19.355    67.742    15.779    4043      14538     337       25        8         12        42        99        
COMBINED                           19.811    81.471    19.945    21.759    92.306    12.903    19.355    67.742    15.779    4043      14538     337       25        8         12        42        99        

Identity: nop_bai_video1-pedestrianIDF1      IDR       IDP       IDTP      IDFN      IDFP      
video_1                            29.354    18.137    76.941    3370      15211     1010      
COMBINED                           29.354    18.137    76.941    3370      15211     1010      

Count: nop_bai_video1-pedestrian   Dets      GT_Dets   IDs       GT_IDs    
video_1                            4380      18581     52        62        
COMBINED                           4380      18581     52        62        
```

*So sánh với cấu hình ByteTrack (conf=0.3, iou=0.5): HOTA: 26.912, MOTA: 17.292, IDF1: 25.713. Cấu hình BoTSORT vượt trội hơn trên cả 3 thước đo chính.*

`video_2` đến `video_5` không có nhãn trong gói lab. Không điền số cho các video đó.

## 3. Phân tích

### Phân tích Video 1 (Quảng trường ngoài trời, camera tĩnh, ban ngày):
BoTSORT duy trì ID liên tục và chính xác hơn ByteTrack khi có sự giao cắt giữa các đối tượng. Trong cảnh ban ngày, ánh sáng rõ ràng và độ tương phản cao giúp mô hình Re-ID (`osnet_x0_25_msmt17`) trích xuất các vector đặc trưng ngoại hình rất tách bạch giữa những người đi bộ khác nhau. Nhờ vậy, ngay cả khi người đi bộ bị che khuất một phần trong vài frame lúc đi ngang qua nhau, Re-ID đã hỗ trợ hiệp hội (association) chính xác mà không bị hoán đổi danh tính (ID switch). Điểm HOTA tăng từ 26.912 lên 29.460 và IDF1 tăng từ 25.713 lên 29.354 minh chứng rõ nét cho vai trò của Re-ID trong điều kiện thị giác thuận lợi.

### Phân tích Video 2 (Phố đêm, camera tĩnh trên cao, mật độ rất đông):
Khác với Video 1, ở Video 2 ByteTrack lại là lựa chọn tối ưu hơn. Do quay ban đêm từ trên cao, phần lớn người đi bộ xuất hiện với kích thước nhỏ, tối màu và bị nhiễu do thiếu sáng trầm trọng, khiến đặc trưng Re-ID mất đi tính phân biệt và dễ dẫn đến gán nhầm ID giữa các đối tượng ở gần nhau. Ngược lại, ByteTrack dựa trên chuyển động thuần và chiến lược phân tầng hai ngưỡng (low/high confidence detection) đã tận dụng tối đa các phát hiện mờ nhòe ở ngưỡng thấp (`conf=0.25`) để nối lại quỹ đạo người bị che khuất mà không cần Re-ID. Hơn nữa, với hơn 1000 frame và mật độ người dày đặc, ByteTrack đạt tốc độ gần 14 FPS (nhanh gấp hơn 2 lần BoTSORT), giảm thiểu gánh nặng tính toán mà vẫn theo dõi mượt mà.

### Phân tích Video 3 (Camera di động, ảnh nhỏ, khung hình chậm):
Sự kết hợp giữa camera di động và tốc độ khung hình chậm đặt ra thách thức cực lớn cho các tracker dựa trên chuyển động giả định tuyến tính (như Kalman Filter chuẩn của ByteTrack hay OCSORT). Chuyển động của camera làm dịch chuyển toàn bộ nền ảnh, khiến vị trí dự đoán bị sai lệch nghiêm trọng giữa các frame cách xa nhau. BoTSORT giải quyết triệt để vấn đề này nhờ tích hợp thuật toán bù chuyển động camera (GMC - Global Motion Compensation) dựa trên biến đổi affine/homography của ảnh nền, kết hợp Re-ID nhận diện lại người khi góc quay camera thay đổi đột ngột.

## 4. Nếu có thêm thời gian

Nếu có thêm thời gian, nhóm sẽ thực hiện:
1. Thử nghiệm các mô hình Re-ID mạnh hơn và đa dạng hơn hoặc fine-tune Re-ID chuyên dụng cho môi trường thiếu sáng/ban đêm để cải thiện hiệu năng trên các video như video_2.
2. Quét lưới tham số (grid search) mịn hơn cho ngưỡng `--conf` (bước nhảy 0.05) kết hợp ngưỡng `--iou` và các tham số nội bộ của tracker (như `track_high_thresh`, `track_buffer`, `match_thresh`) để tối ưu hóa triệt để điểm HOTA.
3. Trích xuất các frame xảy ra lỗi chuyển ID (ID switch) và mất dấu (fragmentation) trên video_1 để trực quan hóa ma trận cosine distance của Re-ID và ma trận IoU, từ đó phân tích định lượng nguyên nhân gây lỗi ở cấp độ từng frame.
