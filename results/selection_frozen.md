# Cấu hình đã chốt trên DEV (ghi trước khi chạy test)

Tiêu chí công bố trước, theo thứ tự ưu tiên:
1. Context Evidence Rate trên dev (evidence hợp lệ có trong context cuối gửi model).
2. Câu trả lời: số câu có đáp án được trả lời, không từ chối nhầm; không bịa khi thiếu evidence.
3. Chi phí và thời gian trung bình mỗi câu (chỉ để phân thắng bại khi 1-2 hòa).

Kết quả dev (xem results/agent_summary_dev.json): two_tools đạt Context Evidence 15/15, các cấu hình còn lại 13/15.
Chi phí tổng dev: base $0.0097, two_tools $0.0108 (+11%); thời gian TB 3.8s so với 4.4s.

**Chọn: `two_tools`** (dense + BM25-bigram, tối đa 3 lượt, top-3, budget 3600 ký tự). Baseline để so sánh trên test: `base`.
Test chỉ được chạy một lần cho hai cấu hình này, không chỉnh gì sau khi xem.
