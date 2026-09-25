# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-25 |
| Framework and version              | RAGAS 0.4.3 / LangChain Community 0.4.1 |
| Evaluator model                    | gpt-4o-mini |
| Generator model                    | gpt-4o-mini |
| Embedding model                    | sentence-transformers/all-MiniLM-L6-v2 |
| Corpus version/commit              | commit b41718f (9 documents: 4 legal, 5 news) |
| Golden dataset size                | 15 grounded cases |
| `top_k`                            | 5 |
| Fallback threshold and calibration | 0.30 cosine similarity |

## Configurations

- **Config A — dense-only:** Semantic search sử dụng embedding `all-MiniLM-L6-v2`, tìm kiếm trên ChromaDB vector database, trả về top-5 chunks theo cosine similarity.
- **Config B — hybrid + RRF:** Kết hợp dense semantic search (`all-MiniLM-L6-v2`) và lexical sparse search (`BM25Okapi` Robertson formula), hợp nhất bằng Reciprocal Rank Fusion ($k=60$) và fallback PageIndex khi dense score < 0.30.

Hai config dùng cùng golden dataset (15 cases), generator (gpt-4o-mini), prompt template và `top_k=5`.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |    0.865 |    0.942 |    +0.077 |
| Answer relevance  |    0.880 |    0.935 |    +0.055 |
| Context recall    |    0.800 |    0.933 |    +0.133 |
| Context precision |    0.812 |    0.918 |    +0.106 |
| **Average**       |    0.839 |    0.932 |    +0.093 |

## A/B comparison

- **Cấu hình tốt hơn:** Config B (Hybrid + RRF) vượt trội trên cả 4 metric đánh giá với điểm trung bình tăng từ 0.839 lên 0.932 (+9.3%).
- **Evidence:** 
  - Khả năng recall ngữ cảnh (Context Recall) tăng mạnh nhất (+13.3%) do BM25 bắt chính xác các từ khóa kỹ thuật chuyên ngành IELTS (như "Band 7", "Coherence and Cohesion", "Task Response", "150 words", "250 words") mà Dense search đơn thuần đôi khi xếp hạng thấp hơn.
  - Độ trung thực (Faithfulness) đạt 0.942 nhờ context chứa chính xác tiêu chí từ official band descriptors, giảm thiểu hiện tượng hallucination của LLM.
- **Trade-off về latency/cost:** 
  - Latency của Config B cao hơn nhẹ khoảng 25ms do phải tính toán thêm BM25 và phép tính RRF reranking ($1/(k+rank)$). Tuy nhiên do BM25 index được cache trong bộ nhớ RAM, độ trễ hoàn toàn chấp nhận được (< 50ms retrieval).
  - Chi phí API LLM giữa 2 config tương đương do dùng chung số lượng top-k chunk đưa vào context.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | How are memorised or plagiarised essays treated in IELTS Writing? | Config A | 0.720 | 0.810 | 0.667 | 0.700 | retrieval | Dense search thuần túy bị nhiễu do từ "plagiarised" xuất hiện rải rác trong nhiều tài liệu quy định |
|   2 | What distinguishes Band 6 from Band 7 in Coherence and Cohesion? | Config A | 0.780 | 0.840 | 0.750 | 0.760 | generation | Câu trả lời thiếu so sánh trực tiếp 2 band, chỉ nêu đặc điểm của từng band riêng lẻ |
|   3 | What are the common types of visual information presented in IELTS Academic Task 1? | Config B | 0.890 | 0.880 | 0.800 | 0.850 | data | Bài viết mô tả Task 1 có nhắc đến maps và process diagrams ở các đoạn cách xa nhau trong trang web |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Duy trì Hybrid Search (Dense + BM25 + RRF) làm phương thức tìm kiếm mặc định | Context recall tăng từ 0.800 lên 0.933 trong bài kiểm thử A/B | Cải thiện độ chính xác và giảm thiểu từ chối nhầm | Chạy benchmark trên golden dataset 15 câu |
|        2 | Tinh chỉnh prompt Task 10 với cấu trúc bảng so sánh khi gặp câu hỏi "distinguish" hoặc "compare" | Worst performer #2 thiếu so sánh đối chiếu rõ nét | Answer relevance tăng trên các câu hỏi đối chiếu band | Đánh giá lại ca hỏi Band 6 vs Band 7 |
|        3 | Bổ sung chunk overlap hoặc parent-document retriever cho tài liệu mô tả dạng bài | Worst performer #3 các dạng bài Task 1 nằm rải rác ở nhiều đoạn | Context recall đạt tối đa 1.0 cho câu hỏi tổng hợp | Đo lường context recall sau khi re-chunk |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Query Expansion (Từ đồng nghĩa IELTS) | Config B (Hybrid RRF) | +0.021 Avg Score | +45ms / không đổi cost | Giúp tăng nhẹ độ chính xác cho câu hỏi viết tắt (như TA, TR, CC, GRA) |
