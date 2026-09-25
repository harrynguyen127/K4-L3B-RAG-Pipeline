# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 25/09/2026 |
| Framework and version              | Custom evaluator `src/evaluate_retrieval.py`; dependency versions were not captured in the artifact |
| Evaluator model                    | Không dùng LLM evaluator; đối chiếu exact evidence span sau khi chuẩn hóa Unicode và khoảng trắng |
| Generator model                    | `deepseek-flash` được cấu hình mặc định nhưng không được gọi trong retrieval benchmark |
| Embedding model                    | `sentence-transformers/all-mpnet-base-v2` cho benchmark tiếng Anh |
| Corpus version/commit              | 5 trang IELTS Writing đã chuẩn hóa; working tree tại HEAD `594a625` (artifact evaluation chưa commit) |
| Golden dataset size                | 30 câu tiếng Anh: 10 BM25-targeted, 10 semantic paraphrase, 10 multi-constraint |
| `top_k`                            | 5; mỗi dense/BM25 lấy 10 candidate trước khi RRF |
| Fallback threshold and calibration | Mặc định `0.3` theo cosine dense; chưa được hiệu chỉnh và không được kích hoạt trong benchmark này |

## Configurations

- **Config A — dense-only:** Markdown section-aware chunking (`1200/0`), normalized cosine embeddings bằng `all-mpnet-base-v2`, lấy top 5.
- **Config B — hybrid + RRF:** Cùng chunking và dense model với Config A; kết hợp dense top 10 và BM25 top 10 bằng RRF (`k=60`), trả top 5.

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

Benchmark hiện tại chỉ đánh giá retrieval bằng exact evidence-span match. Generator/prompt không được chạy, vì vậy chưa thể báo cáo trung thực bốn metric chất lượng câu trả lời bên dưới.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      | Chưa đo | Chưa đo | N/A |
| Answer relevance  | Chưa đo | Chưa đo | N/A |
| Context recall    | Chưa đo | Chưa đo | N/A |
| Context precision | Chưa đo | Chưa đo | N/A |
| **Average**       | **Chưa đo** | **Chưa đo** | **N/A** |

### Retrieval diagnostics hiện có

| Metric | Config A — dense-only | Config B — hybrid + RRF | Delta B−A |
|---|---:|---:|---:|
| Hit@1 | 0.500 | 0.667 | +0.167 |
| Hit@3 | 0.667 | 0.867 | +0.200 |
| Hit@5 | 0.767 | 0.900 | +0.133 |
| MRR@5 | 0.587 | 0.764 | +0.177 |

## A/B comparison

- Cấu hình tốt hơn: **Config B — hybrid + RRF** trong benchmark retrieval tiếng Anh.
- Evidence: Config B đạt Hit@5 `0.900` và MRR@5 `0.764`, cao hơn Config A lần lượt `+0.133` và `+0.177` trên cùng 30 câu hỏi.
- Trade-off về latency/cost: Chưa ghi nhận số đo runtime/cost. Config B phải chạy thêm BM25 và RRF nên có thêm chi phí xử lý so với dense-only, dù BM25/RRF chạy cục bộ và không phát sinh LLM token.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
| 1 | Q06 — Phân biệt organisation/linking với vocabulary assessment | A và B | N/A | N/A | N/A | N/A | retrieval | Cả dense, BM25 và hybrid đều không đưa chunk chứa trọn evidence vào top 5; câu hỏi multi-constraint cần hai nhóm tiêu chí trong cùng evidence. |
| 2 | Q09 — Cấu trúc đoạn văn được đề xuất cho essay | B | N/A | N/A | N/A | N/A | retrieval | BM25 xếp đúng evidence ở hạng 1 nhưng dense không tìm thấy; RRF làm chunk đúng rơi khỏi top 5. |
| 3 | Q13 — Các định dạng bị cấm trong Academic visual task | B | N/A | N/A | N/A | N/A | retrieval | Dense tìm thấy evidence ở hạng 3 nhưng BM25 không tìm thấy; fusion làm mất kết quả dense-only có ích. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
| 1 | Điều chỉnh RRF/candidate depth hoặc bảo toàn kết quả mạnh từ từng retriever | Q09 và Q13 có evidence trong một retriever nhưng bị loại sau fusion | Giảm lỗi do một retriever kéo tụt evidence đúng | Chạy lại 30 câu, yêu cầu Q09/Q13 đạt Hit@5 và không làm giảm aggregate Hit@5/MRR@5 |
| 2 | Benchmark đúng model production `BAAI/bge-m3`; thử query translation cho tiếng Việt | Benchmark hiện dùng `all-mpnet-base-v2`; bộ tiếng Việt với multilingual MiniLM chỉ đạt hybrid Hit@5 `0.600` | Chọn cấu hình dense phù hợp với truy vấn đa ngôn ngữ | Chạy cùng dataset EN/VI, cùng chunking/top_k và so sánh Hit@k, MRR@5, latency |
| 3 | Hoàn thiện evaluation end-to-end bằng generator và RAGAS | Hiện chưa có Faithfulness, Answer relevance, Context recall, Context precision | Có đủ bằng chứng về chất lượng câu trả lời/citation, không chỉ retrieval | Chạy A/B cùng generator, evaluator, prompt và top_k; lưu raw outputs, latency và cost |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Markdown section-aware chunking `1200/0` | Legacy recursive `500/50` | Hybrid Hit@5 `-0.033`; MRR@5 `-0.016` | Chưa đo | Chunk mới giảm 137 xuống 95 chunks và giữ heading context, nhưng chưa cải thiện aggregate hybrid với model benchmark hiện tại. |

## Limitations

- Kết quả trên đo chunk retrieval, không đo answer correctness hoặc citation quality.
- Corpus mới có 5 trang IELTS, chưa đủ 8 nguồn dự kiến.
- Benchmark model khác model embedding production mặc định; chưa nên xem các số dense/hybrid là production estimate.
- Chưa đo latency, token usage và chi phí; chưa hiệu chỉnh fallback threshold bằng tập in-domain/out-of-domain.
