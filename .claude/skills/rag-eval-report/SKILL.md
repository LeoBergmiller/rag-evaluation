---
name: rag-eval-report
description: Produce the strategy comparison report from the latest benchmark run. Use after running evaluate, or when I ask for the ablation table, strategy comparison, or eval report.
---

Read the newest results file in results/. Produce a markdown report with:
1. An ablation table: rows = retrieval strategies; columns = faithfulness, answer relevancy, context precision, context recall, recall@k, p95 latency, cost/query. Include 95% confidence intervals where sample size allows.
2. A short paragraph naming the winning strategy and WHY, with the tradeoff (e.g., rerank wins precision at a latency cost).
3. Any regression vs. the committed baseline, flagged.

Keep it concise and interview-ready. Do not re-run the benchmark unless I ask.