---
name: add-strategy
description: Scaffold a new retrieval strategy for the RAG evaluation project. Use when I ask to add a retrieval strategy, implement a new retriever, or "add [name] strategy".
---

Add a new retrieval strategy end to end, consistent with the existing pattern:
1. Create src/rag_eval/retrieval/[name].py implementing the Retriever interface in base.py. Match the existing signatures and return types exactly.
2. Add configs/strategies/[name].yaml (and any keys it needs in config.yaml).
3. Register it so it's selectable by config — no hardcoding.
4. Add tests/test_[name].py asserting it returns top_k chunks with scores for a sample query, mirroring the existing strategy tests.
5. Run /test and fix failures.

Do not modify the Retriever interface or other strategies unless I ask. Report what you added and any interface friction you hit.