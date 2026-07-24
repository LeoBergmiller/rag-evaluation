# Cross-domain generalization: arXiv vs medical (PMC)

Same harness, same controlled config (`config_fingerprint=defa37d5071c7049`), two domains. arXiv eval_set_hash=`403ff26e0eff39a0` · medical eval_set_hash=`988a0ec1f7c79ee8`.

| corpus | strategy | faithfulness | context_recall | recall_at_k | ndcg_at_k | p95_latency_ms | cost_per_query_usd | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arXiv | bm25 | 0.9410 | 0.9327 | 0.9231 | 0.8129 | 5.40 | 0.0118 | ✓ |
| arXiv | dense | 0.9318 | 0.8462 | 0.7500 | 0.5420 | 39.91 | 0.0117 | ✓ |
| arXiv | hybrid | 0.9408 | 0.9135 | 0.8269 | 0.6831 | 40.64 | 0.0118 | ✓ |
| arXiv | hyde | 0.9267 | 0.8494 | 0.7692 | 0.6182 | 11444.68 | 0.0118 | ✗ |
| arXiv | rerank | 0.9473 | 0.9231 | 0.9038 | 0.8254 | 1104.74 | 0.0118 | ✗ |
| medical | dense | 0.9634 | 0.8029 | 0.6923 | 0.5838 | 42.11 | 0.0103 | ✓ |
| medical | bm25 | 0.9088 | 0.8854 | 0.8846 | 0.8206 | 108.09 | 0.0107 | ✓ |
| medical | hybrid | 0.9396 | 0.9100 | 0.8654 | 0.7339 | 100.60 | 0.0107 | ✓ |
| medical | rerank | 0.9441 | 0.9615 | 0.9615 | 0.9068 | 1316.44 | 0.0108 | ✗ |
| medical | hyde | 0.8751 | 0.6442 | 0.4808 | 0.3744 | 12301.34 | 0.0103 | ✗ |

## Does the ranking reproduce?

- nDCG@k ranking — arXiv:   rerank > bm25 > hybrid > hyde > dense
- nDCG@k ranking — medical: rerank > bm25 > hybrid > dense > hyde
- **The leading 3 strategies reproduce exactly** (rerank > bm25 > hybrid) — bm25 holds a top spot in both, so the 'cheap retrieval stays competitive' finding generalizes.
- The one clear divergence is **hyde**: nDCG@k 0.62 (arXiv) → 0.37 (medical), Δ-0.24 — query transformation that helps modestly on one domain can hurt on another.
- Operational reproduction: hyde, rerank fail the gate on the latency ceiling in **both** domains — the fancier strategies don't earn their cost on either corpus.
