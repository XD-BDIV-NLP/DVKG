# Reproduction data

This archive accompanies the DVKG manuscript. It contains numerical results, study metadata, opaque identifiers, and metric replay inputs. Run the offline integrity and result checks with Python 3.9 or later:

```
python data/verify.py
```

The checks require the Python standard library. They verify file SHA-256 hashes, graph-statistic identities, KGE split constraints and metric summaries, QA score aggregates, and measured query latency. Existing study runs supply these records.

| Manuscript item | Data | What can be checked from this archive |
| --- | --- | --- |
| Table 1 | `textbooks.json` | Corpus scope, textbook counts, and source fingerprints. |
| Table 2 and Figures 3–4 | `topology.json`, `geometry.json` | Numeric topology statistics and recorded geometry summaries. Rebuilding the original semantic graphs requires the original inputs. |
| Table 3 | `kge_metrics.json`, `kge_splits/` | Results from 24 runs and the four original opaque-ID train/validation/test splits. Training weights and hardware-specific state are outside the archive. |
| Tables 4–5 | `qa/textbook_qa.jsonl`, `qa/ablation.jsonl`, `qa/aggregates.json` | Paragraph Recall@5 from IDs and score summaries from stored per-question F1. The main QA archive also preserves the previously reported exploratory recall-guard variant. |
| Table 6 | `construction_cost.json` | Recorded LLM input/output token use and active minutes. K12-KGraph is a partial measurement. |
| Table 7 | `query_latency.json` | Recompute P50 and nearest-rank P95 from 120 complete timing samples. The three native Computer Networks groups contain only published aggregates in this release. |
| Table 8 | `qa/expanded_pools_current.json`, `qa/expanded_pools_cross_encoder_saved_f1.jsonl`, `qa/expanded_pools_archived.jsonl` | The four baseline rows can be checked from the archived scores and paragraph IDs. The current DVKG cross-encoder row has 706 saved per-question F1 scores and audit-derived Recall@5 aggregates. Its reranked paragraph-ID lists are outside the archive. |

The frozen archive used earlier table numbers 3, 4, and 6 for what is now textbook QA, ablation, and expanded-pool QA. The three ID replay files cover 5,600 question–configuration pairs in those original conditions. Within the expanded-pool archive, the base DVKG rows are historical and differ from the final cross-encoder configuration. The current table specifies the cross-encoder row explicitly. The archived questions were observed during development and use machine-generated references and support annotations.

`qa/*.jsonl` contains labels, original opaque IDs, metric values, and optional artifact hashes. Paragraph Recall@5 deduplicates the ranked paragraph IDs before taking the first five and divides the count of retrieved support IDs by the annotated support count. Answer F1 is the original multiset token-overlap score. The released F1 numbers support aggregation only, because question and answer text are outside this archive.

The textbooks and their parsed text, retrieved passages, private indexes, vectors, and full graph artifacts are excluded. Researchers can acquire the listed editions through the applicable rights holders and prepare their own inputs. The dataset and artifact fingerprints identify the inputs used for the recorded experiments. They do not grant a license to redistribute any underlying textbook material.
