# DVKG source

Run these commands from the repository's `src/` directory using Python 3.10+.

```sh
python -m pip install -r requirements.txt
python -m dual_kg demo --output ./outputs/demo
```

The offline demo builds both graph views on a synthetic textbook. It uses fixed model responses to exercise the interfaces and makes no model-service requests. Use a fresh output directory for each demo run.

| Component | Entry point or source |
| --- | --- |
| Book, chapter, section and paragraph graph with source positions | `python -m dual_kg structure-build`, `legacy_text_graph_tools/build_hierarchy_view.py`, `course_experiments/dual_view.py` |
| Source preparation and earlier MinerU/PDF processing tools | `dual_kg concepts-prepare`, `domain_ckg_workspace/ckg_incremental/sources.py`, `legacy_text_graph_tools/` |
| Concept prior, textbook contributions, identity resolution and fusion | `python -m course_experiments.construction`, `course_experiments/fusion.py`, `domain_ckg_workspace/ckg_incremental/` |
| ECHA candidate preparation, verification, routing and selected-book QA | `python -m course_experiments.evaluation.run_echa`, `course_experiments/evaluation/echa_pipeline.py`, `course_experiments/evaluation/evidence_hierarchical_alignment.py`, `course_experiments/evaluation/echa_rag.py` |
| Graph geometry, RAG, ablations and native-graph KGE | `scripts/topology_metrics.py`, `course_experiments/evaluation/` |
| Method prompts and concept schema | `prompts/`, `config/schema.yaml`, `course_experiments/evaluation/` |

## Use your own textbook inputs

### Textbook structure

The structure CLI accepts a physical paragraph graph containing `headings` and `nodes`. The synthetic demo writes a small example under `outputs/demo/fixture/data/demo_book/text_graph.json`. For a custom graph, use:

```sh
python -m dual_kg structure-build --source-graph inputs/text_graph.json --book-id example_book --book-title "Example textbook" --output outputs/structure
```

The structure graph stores containment, reading order and paragraph provenance. See [input format](docs/API.md) and [the historical preprocessing stages](docs/%E6%9E%84%E5%BB%BA%E8%BF%87%E7%A8%8B%E4%B8%8E%E6%BA%90%E7%A0%81%E5%AF%B9%E5%BA%94.md). Beginning with a new PDF also requires OCR/parse inputs and the historical scripts' expected layout. PyMuPDF and Pillow are optional dependencies for those historical PDF tools.

### Concept graph and ECHA

The CLI can prepare evidence from a verified, published textbook workspace, then build a resumable concept graph. Set `SILICONFLOW_API_KEY` in the environment for the example API configuration. Real construction and verification can make paid model requests.

```sh
python -m dual_kg concepts-prepare --project-root inputs/textbook_workspace --book-id example_book --output outputs/prepared
python -m dual_kg concepts-build --corpus outputs/prepared/corpus.json --config config/incremental.example.yaml --output outputs/concepts --allow-api
```

For the manuscript's course-specific build, prepare the legal local inputs, use `python -m course_experiments freeze --help` to create a construction manifest, and build the course prior and each independent textbook contribution with `python -m course_experiments.construction --help`. See [the research configuration and export protocol](course_experiments/README.md). The frozen graph, source units, artifact inventory and tokenizer manifest provide inputs for the ECHA command:

```sh
python -m course_experiments.evaluation.run_echa --help
python -m course_experiments.evaluation --help
python -m course_experiments.evaluation.native_graph_kge --help
```

The research runners keep the input hashes and require the corresponding files to exist in the local execution environment. The synthetic example demonstrates graph construction and the regression tests exercise ECHA verification and retrieval logic. Recreating the paper graphs requires the matching textbook editions, parsing and source ownership artifacts, and model environment. Published aggregates can be checked using the corresponding replay material on their own.

For topology metrics, install `requirements-graph-analysis.txt`. The reported training code uses PyKEEN 1.11.1 and PyTorch 2.5.1 in the frozen CUDA 12.1 environment. Baseline adapters integrate K12-KGraph, GraphRAG, LightRAG and HippoRAG 2 through their external upstream installations and input contracts. The upstream tools, their proprietary model assets and the baseline indexes are installed or prepared separately. The optional Neo4j projection requires the Neo4j Python driver and your own local database configuration.

## Historical experiment scripts

`scripts/historical/` contains the cold-build timing worker, expanded-pool retrieval and reranker runners, and an earlier geometry audit. These record the execution protocol and require the original frozen source assets, including the authorized local model files for the GPU reranker. `DVKG_HISTORICAL_ROOT` points the expanded-pool scripts to a prepared artifact tree. `DVKG_RERANKER_ROOT` points the reranker to the directory containing the model manifest and exact model files. `DVKG_KGE_LOCK_PATH` can designate a common lock file for users sharing a GPU server. The repository's offline replay materials are available without these runtime assets.

## Offline checks

```sh
python -m unittest tests.test_api tests.test_http
python -m unittest course_experiments.tests.test_dual_view course_experiments.tests.test_construction course_experiments.tests.test_fusion course_experiments.tests.test_evidence_hierarchical_alignment course_experiments.tests.test_echa_pipeline course_experiments.tests.test_echa_rag course_experiments.tests.test_native_graph_kge course_experiments.tests.test_rag_component_ablation
```

[Source provenance](SOURCE_SNAPSHOT.json) records the source and release hashes for copied files and describes the portability changes. The previously released construction interfaces and derived portions retain the [upstream MIT notice](licenses/MIT-dual-view-textbook-kg.txt).
