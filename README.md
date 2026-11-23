# DVKG

**Dual-View Knowledge Graph Construction for Textbook-Specific Retrieval-Augmented Generation**

Kaixi Wu, Ronghan Li, Ruyi Liu, and Qiguang Miao · Xidian University

DVKG builds a shared concept graph for each course and a separate structure graph for each textbook. The concept graph captures concepts and semantic relations. The structure graph preserves textbook chapters, sections, paragraph order, and source locations. Evidence-Constrained Hierarchical Alignment (ECHA) verifies links from concepts to supporting passages. A query can route through the shared concepts and retrieve evidence from the selected textbook.

## Repository contents

| Path | Contents |
| --- | --- |
| [`src/`](src/README.md) | Construction code for the two graph views, ECHA, retrieval and evaluation, and offline checks |
| [`data/`](data/) | Textbook inventory, released numerical inputs and results, and verification scripts |
| [`docs/JOURNAL_SUBMISSION.md`](docs/JOURNAL_SUBMISSION.md) | Journal statements and versioned archive instructions |

## Reproducing the experiments

The experiments use eight textbooks across Computer Organization, Operating Systems, and Computer Networks. The inventory identifies each book and the evaluation materials specify which subsets were used. The main textbook QA comparison uses two books per course. The four-book setting is specific to Computer Organization.

Two checks are available without access to the original textbooks or model services. The construction smoke test uses a synthetic textbook. Numerical replay uses the public evaluation bundle in [`data/`](data/).

To run the synthetic graph demo from a Python environment:

```sh
cd src
python -m pip install -r requirements.txt
python -m dual_kg demo --output ./outputs/demo
```

The [source guide](src/README.md) describes the construction and evaluation entry points. Run `python data/verify.py` from the repository root to check the numerical results.

Full construction requires lawfully obtained textbook inputs and the parsing, language-model and embedding environments documented with the code. Full link prediction also requires the documented PyKEEN environment. Baseline adapters may need independent upstream software and model credentials.

## Data access

The release contains numerical evaluation artifacts and opaque source identifiers. Paragraph Recall@5 can be recalculated from the released identifier lists. Saved answer F1 values can be aggregated. Original textbook files, cached source passages, and generated answers are subject to their respective access and redistribution conditions. See the [data documentation](data/README.md) for details and links to provenance records.

## Citation and license

See [`CITATION.cff`](CITATION.cff) for citation metadata. The original software is available under the [MIT license](LICENSE). The [notice](NOTICE.md) describes the license scope.
