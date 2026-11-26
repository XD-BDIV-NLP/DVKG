# Computer Organization textbook graph example

This example comes from *Computer Composition and System Architecture* (计算机组成与系统结构), one of the textbooks evaluated in the DVKG paper. See [the study inventory](../../textbooks.json) for the source fingerprint, and the [publisher catalog](https://www.xduph.com/newPages/BookListContent.aspx?className=&classid=2.2.11.%253B&kType=0&keyword=&newClassid=%253B) for bibliographic access. The study record identifies the research input by fingerprint.

| File | Records | Meaning |
|---|---:|---|
| `textbook.json` | 1 | Textbook metadata, source fingerprint, and graph counts |
| `concept_nodes.csv` | 2,337 | Typed concepts that appear in selected semantic or alignment edges |
| `semantic_edges.csv` | 3,261 | Directed relations from the shared course graph with a recorded source in this textbook |
| `structure_nodes.csv` | 3,889 | One book, 9 chapters, 163 sections, and 3,716 paragraph nodes |
| `structure_edges.csv` | 7,603 | Chapter/section containment and paragraph order |
| `alignment_edges.csv` | 4,149 | Accepted ECHA concept-to-paragraph links with definition, explanation, or example roles |

The semantic graph is a projection of the frozen shared course graph scoped by textbook provenance. The two views can be joined through the concept and paragraph IDs in the alignment file. IDs are opaque and consistent across these files. Each CSV has a header row and can be read with Python's standard `csv.DictReader` or imported into a graph database.

This release contains derived graph structure and bibliographic metadata. It omits textbook pages, passages, quotations, concept definitions and names, verifier explanations, model caches, and QA responses. It supports inspecting graph structure and cross-view alignment. To rerun extraction or evidence-grounded question answering, obtain the corresponding textbook from a lawful source and follow the [construction instructions](../../../src/README.md).
