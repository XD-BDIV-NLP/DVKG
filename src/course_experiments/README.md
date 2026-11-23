# 课程图谱实验输入与矩阵

本目录只做离线清单冻结、摘要验证及矩阵生成，不调用模型、不训练、不生成分数。
两门课分别使用 A、B、A+B 教材；主方法覆盖 fixed_g0、fused、textbook_only 与
rag、topology、kge。每个声明的基线覆盖同样教材与评测，状态为 native，图谱作为
external_reference。BookQA 已按用户指示移除。

G0 的内容身份只依赖课程描述和 Schema，不包含教材、全局图谱或历史运行结果。
生成协议另由模型、种子、提示词和预算确定，并包含在整个清单摘要中。
v4604 完成图谱清单只作历史参考，不能作为 textbook_only 的输入。

## 输入格式与阶段

`sources.json` 为 `{ "book_id": "/absolute/path/to/independent/corpus.json" }`。
必须恰好覆盖 courses.yaml 声明的四本教材。服务器元数据
`course_assets_20261014.json` 的 `books[].corpus` 可用来准备这份路径映射；
运行命令的主机须能访问真实文件，不在本地把远程路径当成教材内容。

`assets.json` 为 `{ "role": "/absolute/path/to/file" }`。
构图阶段必需 role：`schema`、`g0_prompt`、`textbook_prompt`。
实际运行所用的其他提示词、runtime_config 和构图实现文件也应逐个作为额外 role 登记，
冻结会保存所有登记文件的完整 SHA-256。一个角色对应一个文件；提示词目录内每个文件
分别登记。不要只登记目录或凭空填摘要。

评测阶段额外必需 role：`rag_prompt`、`questions`、`reference_facts`、`fact_split`、
`gpu_environment`、`evaluator_rag`、`evaluator_topology`、`evaluator_kge`。
独立问题、参考事实和等价组拆分必须准备完成；评测器依赖文件也应登记额外 role。

构图阶段只要求 g0/textbook 预算。评测阶段要求全部四组预算，字段见 experiments.yaml。
缺失字段或 null 均拒绝冻结；没有冻结后的默认预算。当前 rag/kge 保留 null，
不代表已冻结。构图预算采用已观察配置：12000 输出 token、温度 0，教材 chunk_chars
4500、evidence_chars 650、重试 6；完整运行配置必须在实际冻结时另行登记和核对。

## 命令

在仓库根目录、可访问所有输入的主机上执行：

```text
python -m course_experiments freeze --stage construction --courses course_experiments/config/courses.yaml --experiments course_experiments/config/experiments.yaml --sources sources.json --assets assets.json --output construction.frozen.json
python -m course_experiments validate --manifest construction.frozen.json
python -m course_experiments matrix --manifest construction.frozen.json --output construction.matrix.json
```

评测输入准备后用 `--stage evaluation` 写入新的清单。评测清单必须再次登记同一构图
输入与协议。命令对输出采用独占创建；已有文件不能覆盖。输入文件和配置文件必须保持
原位且原内容，validate 会重新读取所有文件核对摘要。改动任何输入都需新的清单和结果，
不能用旧清单或旧 execution_id 复用。清单本身也有内容摘要。

构图阶段可调用 `manifest.require_construction(path)`，但对应评测矩阵全部 blocked、
score=null。评测执行前调用 `matrix.require_executable(row, manifest_path)`；它重新
验证文件摘要、清单身份和规范矩阵记录。未核验基线要求完整 40 位 commit、repository、
status=verified 和包含当前评测的 capability_scope；否则执行被阻止，分数保持 null。
能力范围仅声明已核验可执行的评测，不将原生状态伪装成融合状态。

相同 execution_id 只执行一次。fixed_g0 的 topology/KGE 在固定参考事实和拆分下
按同一输入去重；RAG 可检索教材正文集合不同，A/B/AB 保留不同 execution_id。
row_id 保留每条比较记录，deduplicated 标记重复执行输入。阶段包含在清单摘要中，
构图冻结不能当作评测冻结使用。所有分数由后续实际执行器产生，本模块不产生占位分数。

## 离线测试

```text
python -m unittest discover -s course_experiments/tests -v
```

Windows 测试使用正常系统临时目录；若沙箱拒绝临时文件或不可变硬链接操作，用正常 Windows 权限运行已授权的离线测试，不重设 TEMP/TMP。

## Task6 调度与结果查看

入口 `python -m course_experiments.run`，只有 prepare/run/collect/report，不接受自由命令。

```powershell
python -m course_experiments.run prepare --manifest evaluation_manifest.json --catalog catalog.json --output plan.json
python -m course_experiments.run run --plan plan.json --root jobs
python -m course_experiments.run collect --plan plan.json --job-id DIGEST --result verified_gpu/external_kge_result.json --root jobs
python -m course_experiments.run report --plan plan.json --root jobs --output report
```

使用 `run.freeze_catalog(entries, output, additional=[])` 创建不可变目录。entries 按 matrix.generate 的 row_id 索引；每条包含 phase（Task5冻结阶段路径）、scope(full/smoke)、metadata(file_record)、inputs（阶段角色→真实文件路径）、arguments（下表固定参数）、可选 control/native_undirected/native_semantic_kge/epochs/device。metadata JSON 必须包含 course_id/method_id/state/book_ids/scope/status=complete/phase_id/graph(file_record)/source_records（实际源元数据file_record列表；每项必须在 phase.files 中），native或schema_aligned额外 native_commit，必须与实际evaluation manifest内verified方法的commit完全一致。prepare将canonical_method和manifest_record冻结到计划，运行与复用再次核验，baseline身份进入job_id。metadata 自身由目录哈希绑定，避免与phase形成循环引用。GPU source_records 只填本地派生construction/attestation元数据，不填主服务器原文路径。

| 评测 | arguments 必须恰好包含 |
|---|---|
| rag | construction/reference/index/graph |
| topology | graph/evaluator |
| kge | graph/protocol/patched_evaluator |

inputs 包含参数对应的数据角色，及需核验的 projection/fact_split/common_mask 等；evaluator和patched_evaluator通过各自phase角色绑定。Task5其他准备步骤仍通过 `python -m course_experiments.evaluation` 执行，调度器不重写其指标或GPU训练。phase包括完整真实输入、预算、seed、代码和环境摘要；调度身份额外绑定调度器自身代码及protocol。G0 topology可复用完全相同身份；RAG active_books必须与冻结budget一致，不跨教材集合复用。

additional 列表每项为 `{row_id: 原生行ID, variant: schema_aligned, entry: 新目录条目}` 或 `{row_id: RAG行ID, control: vector_only或no_retrieval, entry: 条目}`。aligned method显示为原method追加 `_schema_aligned`，state=schema_aligned，其metadata也使用该明确身份；不能复制主方法结果充当原生基线。相同输入原生baseline只训练一次，各比较位置引用同job。缺少必要baseline、控制或任何正式评测仍未完成时，core_comparison_complete为false。Tree保留unavailable+null。

`textbook_only(course_id, schema_id, contributions, active_books, runtime=...)` 使用0节点/claims/evidence/books的确定性记账carrier调用已审fusion.build_view；无G0生成或原图过滤。真实调用需父任务先冻结该组合代码与阶段；夹具可显式alignment。源贡献不变，独立A/B/AB可分别生成并用views.export_view冻结。

每job独占锁、不可变record和真实raw_result文件哈希；失败现场保留，独立job继续。HTTP401/402/403停止本次后续新执行。失败记录不自动重试，诊断后需新phase/attempt身份。收集GPU结果必须带Task5真实protocol、split、graph、scope、run_id与options，并匹配covered/common mask选定的有序test_ids、fixed/covered/common计数；完成结果额外匹配epochs/train/validation/test计数，裸CSV拒绝。真实同目录pykeen_metrics.json的摘要绑定到record，复用和viewer再次核验，展示其实际数值，保留diagnostic failure状态。GPU本地phase资产必须可验证；远端路径无法访问时先在GPU环境完成核验/目录准备，再移植成可访问且重新冻结的收集资产，不能假装主服务器能核验远端文件。

输出summary.json和无依赖、内容转义的index.html。显示各比较位置、唯一执行、层级、状态、数量、覆盖、指标、原路径与身份，不导出原书正文或响应。score保持null，指标展示Task5实际metrics或摘要绑定的pykeen_metrics.json；原reason/error_message与raw_paths显示，验证失败不显示旧指标。machine_generated参考不是人工金标准。正式与smoke路径/身份分离；该本地实现测试不代表API或GPU实跑通过。

KGE 500/1000轮分别登记比较位置；catalog.additional可用 `{row_id: KGE行ID, epochs: 1000, entry: 对应冻结条目}` 添加另一轮。smoke只2轮。viewer明确显示epochs；任一适用KGE分组缺500或1000则core_comparison_complete=false。

Native语义KGE的N/A只能由phase内native_artifact的已冻结真实capabilities、原因及实际无向/未类型化图证明，并与待评测graph完全匹配。native_semantic_kge=false不能关闭primary或schema_aligned评测。输入/缓存失效保存独立validation_failure摘要记录，保留原error/type，不覆盖已有结果，继续其他job。收集与执行共用同job独占锁。
