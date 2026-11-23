# 固定版本构图对照

本目录只做原生方法调用、输入适配、检查点及结果投影。BookQA 已排除。Tree-KG 缺少本地构图源码，不能运行，必须显示不可用原因。

## 环境与冻结

每种方法使用独立 Python 环境和完整官方固定源码。依赖文件是静态导入闭包推断，尚不保证完整安装；运行前必须验证实际原生导入。GraphRAG 最稳妥采用对应官方 pyproject 的完整依赖。KGGen 准备官方 NLTK punkt/punkt_tab 数据后再导入，避免运行时下载；不创建本地 SentenceTransformer 模型。

`source_pins.json` 保存已核验官方 tracked tree 的全部文件 SHA256；排除 Git 数据、编译缓存及安装生成 egg-info。freeze 与 run 都要求源码逐文件相同。依赖锁和当前安装版本、全部适配代码、Schema、模型、预算、原始输入都进入阶段身份。不得修改已存在 construction_manifest。

在执行环境写入 parameters.json：

```json
{"base_url":"https://api.siliconflow.cn/v1","model":"deepseek-ai/DeepSeek-V4-Flash","embedding_model":"BAAI/bge-m3","max_tokens":12000,"temperature":0,"concurrency":2,"retries":2,"max_gleanings":1,"seed":42}
```

```text
python -m pip freeze > environment.lock.txt
python -m course_experiments.baselines.runner freeze --construction-manifest construction_manifest.json --method k12_kgraph --native-source /absolute/K12-KGraph --parameters parameters.json --dependency-lock environment.lock.txt --phase k12_phase.json
python -m course_experiments.baselines.runner run --construction-manifest construction_manifest.json --method k12_kgraph --native-source /absolute/K12-KGraph --phase k12_phase.json --course-id computer_organization --book-set AB --smoke-chunks 2 --output-root /absolute/baselines
```

另外两种方法 ID 是 `kggen_lm_dedup` 和 `graphrag_extract`，各自源码和 phase 分开。正式运行去掉 smoke 参数；A、B、AB 每课程分别执行。必须在原始服务器运行；教材原文不下载到本地。密钥仅通过 `SILICONFLOW_API_KEY` 环境读取，不写入配置、报告或冻结文件。冻结前原运行 runtime_config 的两个 base_url 必须等于官方 URL。

API：`freeze_phase(construction, method, source, parameters, dependency_lock, output)`；`execute(construction, phase, method, source, course_id, aliases, output, smoke=None, schema_align=False)`。源码路径必须指向完整固定 checkout；正常安装如果原包从 site-packages 先导入，将拒绝外部源码身份。每种方法应新进程运行。

## 调用与差异

- K12：官方 `process_section` 和 `merge_graphs`。既有片段代替 Markdown 分节；每片段单独原生节点前缀；只合并当前课程输入，不读取 subject/global 历史图。使用官方 prompt、解析、节点清洗和名称去重。主概念层只取 Concept，原始其他类型不丢弃。
- KGGen：官方 `generate(cluster=False)`、`aggregate`、`run_deduplication(LM_BASED)`。构造器 `retrieval_model=None`，只向 LM_BASED 注入 BGE API encode。原生 64 线程池改为阶段冻结 concurrency（仅调用期 monkeypatch），其余聚类/去重算法保留。NumPy seed42 但集合顺序和并发可能影响可复现性；原始图和簇保存，不声称严格确定性。空图或无关系明确记录未调用去重，不冒充有去重结果。
- GraphRAG：官方 GraphExtractor 的原生 prompt、gleaning、解析、跨输入描述合并。ChatModel 包装保留完整历史并调用现有 ModelRuntime；Schema 仅提供实体类型词表。无向边保留描述、weight、source_id；weight 不是置信度。

统一输入是按原顺序串联片段 `evidence_ids` 对应完整正文，**不加入 context、G0、课程图或以前教材图**。本次受控分节策略显式记录。每次模型输出上限12000、温度0；服务和片段重试次数均由 retries 指定，因此一个片段存在多层有界重试；这是次数上限而非相同总 token 成本。原生不同数量的提示/gleaning产生不同实际成本，应从原响应 usage 汇总。

## 输出及适用范围

完整结果路径为 course/method/full或smoke/input_id。单片段不可变 complete 检查点按方法、课程、phase、范围和 source unit 隔离；A/AB 复用相同单书输入，正式与小样本永不碰撞。失败尝试保留原响应和异常类别/HTTP码；HTTP402立即停止，不空重试。格式故障有界重试并继续后续独立片段，任何未完成输入禁止发布 complete 总结果。聚合后再次独立落盘，发布中断不会重跑成功的模型聚合。

`artifact.json` 包含 native_graph、semantic_view、来源、冻结参数、受控变体差异和执行状态。`entities/relations/evidence.json`、`concept_kg.json`、`extractable_doc_nodes.json` 对接现有评测器字段；`native_concept_topology.json` 是单独原生概念邻接拓扑，`native_topology.json` 保留全部原生邻接。KGGen 未提供原生实体类型，字符串实体标记 candidate；公共概念层评测要求另做下面的类型对齐。

原生提取来源 sidecar 标记 `source_unit_provenance_not_verified_sentence`，不能称每条事实已核验原文。聚合后只能通过原生精确名称/簇成员追溯的事实保留单位证据；无法追溯明确标记 unresolved。原生方法未建模 conditions，明确记录限制。

公共类型关系投影只接受已知精确映射，K12 的 is_a 和 prerequisites_for→prerequisite_of；KGGen 仅 Schema 已有谓词。所有未知谓词、结构层关系、无向描述边保留为 unmapped。映射覆盖率明确记录，不补造 causal/related_to。

GraphRAG 原生概念拓扑与 RAG 可用；共同类型化语义拓扑和 KGE 为 null，并附缺谓词/方向原因，不能把空关系列表当0分。KGGen 未对齐原生实体类型时，公共概念评测同样不可用。native_graph_state 一律 native，不制造外部方法的融合前后状态。

## 单独的原文支持模式对齐

为比较公共概念语义图，可在阶段参数中加入 `"alignment":{"max_tokens":12000,"temperature":0,"retries":6}` 后冻结，以 `run ... --schema-align` 执行。适配代码和 alignment_prompt.txt 被阶段资产哈希覆盖；没有此预算块禁止执行对齐。

对齐只读取当前原生现有节点/边、其描述、Schema和已选片段原文，按单个原生来源单位调用模型。每个节点被分类为允许概念类型或排除，每个边被映射为允许谓词与有原文支持的方向或保持 unmapped。覆盖全部输入ID、不能增加节点/边、不能改成其他端点；每次有效判断要求精确原文子串，适用条件必须是引文子串。原生无向边只有源句支持时才获得方向。映射冲突或类型冲突保留审计并排除对应公共图对象，绝不猜测。

额外调用和费用的受控变体名称为 k12_kgraph_schema_aligned、kggen_lm_dedup_schema_aligned、graphrag_extract_schema_aligned。结果在原生目录下 `schema_aligned/`，包含原图ID、逐单位响应检查点、实体/边全量审计、引用、条件、覆盖率、独立请求统计与原响应 usage 文件，以及公共评测导出。原生结果先完整落盘，不会被对齐替换。对齐已成功的单位不会重发；402立即停止并保留全部记录。

引文存在性和Schema一致性是程序校验；概念类型与方向的语义判断仍由冻结模型做出，标记 `model_source_backed_not_human_verified`，不能声称人工金标准。

本地离线验证不等于实际外部方法复现。部署后原生导入、模型小样本、完整实验和真实评测均由控制任务核实后才能标记执行完成。
