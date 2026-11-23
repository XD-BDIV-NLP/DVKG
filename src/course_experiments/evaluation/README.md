# 固定评测薄适配

所有真实原文、引用题集、RAG 索引和模型响应保留在主服务器。这里不修改 construction/fusion/baseline 模块，不覆盖原 evaluator/sweep。BookQA 排除，Tree 不可用状态由调度器保留。

## 主服务器阶段

命令入口 `python -m course_experiments.evaluation`，子命令 `freeze-phase`、`generate-reference`、`freeze-reference`、`project-kge`、`export-kge`、`common-mask`、`build-index`、`rag`、`topology`、`patch-evaluators`、`prepare-kge`、`kge`。各子命令 `--help` 给出参数。

`freeze-phase --paths paths.json --parameters parameters.json --output phase.json`。paths 是角色到绝对文件路径的 JSON；自动冻结全部本评测代码、splits.py、manifest/contributions 及本评测提示词。构建阶段的 immutable manifest 不能重写。运行阶段再次校验实际文件 SHA；原图、索引、题集、拆分或代码变化需新 phase。完成产物不可覆盖，partial 另存摘要命名版本。

参考生成 paths 至少 `construction`；parameters 必须为 `reference.PARAMETERS` 加 `smoke_chunks:null`（正式）或正整数（小样本）。原 construction 的源文件、Schema、Runtime 与提示词每次请求前检查。源片段按 seed42 摘要顺序、跨教材交错；只输入课程说明、Schema、原正文/evidence，不读取任何实验图。正式目标至少100题/400来源事实，覆盖六题型；没足够引用则 partial。事实和实体要求精确引用及端点出现，条件要求引文子串，多跳要求至少两条不同且连接的原文事实。别名要求原文明确共同出现。引用存在性是程序校验，语义仍是机器整理，未声称人工金标准。概念以 course/name/scope 固定身份；同名多义不猜测。

`generate-reference --construction ... --phase ... --course-id ... --output ... [--smoke-chunks N]`；生成完成才 `freeze-reference --reference .../reference.json --output frozen_reference.json --split-output fact_split.json`。别将 partial_reference 当正式题集。参考引文与 question 原件不发 GPU。

拆分按独立参考概念ID、显式别名和scope；consists_of 反向规范到 part_of，compared_with 端点排序，所有条件版本在同一三元组粗组，seed42、.8/.1/.1。原引用及条件在主服务器保留；普通KGE投影损失条件，在结果明确写出。`project-kge --reference ... --split ... --graph concept_kg.json --output projection.json`，所有 candidate 训练删除 validation/test 等价组，未知关系/歧义实体排除并审计。validation/test 保持同课程固定顺序。covered mask 只含训练出现的实体及关系；各图共同 mask 使用 `common-mask --projection p1 ... --output mask.json`，不足不能打零分。

主服务器为导出冻结一次 paths 包含 `graph/projection/fact_split` 的 phase。`export-kge --projection ... --split ... --phase ... --output derived_gpu_package`，产出 projection、去除原文 conditions 的 split元数据、primary_attestation 和 graph_attestation。仅传四个派生文件至GPU；不要传参考原文、题集、完整带evidence图。

索引阶段 `build-index` 的 paths 至少 construction，parameters 为 `{seed:42,embedding_model:BAAI/bge-m3,stage:build_index}`。每课程两本同一正文索引、BGE-M3，缓存向量。RAG阶段 paths 为 `construction/reference/index/graph`，parameters 是 `rag.BUDGET` 加 control 和排序 active_books；control=graph/vector_only/no_retrieval。phase 每图/控制/教材集合分别冻结；不要跨教材集合复用G0 RAG。同index/question与固定top5；整个context JSON的UTF8字节上限3000（包含来源ID/分隔），保守字节上界、不是模型精确token计数，provider原usage另存。answer2048/judge1024、温度0。引用只能来自当前教材且实际可见context子串；截掉引文就不能计召回。G0导航到原文，不自造教材引用。图检索固定一次概念名称匹配与邻居导航，加正文向量分数；native无向描述邻接可以导航，不补造谓词。每题完整query/context/answer/citations及响应usage/延迟/判分保存，完成题检查点复用；无法由当前教材回答的题为 unavailable，分数 null，不混入准确率分母。EM/F1 按汉字单字+英文/数字词、忽略标点；机器 judge 与人审状态分开。

拓扑 `topology --graph ... --evaluator ORIGINAL --phase ... --output ... [--native-undirected]` 使用原 `evaluate_graph`。phase冻结 `graph/evaluator_original/src_utils`；原依赖旁边存在 src/utils 时加载同一克隆，不误用预先导入的另一个 src。仅十个Schema概念类型；结构/未分类candidate排除。native nodes/edges 支持，标明原生邻接层；无向图用双向投影调用原函数，方向密度/出入度/SCC/reciprocity/PageRank 只作投影信息，不列为无向主比较指标。概念层为空为 insufficient 而非0分。

## GPU派生评测阶段

先用 `patch-evaluators` 在 `scripts/` 生成独立副本、原 SHA记录、补丁diff及新SHA。仅接受两份已核实原v2文件 SHA；原 evaluator 增加可选 external_protocol 分支和CLI参数，旧随机拆分路径保留；原 sweep增加参数转发，external模式禁用旧目录复用。没有重写图指标或训练循环，仍用原生 PyKEEN pipeline/原sweep的CSV与SVG格式。

GPU paths 至少 `graph`（graph_attestation.json）、`projection`、`fact_split`（sanitized）、`primary_attestation`、`construction`（仅元数据JSON）、`gpu_environment`、`dependency_lock`、`src_utils`、`evaluator_original`、`sweep_original`、`evaluator_patched`、`sweep_patched`。有共同mask再加入 common_mask。GPU只检元数据文件及本地真实软件/代码SHA，不递归访问 construction里的主服务器教材路径。src/utils实际依赖应为v2克隆；public src/__init__ 可作为额外资产冻结。GPU源图身份由主服务器已验证 phase 摘要和原图/拆分/投影 attestation绑定，sanitize 本身有独立摘要但logical split_id不变。

`prepare-kge --projection ... --split ... --phase ... --output protocol.json [--common-mask ...] [--smoke]`；然后 `kge --graph graph_attestation.json --protocol ... --patched-evaluator ... --phase ... --output ... --epochs 2|500|1000 --device cuda:0`。2epoch 是不同身份的小样本。正式TransE128维/batch256/negative16/lr.001/seed42，500和1000轮。训练/验证/测试来自固定协议、共同词表，不调用随机split。至少10训练且eligible validation/test非空；否则insufficient、无分数。原生pipeline模型与metrics保存；另调用原生RankBasedEvaluator保存可用逐三元组rank，未自写排序规则。GPU程序仍需父任务完成真实2epoch兼容验证；本地fake边界不等于GPU实跑通过。

失败响应/usage保留，HTTP402运行时即传播，停止而不空重试；格式故障有界重试后继续独立题/源片段，未完成不发布 complete。服务网络错误由冻结Runtime作有界重连；耗尽时保留错误身份、直接抛出。已完成模型响应复用。当前全部格式修复尝试失败的单元需要父任务显式复核后新范围/新阶段身份重试，不能把已缓存无效响应当已完成。

## 审查修复契约

参考集装配先在相同scope内合并明确同义称谓组件，再固定概念身份和拆分；引用、来源实体ID及显式别名映射保留审计。已有两个参考实体ID也在freeze_split和project_graph使用同一组件映射，避免CPU/中央处理器两份等价事实落入不同fold。跨scope同名仍保持独立，无法唯一映射则排除。

真实Runtime的llm.extra_body.enable_thinking必须明确为false，缺失或true在缓存/API前拒绝；请求体再强制false。参考/RAG/索引入口将整个实际runtime.config绑定至phase的runtime_config角色，或construction元数据中的已冻结runtime_config；字段变化会拒绝，配置摘要进入RAG身份和模型请求键。RAG夹具/自定义调用也需显式phase runtime_config，不得只提供graph/index/reference而绕过运行选项冻结。

KGE进入原生训练前写external_kge_identity.json、external_kge_protocol.json和阶段标记；factory/训练/模型保存/metrics保存异常保存failure_STAGE.json及external_kge_result.json（error_type/message/traceback/现存文件路径），随后原样抛出。已生成模型文件保留。可选rank诊断异常保留主训练模型和metrics，明确返回completed_with_diagnostic_failure、training_status=completed、diagnostics_status=failed，不冒充全部完成。完成结果可按同run_id复用；失败/insufficient/missing_dependency尝试目录保持现场，诊断后选择新的报告尝试目录，不覆盖旧文件。此状态需调度器显示为诊断待处理，不与completed合并。
