# Docbot evaluation data

`gold.sample.jsonl` 是 10 条 schema 示例，不是当前语料的 gold。它包含示例答案、规范证据和占位 chunk ID，不能直接用于当前索引。

## 当前评测数据

`gold.jsonl` 覆盖冻结的 10 个文档，共 200 题，每个文档 20 题（18 道可回答题、2 道无答案题）。dev/test 各 100 题，每个文档各占 10 题。索引使用 `chunk_size=384`、`chunk_overlap=64`，共 15,049 个 chunk。

题目覆盖具体事实、情节因果、相近人物或事件辨析，以及需要互补证据的跨段推理，其中 125 题包含多处证据。跨 chunk 不能仅靠增加引用数量成立：每处证据都应提供回答所需的信息；`notes` 记录评分要点和容易混淆的边界。

同一事实、共享 chunk 或引用重叠的题目按组划分，dev/test 兼顾文档前、中、后部。修改问题、答案或 split 后须重新生成预测；题库更新前的报告不能作为当前题库得分。调整索引切分时，需保留原文证据并重新映射 `gold_chunk_ids`。

## 人工金标流程

1. 冻结语料并记录版本/hash，完成索引后不再边评测边重建。
2. 按真实使用需求覆盖细节、情节、辨析、跨段推理和无答案问题；同时检查章节分布，不以某种检索配置是否获胜来筛题。
3. 模板或 LLM 只能提出 `candidate`。`prepare_eval.py` 可按 source 轮询抽取现有证据，但生成的问题占位、答案空白，绝不自动成为 gold。
   为减轻 evidence-first 的措辞偏差，还应混入由看不到 chunk 的人独立编写的 query-first 问题，以及贴近领域的困难无答案题。
4. 标注者定位原文，填写最短可证实答案，把短原文抄入 canonical `gold_evidence`，并将其映射到当前索引的 `gold_chunk_ids`。
5. 把同一事实的改写、重叠 chunk、近重复段落和相同证据合并为同一 `group_id`，再按题型与可回答性在组级分层分配 split。`prepare_eval.py` 生成的 group/split 只是候选初值。
6. 第二人处理全部争议、查重并复核；加载器会拒绝同一 group 跨 dev/test。只在 dev 调 prompt、检索配置与阈值，冻结后单独运行 test。

重新切分时，`source/section/quote` 证据仍是金标，`gold_chunk_ids` 只是需要重做的派生映射。benchmark 和 `eval.py` 都固定使用 `index/`，运行前会拒绝当前索引中不存在的 gold chunk，并记录 manifest hash。

评测器统计所选 split 的全部条目。审核记录不写入 JSONL；如需追踪修改历史，使用 Git。

## Gold JSONL

每行一个对象。当前题库使用以下字段，除 `notes` 外均必需：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `group_id` | string | 同一事实/近重复证据共享；不得跨 split |
| `question` | string | 最终用户问题 |
| `split` | string | `dev` 或 `test` |
| `gold_answer` | string | 去除首尾空白后非空表示可回答；无答案题为 `""` |
| `gold_evidence` | object[] | canonical `{source,section,quote}`；可回答题至少一条 |
| `gold_chunk_ids` | string[] | canonical 证据在当前索引中的派生映射；可回答题必须非空 |
| `notes` | string | 可选标注/争议说明 |

加载器仍兼容旧格式的 `question_id`（string）、`category`（string）和 `answerable`（boolean），但当前题库不需要填写。省略时分别生成内部关联键、使用 `uncategorized`、从标准答案推导可答性；不接受其他未知字段。

`section` 可以是 `null` 或非空章节名。无答案题的 `gold_answer/gold_evidence/gold_chunk_ids` 必须全部为空。所有条目都必须有非空 `group_id`；候选脚本给出的值只是临时值，人工应在定稿前合并同事实组。

对可回答条目，评测器会检查每段规范化后的短 `quote` 都能在同 source、同章节的 gold chunk 中找到，并检查每个 gold chunk 都有对应 evidence。当前 flat `gold_chunk_ids` 不表达替代证据 qrels，因此每个必需证据原子只选一个 canonical chunk；需要“多个证据任选其一”时再升级 schema。

## Prediction JSONL

预测记录仍需 `question_id` 与题目关联，由 benchmark 使用加载器生成的内部键自动写入，无需向题库添加 ID。以下 ID 仅为示例。

纯检索记录：

```json
{"question_id":"q-001","retrieved_chunk_ids":["chunk_a"],"cited_chunk_ids":[],"answered":true,"top_score":0.82}
```

端到端记录额外包含 `answer`：

```json
{"question_id":"q-001","retrieved_chunk_ids":["chunk_a","chunk_b"],"context_chunk_ids":["chunk_a"],"cited_chunk_ids":["chunk_a"],"answered":true,"answer":"退款申请须在 7 天内提交。[chunk_a]","citation_valid":true,"top_score":0.82,"answer_correct":1,"faithful":1}
```

- `answer` 的存在标记端到端生成；没有 `answer` 的 retrieval-only 记录不计算 citation，结果为 N/A。
- `retrieved_chunk_ids` 保存完整排序；`context_chunk_ids` 保存实际传给模型的子集，未调用模型时为 `[]`。旧预测省略该字段时，以 retrieved 作为上下文。
- 引用 gate 失败时，`answer` 记录系统最终交付的“无答案”，`raw_answer` 保留被拦截的模型原文，且 `citation_valid` 为 `false`。Citation 指标仍使用原文解析出的 `cited_chunk_ids` 诊断失败。
- 每个模型实际写出的 citation 都应保留。只有同时属于 cited、gold 和 context 的 ID 才是 TP；其余引用计 FP，未正确引用的 gold chunk 计 FN。
- `top_score` 可选；阈值校准时必填，且数值必须随可回答性升高。不同配置的 score 不可互用。
- `answer_correct` 与 `faithful` 是可选人工布尔/0-1 标签，评价用户实际收到的 `answer`，不是被拦截的 `raw_answer`；必须和 `answer` 同时出现。`answer_correct` 对端到端生成的全部题评分；`faithful` 只允许用于其中实际作答的非拒答输出，各自 coverage 以对应 eligible outputs 为分母。retrieval-only 的两项 eligible cases 都是 0。不要用 LLM judge 冒充人工标签。

## Dev 校准与独立 Test

先生成 dev prediction。输出文件默认防覆盖；确需替换时才给 benchmark 传 `--force`。

```bash
uv run python benchmark.py eval/gold.jsonl \
  --config hybrid_rerank \
  --split dev \
  --output runs/hybrid_rerank.dev.jsonl \
  --stats-output runs/hybrid_rerank.dev.stats.jsonl

uv run python eval.py \
  eval/gold.jsonl \
  runs/hybrid_rerank.dev.jsonl \
  --split dev \
  --calibrate-threshold \
  > runs/hybrid_rerank.dev.calibration.json
```

从 calibration JSON 记录 threshold 并冻结。然后只跑 test：

```bash
DEV_THRESHOLD='替换为 dev threshold'
uv run --env-file .env python benchmark.py eval/gold.jsonl \
  --config hybrid_rerank \
  --split test \
  --threshold "$DEV_THRESHOLD" \
  --generate \
  --output runs/hybrid_rerank.test.generated.jsonl \
  --stats-output runs/hybrid_rerank.test.generated.stats.jsonl

uv run python eval.py \
  eval/gold.jsonl \
  runs/hybrid_rerank.test.generated.jsonl \
  --split test \
  > runs/hybrid_rerank.test.generated.eval.json
```

`eval.py --split` 使 dev-only prediction 不需要伪造 test 结果。禁止使用 test prediction 重新选 threshold。

`benchmark.py` 会检查并增量维护完整的 `index/`；如果索引不存在，会从默认 `doc/` 自动生成。需要更换切分参数或强制全量更新时显式传 `--rebuild-index`。`gold_chunk_ids` 仍必须与生成后的当前索引一致。

## 指标口径（v2）

默认 `k=10`。检索主指标按题计算后取宏平均，每题权重相同；只统计有 gold chunk 且提供排序的题，空检索计 0，无答案题不进入分母。

| 用途 | 指标 | 定义 |
| --- | --- | --- |
| **主指标：覆盖** | `recall@10` | 找回的 gold chunk 数 / 该题全部 gold chunk 数 |
| **主指标：排序** | `ndcg@10` | 所有命中的折扣增益之和 / 理想排序的折扣增益 |
| 定位漏召回 | `recall@20`、`recall@50` | 同一覆盖率公式，扩大截断深度 |
| 定位首条命中 | `hit@10`、`mrr@10` | 至少命中一条的题目比例；首条命中排名倒数的均值 |
| 引用匹配 | `citation` P/R/F1 | 按 chunk 累加 TP、FP、FN 后计算微平均 |
| 作答决策 | `answerable`、`refusal` P/R/F1 | 分别以可回答、应拒答为正类，按题计数 |
| 回答质量 | `answer_correct`、`faithful` | 人工 0/1 标签均值，同时报告标注数量和覆盖率 |

令 `G` 为该题 gold chunk 集合，相关性为二值：`DCG@k = Σ[第 i 条属于 G] / log₂(i+1)`，`IDCG@k = Σ(1/log₂(i+1)), i=1…min(k, |G|)`。`nDCG = DCG/IDCG`，因此完整排序为 1；只命中第一条不再视为全部证据已找回。定义参考 [Stanford 信息检索教材](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)。

引用使用 `TP=|cited ∩ G ∩ context|`、`FP=|cited|−TP`、`FN=|G|−TP`。可回答题被拒答且没有引用时，全部 gold 计 FN；无答案题的任何引用计 FP。它衡量 canonical chunk 匹配，不证明回答中的每条论断都获支持；未标注的等价证据也可能被判错。

人工评分以最终交付为准：`answer_correct=1` 要求覆盖标准答案的必要要点且无事实错误，无答案题须正确拒答；`faithful=1` 要求实际作答中的事实陈述都得到上下文支持。纯检索报告里的作答决策只是阈值判断，不能代替回答质量；缺少人工标签时质量均值为 `null`，不能当作 0 分或满分。

报告包含 `metric_version: 2`、split 切片和 gold/prediction/index 哈希。`retrieval_cases` 应与 `retrieval_eligible_cases` 一致；差额表示存在 `ranked:false` 的未计分题。无可评样本的均值为 `null`；P/R/F1 的零分母项为 0，完全没有引用 TP/FP/FN 时三项为 `null`。

### 比较约束

- benchmark 每路取 50 个候选，最终保留 50 条排名，生成上下文最多 10 条；stats 记录这三个上限。hybrid 使用更多检索路径，须同时比较延迟，不能称为相同总计算预算。
- 固定 gold、索引、split 和截断深度比较指标。分数微小差异需回看逐题得失，不以融合配置必须获胜为标准。
- 拒答阈值仅在 dev 按配置独立校准，空检索总是拒答；冻结后评 test。stats 另报检索/端到端延迟、内存、token 与可选成本。
- v1 的 `recall` 实际是 Hit，`ndcg` 只看首条命中，引用计数混用了题数与 chunk 数。旧分数不能与 v2 直接比较；旧预测若只保留 10/20 条，也不能据此恢复 50 条候选，需重跑预测。
