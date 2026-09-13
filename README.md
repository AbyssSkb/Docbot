# Docbot

面向中文长文档的知识库问答应用。导入文档后，通过 Web 对话检索内容、生成回答，并查看引用原文。

- **文档接入**：支持 TXT、Markdown、PDF、Office 和图片，增量构建本地索引。
- **混合检索**：BM25 与两路向量检索融合，可选 Qwen3 Reranker 重排。
- **多轮问答**：按需搜索，展示检索过程，支持回看文件、章节和引用片段。

## 快速开始

需要 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/AbyssSkb/Docbot.git
cd Docbot
uv sync
cp .env.example .env
```

在 `.env` 中配置兼容 OpenAI API 的模型服务：

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_LLM_MODEL=gpt-4o
# OPENAI_BASE_URL=https://your-provider.example/v1
```

将文档放入 `doc/`，建索引并启动：

```bash
uv run python create_index.py
uv run streamlit run main.py
```

## 文档与检索

TXT、Markdown 直接读取；PDF、Office 和图片通过 MinerU 解析。默认使用免 token 的 `flash_extract`，配置 `MINERU_API_TOKEN` 后使用 VLM 解析，并支持旧版 `.doc`、`.ppt`。TXT、Markdown 会在索引前转存为 UTF-8。

索引保存在 `index/`，默认切分长度 384、重叠 64。文档更新后重新运行建索引命令即可；完整重建使用 `--force`。`doc/` 和 `index/` 不随仓库提交。

```text
文档 → 解析与切分 → BM25 + Qwen3 / ritrieve 向量索引
                         ↓
问题 → 按需检索 → RRF 融合 → 可选重排 → LLM 回答与原文引用
```

Web 默认使用 `hybrid`，可在侧栏切换到 `hybrid_rerank`。向量模型自动使用 MPS、CUDA 或 CPU，FAISS 在 CPU 上运行。索引在本地，回答与 MinerU 解析使用配置的服务。

## 评测与开发

以下评测使用本地 200 题评测集，覆盖细节、情节和跨 chunk 综合，以 Recall@10 和 nDCG@10 为检索主指标。完整题库、原始语料、索引和运行结果保留在本地；复现结果需要相同题库、语料及索引。仓库提供[自编格式示例](eval/gold.sample.jsonl)和[评测说明](eval/README.md)，可用于准备自己的评测集。

以下结果覆盖全部 200 题。每路取 50 个候选；hybrid_rerank 对融合后的 50 条候选重排，所有配置均未启用答案生成。

| 配置 | Recall@10 | nDCG@10 | Recall@50 | 平均耗时（ms） |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 34.54% | 0.2693 | 49.13% | 43.0 |
| embed1 | 64.04% | 0.5036 | 80.29% | 61.4 |
| embed2 | 42.63% | 0.3306 | 61.33% | 41.7 |
| dual_dense | 56.38% | 0.4404 | 79.13% | 102.3 |
| hybrid | 63.42% | 0.4951 | 79.83% | 146.0 |
| hybrid_rerank | 75.04% | 0.6529 | 79.83% | 25884.5 |

hybrid_rerank 的前 10 条证据覆盖和排序得分最高，Recall@10 比 hybrid 高 11.63 个百分点，但平均每题耗时约 25.88 秒。重排不扩充候选池，因此两者 Recall@50 相同。

向量模型和 Qwen3-Reranker-0.6B 使用 MPS，FAISS 使用 CPU。指标由合并后的逐题预测重新计算；平均耗时按两批运行的题数加权，不含模型加载，反映本机 16 GB 内存环境下的实际负载。

题库 SHA-256：`958d33e7e42b`；索引 manifest：`a56e2107eadc`。

运行单元测试：

```bash
uv run python -m unittest discover -s tests
```

| 文件 | 用途 |
| --- | --- |
| `main.py` | Web 界面 |
| `create_index.py` | 文档解析与索引构建 |
| `pipeline.py` | 问答流程与引用校验 |
| `model.py` | 检索、融合与重排 |
| `benchmark.py` / `eval.py` | 评测运行与指标计算 |
| `eval/` | 公开格式示例与评测说明 |

## License

[MIT](LICENSE)
