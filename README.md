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

评测集包含 200 道题，覆盖细节、情节、跨 chunk 综合和无答案问题，dev/test 各 100 道。以证据召回率和 nDCG 为检索主指标，同时评估引用、拒答与人工答案质量。运行方式与指标口径见 [评测文档](eval/README.md)。

Dev 共 100 题，检索指标统计其中 90 道可回答题。每路取 50 个候选，使用 v2 指标；未启用答案生成或重排。

| 配置 | Recall@10 | nDCG@10 | Recall@50 | 平均耗时（ms） |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 41.39% | 0.3490 | 53.52% | 46.3 |
| embed1 | 71.15% | 0.5920 | 84.15% | 54.9 |
| embed2 | 47.91% | 0.4000 | 66.83% | 35.5 |
| dual_dense | 62.63% | 0.5223 | 84.24% | 91.4 |
| hybrid | 70.69% | 0.5791 | 85.81% | 144.9 |

embed1 的前 10 条证据覆盖和排序得分最高；hybrid 在前 50 条的证据召回率最高。向量模型使用 MPS、FAISS 使用 CPU；耗时为 100 题均值，不含模型加载。

题库 SHA-256：`e66b2a85d3d3`；索引 manifest：`a56e2107eadc`。

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
| `eval/` | 题库与评测说明 |

## License

[MIT](LICENSE)
