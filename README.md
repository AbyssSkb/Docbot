# AI 文档机器人
## 功能
能够读取文件夹下的文档来回答问题。
## 原理
1. 读取文档并切成块。
2. 采用多路召回的方法，使用 `GTE + BGE + BM25`，提高召回率。
3. 使用 `Reranker` 将多路召回的文本进行重排序，选出前几个文本。
4. 将文本同问题合并在一起作为大模型的输入，最终输出答案。
## 使用
1. 克隆仓库。
```bash
git clone https://github.com/AbyssSkb/Docbot
```
2. 安装所需的库。
```bash
cd Docbot
pip install -r requirements.txt
```
3. 将文档放入 `doc` 文件夹下（如果 `doc` 文件夹不存在，请先创建）。
4. 运行 `create_index.py`，将文档切成块，生成 GTE 和 BGE 所需的 embedding，最终保存成索引。
5. 修改 `main.py` 中的 `api_key`，如有需要，可以修改 `base_url` 以及 `llm_model`。
6. 运行 `main.py`。
7. 在命令行中输入你想问的问题。

## 局限
1. `GTE` 和 `BGE` 使用的是中文版本，对于英文文档的效果可能不好，但是可以通过手动切换成英文版本的方式来解决。
2. `BM25` 分词方式是通过 `jieba` 来实现，`jieba` 是一个专门用于中文分词的库，其对英文的分词效果比较基础。
3. 对于各种文档的读取使用的是 `langchain` 的 `DirectoryLoader` 接口，读取某些文档可能会出现错误。
  
