# AI 文档机器人
## 功能
能够读取文件夹下的文档来回答问题
## 原理
1. 读取文档并切成块
2. 采用多路召回的方法，使用 `GTE + BGE + BM25`，提高召回率
3. 使用 `Reranker` 将多路召回的文本进行重排序，选出前几个文本
4. 将文本同问题合并在一起作为大模型的输入，最终输出答案
## 使用
1. 克隆仓库
```bash
git clone https://github.com/AbyssSkb/Docbot
```
1. 安装所需的库
```bash
cd Docbot
pip install -r requirements.txt
```
1. 将文档放入 `doc` 文件夹下（如果 `doc` 文件夹不存在，请先创建）
2. 运行 `create_index.py`，将文档切成块，生成 GTE 和 BGE 所需的 embedding，最终保存成索引
3. 修改 `main.py` 中的 `api_key`，如有需要，可以修改 `base_url` 以及 `llm_model`
4. 运行 `main.py`
5. 在命令行中输入你想问的问题
  
