from langchain_text_splitters import RecursiveCharacterTextSplitter
from model import EmbeddingModel
import json
import os
from markitdown import MarkItDown
from tqdm import tqdm

embed1 = EmbeddingModel("thenlper/gte-large-zh")
embed2 = EmbeddingModel("BAAI/bge-large-zh-v1.5")

if __name__ == "__main__":
    md = MarkItDown()
    chunk_size = 512
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, separators=["\n\n", "\u3000\u3000", "\n", "。", " "]
    )
    texts = []
    for file in tqdm(os.listdir("doc")):
        content = md.convert(os.path.join("doc", file)).text_content
        chunks = text_splitter.split_text(content)
        texts += chunks

    if not os.path.exists("index"):
        os.mkdir("index")
    embed1.save_index(texts, "index/embed1.index")
    embed2.save_index(texts, "index/embed2.index")

    with open("index/texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f)
