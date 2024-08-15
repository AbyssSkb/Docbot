from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from model import EmbeddingModel
import json

embed1 = EmbeddingModel('thenlper/gte-large-zh')
embed2 = EmbeddingModel('BAAI/bge-large-zh-v1.5')
models = [embed1, embed2]

if __name__ == '__main__':
    document_dir = 'doc'
    loader = DirectoryLoader(document_dir)
    document = loader.load()

    chunk_size = 512
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, separators = ["\n\n", "\u3000\u3000", "\n", "。", " "])
    chunks = text_splitter.split_documents(document)
    texts = [chunk.page_content for chunk in chunks]

    embed1.save_index(texts, 'index/embed1.index')
    embed2.save_index(texts, 'index/embed2.index')

    with open('texts.json', 'w', encoding='utf-8') as f:
        json.dump(texts, f)