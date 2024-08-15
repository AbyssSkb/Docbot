from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm
import numpy as np
from rank_bm25 import BM25Okapi
import jieba

class EmbeddingModel():
    def __init__(self, model_path, index_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.index = faiss.read_index(index_path) if index_path is not None else None

    def embed_text(self, text):
        inputs = self.tokenizer(text, max_length=512, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings[0].tolist()
    
    def save_index(self, texts, save_path):
        embeddings = []
        for text in tqdm(texts):
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        faiss.write_index(self.index, save_path)

    def query(self, query, k=20):
        query_embedding = np.array(self.embed_text(query)).reshape(1, -1)
        _, indices = self.index.search(query_embedding, k)
        return indices[0]

class BM25Model():
    def __init__(self, texts):
        self.tokenized_texts = [jieba.lcut(text) for text in texts]
        self.model = BM25Okapi(self.tokenized_texts)

    def query(self, query, k=20):
        query = jieba.lcut(query)
        scores = self.model.get_scores(query)
        indices = np.argsort(scores)[-k:][::-1].tolist()
        return indices
    
class RerankingModel():
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)

    def select(self, query, context, k=10):
        self.model.eval()
        context = list(set(context))
        pairs = [(query, content) for content in context]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt').to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float().tolist()
        
        docs = [(context[i], scores[i]) for i in range(len(context))]
        docs = sorted(docs, key = lambda x: x[1], reverse=True)
        selected_contexts = [doc[0] for doc in docs][:k]
        return selected_contexts