import logging
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import Annotated, List, Tuple, Union, Literal, Sequence

from langchain.embeddings.base import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END

import lancedb
from lancedb.embeddings import get_registry
from lancedb.rerankers import CrossEncoderReranker


class CustomLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        logging.basicConfig(
            format='%(asctime)s || %(name)s || %(levelname)s || %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler(self.log_file, mode='a'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)


class GTEEmbeddings(Embeddings):
    def __init__(self, model_path='Alibaba-NLP/gte-large-en-v1.5', device='cuda', batch_size=32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        self.device = device
        self.batch_size = batch_size

    def embed_documents(self, texts):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch_texts = texts[i:i+self.batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().tolist())

        return all_embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# Creating custom retriever
# https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/custom_retriever/
class CustomLanceDBRetriever(BaseRetriever):
    _tbl = None
    _reranker = None

    @classmethod
    def initialize(cls, db_table, reranker, embedder, category: str):
        cls._tbl = db_table
        cls._reranker = reranker
        cls._embedder = embedder
        cls._category = category
        return cls()

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        if self._tbl is None or self._reranker is None:
            raise ValueError(
                "Retriever not properly initialized. Call CustomLancetDBRetriever.initialize(db_table, reranker) first.")

        SEARCH_TOP_K = 30
        RERANK_TOP_K = 10
        embedded_query = self._embedder.embed_documents([query])[0]

        ### Hybrid + reranker
        print(f"Query: {query}")
        print(f"Category: {self._category.lower()}")
        results = (self._tbl.search((embedded_query, query),
                                    query_type="hybrid",
                                    vector_column_name="embeddings")
                   .where(f"categories = '{self._category.lower()}'")  # Add category filtering
                   .limit(SEARCH_TOP_K)
                   .rerank(reranker=self._reranker)
                   .to_pandas())

        documents = []
        for _, row in results.head(RERANK_TOP_K).iterrows():
            metadata = {
                "original_content": row["original_content"],
                "summarized_content": row["summarized_content"],
                "embeddings": list(row["embeddings"]),
                "id": row["id"],
                "type": row["type"],
                "doc_name": row["doc_name"],
                "pg_no": row["pg_no"],
                # "_distance": row["_distance"]
                "_relevance_score": row["_relevance_score"]
            }
            document = Document(
                page_content=row["original_content"],
                metadata=metadata
            )
            documents.append(document)
        return documents