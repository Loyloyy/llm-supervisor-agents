from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated, List, Tuple, Union, Literal, Sequence
import lancedb
from lancedb.embeddings import get_registry
from lancedb.rerankers import CrossEncoderReranker
from utils import GTEEmbeddings, CustomLanceDBRetriever


def tavily_tool(query: str):
    """
    search the internet with Tavily web search
    :return: a list of search results
    """
    search = TavilySearchResults(max_results=5)
    return search.invoke({"query": query})


# TOOL II - API call
def get_powersizer_api_tool(workload: Literal["GenAI", "ClassicalAI"],
                            llm_B_params: int,
                            use_case: Literal["inference", "fine-tune"]) -> dict:
    """
    PowerSizer API call.
    Based on a provided AI workload, LLM params size and use case, return Powersizer results for server requirements.
    """
    # example API output
    response = {
        "server_model": None,
        "gpu_model": None,
        "num_gpus": 0,
        "latency": 0.5
    }
    if workload=="GenAI":
        if use_case=="inference":
            if llm_B_params <= 30:
                response.update({
                    "server_model": "R760XA",
                    "gpu_model": "Nvidia L40S",
                    "num_gpus": 4,
                })
            elif llm_B_params > 30:
                response.update({
                    "server_model": "XE9680",
                    "gpu_model": "Nvidia H100",
                    "num_gpus": 4,
                })
        elif use_case=="fine-tune":
            response.update({
                "server_model": "XE9680",
                "gpu_model": "Nvidia H100",
                "num_gpus": 8,
            })
    elif workload=="ClassicalAI":
        response.update({
            "server_model": "R760XA",
            "gpu_model": "Nvidia L40S",
            "num_gpus": 2,
        })
    return response


# Initialize the new embedding model
gte_embedding = GTEEmbeddings(
    model_path='/Alibaba-NLP/gte-large-en-v1.5', device="cuda:0", batch_size=8
)

# CONNECT to db
db = lancedb.connect("/location_of_vector_store")
tbl = db.open_table("table_name")
tbl.create_fts_index(['original_content'], replace=True)

# Create the reranker
RERANKER_MODEL = "/BAAI/bge-reranker-base"
ce_reranker = CrossEncoderReranker(model_name=RERANKER_MODEL, column="original_content", device="cuda:0")


def retrieve_docs_powerstore_tool(query: str) -> list:
    """Retrieve Powerstore related documents from the vector store based on the query."""
    retriever_powerstore = CustomLanceDBRetriever.initialize(tbl, ce_reranker, gte_embedding, category="PowerStore")
    docs = retriever_powerstore.get_relevant_documents(query)
    # return retriever.invoke(query)
    return [d.page_content for d in docs]


def retrieve_docs_connectrix_tool(query: str) -> list:
    """Retrieve Connectrix documents from the vector store based on the query."""
    retriever_connectrix = CustomLanceDBRetriever.initialize(tbl, ce_reranker, gte_embedding, category="Connectrix")
    docs = retriever_connectrix.get_relevant_documents(query)
    return [d.page_content for d in docs]