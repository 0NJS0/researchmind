from langchain_community.embeddings import HuggingFaceBgeEmbeddings


_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
    return _embeddings