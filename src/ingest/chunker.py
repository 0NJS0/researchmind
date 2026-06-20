from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs):

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )

    return splitter.split_documents(
        docs
    )