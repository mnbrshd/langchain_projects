from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

embeddings = OllamaEmbeddings(model="llama3")


def ingest_docs():
    loader = ReadTheDocsLoader(
        "docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()

    print(f"Splitting {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="langchain-doc-index")
    print("***FINISHED***")


if __name__ == "__main__":
    ingest_docs()
