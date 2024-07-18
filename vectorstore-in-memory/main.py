import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


load_dotenv()
if __name__ == "__main__":
    pdf_path = "/home/muneeb/Desktop/Courses/LLMs/vectorstore-in-memory/IT-AUDIT-MANUAL.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    
    embeddings = OllamaEmbeddings(model="llama3")
    # vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    # vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("vectorstore-in-memory/faiss_index_react", embeddings, allow_dangerous_deserialization=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(Ollama(model="llama3"), retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(new_vectorstore.as_retriever(), combine_docs_chain)

    res = retrieval_chain.invoke({"input": "Give me the gist of IT Audit in 3 sentences"})
    print(res["answer"])

