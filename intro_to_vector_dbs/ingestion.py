import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
if __name__ == "__main__":
   print("Ingesting...") 
   loader = TextLoader("/home/muneeb/Desktop/Courses/LLMs/intro_to_vector_dbs/mediumblog1.txt")
   document = loader.load()

   print("Splitting...")
   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
   texts = text_splitter.split_documents(document)
   print(f"Created {len(texts)} chunks")

   embeddings = OllamaEmbeddings(model="llama3",)

   print("ingesting...")
   PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
   print("Finished")