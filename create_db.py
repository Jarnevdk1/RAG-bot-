import os 
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

#LOAD THE PDF FILES
DATA_PATH = "./data-sources"

def load_documents():
    print("Loading PDF documents from the chosen folder")

    loader = DirectoryLoader(path=DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents


#SPLIT INTO CHUNKS
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 200,
        length_function=len,
        add_start_index=True)

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


#CREATE EMBEDDING AND STORE IN VECTOR DB
CHROMA_PATH = "./chroma"

embeddings = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    gpt4all_kwargs={'allow_download': 'True'}
)

def save_to_chroma(chunks):


    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Saved {len(chunks)} to {CHROMA_PATH} (vector DB).")


#EXECUTE ALL THE FUNCTIONS
def create_vector_db():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    create_vector_db()





