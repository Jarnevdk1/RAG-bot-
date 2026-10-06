import os 
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

#LOAD THE PDF FILES
DATA_PATH = "./data-sources"

def load_documents():
    print("Loading PDF documents from the chosen folder")
    loader = DirectoryLoader(path=DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


#SPLIT INTO CHUNKS
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function=len,
        add_start_index=True)

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


#CREATE EMBEDDING AND STORE IN VECTOR DB
CHROMA_PATH = "./chroma"

embeddings = GPT4AllEmbeddings(
    model_name="all-MiniLM-L6-v2.gguf2.f16.gguf",
    gpt4all_kwargs={'allow_download': True}
)

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        vector_db.add_documents(chunks[i:i + batch_size])
        print(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH} (vector DB).")


#EXECUTE ALL THE FUNCTIONS
def create_vector_db():
    documents = load_documents()
    if not documents:
        print(f"No PDFs found in {DATA_PATH}. Aborting, existing DB left untouched.")
        return
    chunks = split_text(documents)
    if not chunks:
        print("PDFs loaded but contained no text (scanned images?). Aborting.")
        return
    save_to_chroma(chunks)

if __name__ == "__main__":
    create_vector_db()





