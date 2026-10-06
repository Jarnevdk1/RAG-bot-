from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


