from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# === Paths ===
PDF_PATH = "C:/Users/Sarthak singh/Downloads/cbse_courses_mock_dataset.pdf"
CHROMA_DIR = "./vectorstore/chromadb"

# === Load PDF ===
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

# === Split Text into Chunks ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
docs = text_splitter.split_documents(pages)

# === Embedding Model ===
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# === Vectorstore Save ===
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR
)
vectordb.persist()

print("✅ Vectorstore rebuilt from PDF successfully.")
