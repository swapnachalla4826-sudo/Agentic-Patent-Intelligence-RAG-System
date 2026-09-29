"""
PatentLens - Indexing layer
Loads PDFs from Patent_Data/, indexes them in Qdrant, and exposes:
  - retriever
  - llm
The agent logic lives in agent.py.
"""

import glob
import os
import re

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# 1. Load PDFs
PDF_DIR = "Patent_Data"
pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        f"No PDF files found in '{PDF_DIR}'. Put your patent PDFs in that folder."
    )

print("PDF files found:", len(pdf_files))

all_documents = []
for pdf_file in pdf_files:
    print(f"Loading: {pdf_file}")
    all_documents.extend(PyPDFLoader(pdf_file).load())

print("Total pages loaded:", len(all_documents))


# 2. Clean text
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


for doc in all_documents:
    doc.page_content = clean_text(doc.page_content)


# 3. Chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(all_documents)
print("Total chunks:", len(chunks))


# 4. Patent ID metadata
for chunk in chunks:
    source = chunk.metadata.get("source", "unknown")
    chunk.metadata["patent_id"] = os.path.splitext(os.path.basename(source))[0]


# 5. Embeddings + Qdrant
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qdrant_client = QdrantClient(":memory:")
collection_name = "patentlens"
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_model,
)
vector_store.add_documents(chunks)
print("Documents stored in Qdrant successfully!")


# 6. Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# 7. LLM
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0, api_key=groq_api_key)

print("PatentLens pipeline loaded successfully!")
