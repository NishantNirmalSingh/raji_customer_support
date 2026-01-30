from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


# Load documents for uploading data 

with open("data/raji_docs.txt", "r", encoding="utf-8") as f:
    text = f.read()


# Split into chunks just like tokenization

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_text(text)

print(f"Total chunks created: {len(chunks)}")

# Create embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Build FAISS index
vectorstore = FAISS.from_texts(chunks, embeddings)

# Save index
vectorstore.save_local("faiss_index")

print("FAISS index built and saved successfully.")
