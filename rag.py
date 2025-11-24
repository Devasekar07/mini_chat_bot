import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# FILE READING & TEXT SPLITTING
# -----------------------------
def load_text_from_file(path="knowledge_base.txt"):
    """Reads text from PDF or TXT."""
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() for page in reader.pages)
        return text
    elif path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format. Use .pdf or .txt")

def split_text(text, chunk_size=200, chunk_overlap=10):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# -----------------------------create_collection
# RAG SETUP
# -----------------------------
def chroma_setup(file_path="./knowledge_base.txt"):
    print("🔄 Loading text...")
    text = load_text_from_file(file_path)
    print(f"Loaded text size: {len(text)} characters")

    chunks = split_text(text)
    print(f"Total chunks created: {len(chunks)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("🔄 Creating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print("hi")

    client = chromadb.PersistentClient("./sampledb")
    collection = client.get_or_create_collection("default")
    # batch_size  = 500
    # for i in range(0,len(chunks),batch_size):
    #     batch_docs = chunks[i:i+batch_size]
    #     batch_embed = chunks[i:i+batch_size]

    ids = [str(i) for i in range(len(chunks))]
    print("🔄 Adding to Chroma collection...")
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
    

    print("✅ RAG setup complete.")
    return model, collection

# model,collection=chroma_setup()
# print(collection)
# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search_chunks(query, model, collection,top=5):
    # if model is None:
    #     model = globals()['model']
    # if collection is None:
    #     collection = globals()['collection']

    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    return results


# -----------------------------
# RUN SETUP
# -----------------------------
if __name__ == "__main__":
    # Step 1: Setup RAG
    model, collection = chroma_setup()

    # # Step 2: Ask questions
    # while True:
    #     query = input("Ask question (or type 'exit'): ")
    #     if query.lower() == "exit":
    #         break

    #     # Pass model and collection explicitly
    #     results = search_chunks(query, model=model, collection=collection,top=5)

    #     # Print top results
    #     print("Bot Answer:")
    #     print("\n".join(results['documents'][0]))

