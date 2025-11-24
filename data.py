# prepare_data.py

import os
import fitz # type: ignore
import pickle
import chromadb # type: ignore
from chromadb.config import Settings # type: ignore
from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]
from langchain_text_splitters import RecursiveCharacterTextSplitter # pyright: ignore[reportMissingImports]
from rag import search_chunks

# ----------- GLOBAL VARIABLES (Lazy Loading) --------------
model = None
collection = None
client = None
chunks = None

DB_PATH = "./chroma_db"
COLLECTION_NAME = "Rag_collection"
TEXT_PATH = "/home/deva/Documents/mini_chat_bot/K_b.txt"
CHUNKS_FILE = "chunks.pkl"


# ----------- EXTRACT TEXT ----------------
def extract_text(path):
    doc = fitz.open(path)
    txt = ""
    for page in doc:
        txt += page.get_text() + "\n"
    return txt


# ----------- CHUNKING ----------------
def split_text(text, chunk_size=200, chunk_overlap=0):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


# ----------- PREPARE CHROMA DB ----------------
def prepare_chroma():
    """Run once: extract, chunk, embed, save to ChromaDB."""
    print("[RAG] Extracting text...")
    text = extract_text(TEXT_PATH)

    print("[RAG] Splitting text into chunks...")
    chunks = split_text(text)

    print("[RAG] Saving chunks to file...")
    pickle.dump(chunks, open(CHUNKS_FILE, "wb"))

    print("[RAG] Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("[RAG] Generating embeddings...")
    embeddings = model.encode(chunks)

    print("[RAG] Creating ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Clear old data
    try:
        collection.delete(ids=[str(i) for i in range(len(chunks))])
    except:
        pass

    print("[RAG] Storing data in ChromaDB...")
    ids = [str(i) for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    print("[RAG] ChromaDB preparation complete.")


# ----------- LAZY LOADING FOR INFERENCE ----------------
def load_for_inference():
    """Load model, chunks, and Chroma collection only once."""
    global model, chunks, client, collection

    if model is None:
        print("[RAG] Loading model for inference...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

    if chunks is None:
        print("[RAG] Loading text chunks...")
        chunks = pickle.load(open(CHUNKS_FILE, "rb"))

    if client is None:
        print("[RAG] Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=DB_PATH)

    if collection is None:
        print("[RAG] Loading collection...")
        collection = client.get_or_create_collection(name=COLLECTION_NAME)


# ----------- MAIN SEARCH FUNCTION ----------------
from prmpt import PROMPT

def get_answer(query, llm):
    if not query:
        return "No question provided."

    # Load inference components if not loaded
    load_for_inference()

    # Do RAG search
    chunks = search_chunks(query, model, collection)
    context = " ".join(chunks["documents"][0])  # ensure this is not empty

    prompt = f"""
    You are a knowledgeable Kollywood assistant.
    Answer the user question only based on the context.
    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    try:
        answer = llm(prompt)
        return answer if answer else "No response returned from LLM."
    except Exception as e:
        print("Error calling LLM:", e)
        return f"Error calling LLM: {e}"



