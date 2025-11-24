from langchain_groq import ChatGroq
import os



def setup_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in environment variables")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",  # WORKING MODEL
        temperature=0.1,
        max_tokens=512
    )
    return llm
