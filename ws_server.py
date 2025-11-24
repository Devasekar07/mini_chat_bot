# ws_server.py
import asyncio
import websockets
from prmpt import PROMPT  
from rag import search_chunks, chroma_setup
from llm import setup_llm
from langchain_core.messages import HumanMessage

HOST = "localhost"
PORT = 8756
_path = "/home/deva/Documents/mini_chat_bot/K_b.txt"

MEMORY=[]
MAX_MEMORY=10

def update_memory(user_msg, bot_msg):
    MEMORY.append(f"User: {user_msg}")
    MEMORY.append(f"Bot: {bot_msg}")

    # keep only recent N messages
    if len(MEMORY) > MAX_MEMORY * 2:
        del MEMORY[0:2]

def build_prompt(context: str, query: str) -> str:
    memory_text = "\n".join(MEMORY) if MEMORY else "No previous conversation."

    return f"""
{PROMPT}

Conversation Memory:
{memory_text}

Context:
{context}

Question:
{query}

Answer:
"""


model, collection = chroma_setup()
llm = setup_llm()


def get_answer(user_query: str) -> str:
    result = search_chunks(user_query, model=model, collection=collection, top=3)
    docs = result.get("documents", [[]])[0]
    context = " ".join(docs) if docs else "No context found."
    prompt = build_prompt(context, user_query)

    try:

        answer_obj = llm.generate([[ HumanMessage(content=prompt) ]])
        #answer_obj = llm.invoke([prompt])
        bot_answer = answer_obj.generations[0][0].text

        # Update memory
        update_memory(user_query, bot_answer)

        return bot_answer

    except Exception as e:
        print("[WS] LLM error:", e)
        return f"Error calling LLM: {e}"



async def handler(ws):
    async for message in ws:
        try:
            message = message.strip()
            print("[WS] Received:", message)

            if not message:
                await ws.send("No question provided.")
                continue

            answer = get_answer(message)
            await ws.send(answer)

        except Exception as e:
            print("[WS] ERROR while handling message:", e)
            try:
                await ws.send(f"Server Error: {e}")
            except Exception:
                pass


async def main():
    print(f"[WS] Server running at ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT, ping_interval=200000):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
