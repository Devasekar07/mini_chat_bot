# ws_client.py
import asyncio
import websockets

async def main():
    uri = "ws://localhost:8756"  # must match ws_server PORT
    try:
        async with websockets.connect(uri, ping_interval=200000) as socket:
            print("Connected to RAG WebSocket server.")
            while True:
                msg = input("\nAsk question (or type 'exit'): ").strip()
                if not msg:
                    continue
                if msg.lower() == "exit":
                    print("Closing connection.")
                    break

                await socket.send(msg)
                reply = await socket.recv()
                print("Bot:", reply)
    except ConnectionRefusedError:
        print("Could not connect to the server. Is ws_server.py running?")
    except Exception as e:
        print("Client error:", e)


if __name__ == "__main__":
    asyncio.run(main())
