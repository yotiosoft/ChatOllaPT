import dotenv
dotenv.load_dotenv()

import asyncio

import json
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig

# Load settings
with open("system_message.txt", "r") as f:
    system_message = f.read().replace("\n", " ")

with open("model_name.txt", "r") as f:
    model_name = f.read().strip()

with open("embedding_model_name.txt", "r") as f:
    embedding_model_name = f.read().strip()

with open("mcp_config.json", "r") as f:
    mcp_config = json.load(f)

# Initialize model and embeddings
llm = ChatOllama(model=model_name, temperature=0.7, streaming=True)

# Setup MCP client
mcp_client = MultiServerMCPClient(mcp_config)

# Tool definition using @tool decorator
client = MultiServerMCPClient(
    {
        "fetch": {
            "command": "uvx",
            "args": ["mcp-server-fetch"]
        }
    }
)

# Interactive loop
def chat():
    config = RunnableConfig(configurable={"thread_id": "chat-1"})
    print("You can now start chatting. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat. Goodbye!")
            break
        print("AI: ", end="", flush=True)
        for output in app.stream({"input": user_input}, config=config):
            content = output.get("output", "")
            print(content, end="", flush=True)
        print()
        
async def main():
    tools = await client.get_tools()

    # Create the agent using LangGraph
    agent = create_react_agent(llm, tools)
    app = agent.compile()

    chat()

if __name__ == "__main__":
    asyncio.run(main())
