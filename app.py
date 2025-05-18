# ChatOllaPT
# A simple chat application for CLI using Ollama

import dotenv
dotenv.load_dotenv()

import asyncio
import mcp

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
import langchain.agents
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import langchain_core.prompts

# Read the system message from system_message.txt
with open("system_message.txt", "r") as f:
    system_message = f.read()
system_message = system_message.replace("\n", " ")

# Read the model name from model_name.txt
with open("model_name.txt", "r") as f:
    model_name = f.read().strip()

model = ChatOllama(model=model_name, temperature=0.7, streaming=True)

# Initialize the chat history
prompt = langchain_core.prompts.chat.ChatPromptTemplate.from_template(
    """Question: {input}
    Thought: Let's think step by step.
    Use one of registered tools to answer the question.
    Answer: {agent_scratchpad}"""
)
chat_history = []

chain = prompt | model | StrOutputParser()

# Set up the MCP server
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem","~/"],
)

async def chat(tools):
    agent = langchain.agents.create_tool_calling_agent(model, tools, prompt)

    # エージェントを実行
    query = "ディレクトリ ~/ にどんなファイルがありますか？"
    print(f"Query: {query}")
    output = await agent.ainvoke({"messages": query})
    print(output)

async def main():
    print("stdio_client")
    async with stdio_client(server_params) as (read, write):
        print("stdio_client connected")
        async with ClientSession(read, write) as session:
            print("ClientSession connected")
            # Initialize the connection
            await session.initialize()
            print("ClientSession initialized")

            # Get tools
            tools = await load_mcp_tools(session)

            await chat(tools)

if __name__ == "__main__":
    print("Welcome to ChatOllaPT! Type 'exit' or 'quit' to end the chat.")
    asyncio.run(main())
    #main()
