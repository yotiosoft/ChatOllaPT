# ChatOllaPT
# A simple chat application for CLI using Ollama

import dotenv
dotenv.load_dotenv()

import json

import asyncio

from langchain.schema import HumanMessage, SystemMessage, Document
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mcp_adapters.client import MultiServerMCPClient

# Read the system message from system_message.txt
with open("system_message.txt", "r") as f:
    system_message = f.read()
system_message = system_message.replace("\n", " ")

# Read the model name from model_name.txt
with open("model_name.txt", "r") as f:
    model_name = f.read().strip()

# Read the embedding model name from embedding_model_name.txt
with open("embedding_model_name.txt", "r") as f:
    embbeding_model_name = f.read().strip()

model = ChatOllama(model=model_name, temperature=0.7, streaming=True)
embeddings = OllamaEmbeddings(model=embbeding_model_name)

def fetch_documents_from_mcp(mcp_client, query: str) -> list[Document]:
        results = mcp_client.search(query, top_k=5)
        return [Document(page_content=r.content, metadata=r.metadata) for r in results]

def chat(mcp_client, prompt):
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break

        # Fetch documents from MCP
        print("🔍 Fetching from MCP...")
        docs = fetch_documents_from_mcp(mcp_client, user_input)
        if not docs:
            print("⚠ No relevant documents found.")
            continue

        # Create and run chain with streaming output
        print("AI: ", end="", flush=True)
        chain = create_chain(prompt, docs)
        for chunk in chain.stream({"question": user_input}):
            print(chunk, end="", flush=True)
        print()

# LCEL chain with retriever & streaming LLM
def create_chain(prompt, docs: list[Document]):
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    chain = (
        {
            "context": RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"])),
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

# Read the MCP config from mcp_config.json
with open("mcp_config.json", "r") as f:
    mcp_config = json.load(f)

if __name__ == "__main__":
    # Init MCP client
    mcp_client = MultiServerMCPClient(mcp_config)
    # Initialize the chat history
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(content="{input}"),
    ])

    chat(mcp_client, prompt)
