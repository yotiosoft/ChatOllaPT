# ChatOllaPT
# A simple chat application for CLI using Ollama

import dotenv
dotenv.load_dotenv()

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

# Read the system message from system_message.txt
with open("system_message.txt", "r") as f:
    system_message = f.read()
system_message = system_message.replace("\n", " ")

# Read the model name from model_name.txt
with open("model_name.txt", "r") as f:
    model_name = f.read().strip()

model = ChatOllama(model=model_name, temperature=0.7, streaming=True)

# Initialize the chat history
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_message),
    MessagesPlaceholder(variable_name="history"),
    HumanMessage(content="{input}"),
])
chat_history = []

chain = prompt | model | StrOutputParser()

def chat():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break

        chat_history.append(HumanMessage(content=user_input))

        print("AI: ", end="", flush=True)
        response_text = ""
        for chunk in chain.stream({"input": user_input, "history": chat_history}):
            print(chunk, end="", flush=True)
            response_text += chunk

        print()  # 改行
        chat_history.append(AIMessage(content=response_text))

if __name__ == "__main__":
    print("Welcome to ChatOllaPT! Type 'exit' or 'quit' to end the chat.")
    chat()
