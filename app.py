# ChatOllaPT
# A simple chat application for CLI using Ollama

import dotenv
dotenv.load_dotenv()

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM

# Read the system message from system_message.txt
with open("system_message.txt", "r") as f:
    system_message = f.read()
system_message = system_message.replace("\n", " ")

# Read the model name from model_name.txt
with open("model_name.txt", "r") as f:
    model_name = f.read().strip()

model = OllamaLLM(model_name=model_name, temperature=0.7)

# Initialize the chat history
chat_history = [
    SystemMessage(content=system_message),
]

def chat():
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        # Add user message to chat history
        chat_history.append(HumanMessage(content=user_input))

        # Generate response from the model
        response = model(chat_history)

        # Add AI message to chat history
        chat_history.append(AIMessage(content=response.content))

        # Print the response
        print(f"AI: {response.content}")

if __name__ == "__main__":
    print("Welcome to ChatOllaPT!")
    print("Type 'exit' or 'quit' to end the chat.")
    chat()
