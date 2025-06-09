RAG Chatbot with Ollama and Streamlit
This project is a simple implementation of a RAG (Retrieval-Augmented Generation) chatbot using a local LLM with Ollama, and a Streamlit UI.

Setup
1. Prerequisites
Python 3.7+

Ollama installed and running. You can download it from ollama.ai.

A local LLM pulled from Ollama. This project uses llama2, but you can use any other model.

2. Installation
Clone this repository or download the files.

Install the required Python packages:

pip install streamlit langchain langchain-community beautifulsoup4 faiss-cpu

3. Knowledge Base
The knowledge_base.html file contains the information that the chatbot will use to answer questions. You can replace the content of this file with your own data. Make sure to keep the HTML structure, as the loader is configured to parse it.

4. Running the Application
Make sure Ollama is running and the llama2 model is available.

Open a terminal and navigate to the project directory.

Run the Streamlit app:

streamlit run app.py

The application will open in your browser, and you can start chatting with your bot.
