import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Set up the Streamlit page
st.set_page_config(page_title="iTion Solutions Chatbot", page_icon=":robot_face:", layout="wide")

st.title("iTion Solutions Chatbot")
st.caption("Your friendly assistant for information about iTion Solutions.")

# Function to create the vector store from the knowledge base
@st.cache_resource
def create_vector_store():
    """
    This function creates a FAISS vector store from the knowledge base file.
    It uses BSHTMLLoader to load the HTML file and OllamaEmbeddings for the embeddings.
    """
    try:
        loader = BSHTMLLoader("knowledge_base.html")
        data = loader.load()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(data)

        # Create the vector store
        embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Create the vector store
vector_store = create_vector_store()

if vector_store:
    # Initialize the LLM
    llm = Ollama(model="llama3.1:latest")

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Think step-by-step before providing a detailed answer.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if query := st.chat_input("Ask a question about iTion Solutions"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        # Get the response from the retrieval chain
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": query})
            answer = response.get("answer", "No answer found.")

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.warning("Could not create the vector store. Please check the knowledge base file.")

