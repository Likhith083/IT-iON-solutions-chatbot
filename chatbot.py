import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import json
import time

# Set up the Streamlit page with a more professional look
st.set_page_config(
    page_title="iTion Solutions Chatbot",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: #f8f9fa;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title("Settings")
    temperature = st.slider("Response Temperature", 0.0, 1.0, 0.7, help="Higher values make responses more creative but less focused")
    max_tokens = st.slider("Max Response Length", 100, 1000, 500, help="Maximum length of the response")
    clear_chat = st.button("Clear Chat History")

# Main title with better styling
st.title("🤖 iTion Solutions Chatbot")
st.caption("Your intelligent assistant for all things iTion Solutions")

def load_knowledge_base():
    """
    Loads and processes the JSON knowledge base into a format suitable for the vector store.
    """
    try:
        with open('knowledge_base.json', 'r', encoding='utf-8') as f:
            kb = json.load(f)
        
        # Convert JSON structure into a list of text chunks
        documents = []
        
        # Company Info
        company_info = kb['company_info']
        documents.append(f"Company Name: {company_info['name']}\nDescription: {company_info['description']}\nVision: {company_info['vision']}\nMission: {company_info['mission']}")
        
        # Services
        services = kb['services']
        service_text = f"Services Overview: {services['description']}\n\n"
        for service in services['items']:
            service_text += f"Service: {service['name']}\nDescription: {service['description']}\nCategory: {service['category']}\n\n"
        documents.append(service_text)
        
        # Products
        products = kb['products']
        product_text = f"Products Overview: {products['description']}\n\n"
        for product in products['items']:
            product_text += f"Product: {product['name']}\nCategory: {product['category']}\n\n"
        documents.append(product_text)
        
        # Features
        features = kb['features']
        feature_text = f"Features Overview: {features['description']}\n\n"
        for feature in features['items']:
            feature_text += f"Feature: {feature['name']}\nDescription: {feature['description']}\nCategory: {feature['category']}\n\n"
        documents.append(feature_text)
        
        # Careers
        careers = kb['careers']
        career_text = f"Career Contact: {careers['contact_email']}\n\n"
        for position in careers['positions']:
            career_text += f"Position: {position['title']}\nCode: {position['code']}\nDescription: {position['description']}\n"
            if 'requirements' in position:
                career_text += "Requirements:\n" + "\n".join(f"- {req}" for req in position['requirements']) + "\n"
            career_text += f"Category: {position['category']}\n\n"
        documents.append(career_text)
        
        # Contact Information
        contact = kb['contact']
        contact_text = f"General Contact:\nPhone: {contact['general']['phone']}\nHours: {contact['general']['hours']}\nEmail: {contact['general']['email']}\n\n"
        for office in contact['offices']:
            contact_text += f"Office: {office['name']}\nCompany: {office['company']}\nAddress: {office['address']}\n"
            if 'city' in office:
                contact_text += f"City: {office['city']}\n"
            if 'state' in office:
                contact_text += f"State: {office['state']}\n"
            contact_text += f"ZIP: {office['zip']}\nPhone: {office['phone']}\n\n"
        documents.append(contact_text)
        
        return documents
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        return None

# Function to create the vector store from the knowledge base
@st.cache_resource
def create_vector_store():
    """
    Creates a FAISS vector store from the knowledge base.
    """
    try:
        with st.spinner("Loading knowledge base..."):
            documents = load_knowledge_base()
            if not documents:
                return None

            # Split the text into chunks with optimized parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.create_documents(documents)

            # Create the vector store with better embeddings
            embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
            vector_store = FAISS.from_documents(chunks, embeddings)
            st.success("Knowledge base loaded successfully!")
            return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        st.info("Please ensure the knowledge_base.json file exists and is properly formatted.")
        return None

# Create the vector store
vector_store = create_vector_store()

if vector_store:
    # Initialize the LLM with configurable parameters
    llm = Ollama(
        model="llama3.1:latest",
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Enhanced prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant for iTion Solutions. Your role is to provide accurate, helpful, and professional responses.
    
    Guidelines:
    1. Base your answers ONLY on the provided context
    2. If you don't know the answer, politely say so
    3. Keep responses clear and concise
    4. Use bullet points for multiple items
    5. Maintain a professional tone
    6. If the question is unclear, ask for clarification
    7. When discussing products or services, mention their categories
    8. For career-related questions, include position codes and requirements
    9. For contact information, provide complete details including office locations

    <context>
    {context}
    </context>

    Question: {input}

    Please provide a well-structured response:
    """)

    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain with better context handling
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
    )
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Clear chat history if requested
    if clear_chat:
        st.session_state.messages = []
        st.rerun()

    # Display chat messages from history with better formatting
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input with better UI
    if query := st.chat_input("Ask a question about iTion Solutions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Get the response with loading animation
        with st.spinner("Processing your question..."):
            try:
                start_time = time.time()
                response = retrieval_chain.invoke({"input": query})
                processing_time = time.time() - start_time
                
                answer = response.get("answer", "I apologize, but I couldn't find a relevant answer.")
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    st.caption(f"Response time: {processing_time:.2f} seconds")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")
                st.info("Please try rephrasing your question or try again later.")
else:
    st.error("Could not initialize the chatbot. Please check the following:")
    st.markdown("""
    1. Ensure the knowledge_base.json file exists
    2. Check if Ollama is running and accessible
    3. Verify that the required models are installed
    4. Check your internet connection
    """)

