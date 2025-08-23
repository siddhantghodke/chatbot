import os
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from datetime import datetime
import asyncio
import time

from dotenv import load_dotenv
load_dotenv() 

# Page configuration
st.set_page_config(
    page_title="iPad Wikipedia Chatbot",
    page_icon="📱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: black;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: black;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        color: black;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_chatbot():
    """Initialize the RAG chatbot"""
    try:
        # FIX: Set up the asyncio event loop for the current thread
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        with st.spinner("🔧 Initializing chatbot..."):
            # Check API key
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                st.error("❌ Google API key not found! Please set GOOGLE_API_KEY in your .env file.")
                return False
            
            # Check for content file
            wikipedia_file = "ipad_wikipedia_content.txt"
            if not os.path.exists(wikipedia_file):
                st.error(f"❌ Content file '{wikipedia_file}' not found! Please run the Wikipedia extractor first.")
                return False
            
            # Load content
            with st.spinner("📖 Loading content..."):
                loader = TextLoader(wikipedia_file, encoding='utf-8') 
                data = loader.load()
            
            # Split into chunks
            with st.spinner("✂️ Processing content..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,  
                    chunk_overlap=300,  
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                docs = text_splitter.split_documents(data)
            
            # Create vector store with error handling
            with st.spinner("🧠 Creating knowledge base..."):
                try:
                    # Set up embeddings with explicit API key
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=api_key
                    )
                    
                    vectorstore = Chroma.from_documents(
                        documents=docs, 
                        embedding=embeddings
                    )
                except Exception as e:
                    st.error(f"❌ Error creating vector store: {str(e)}")
                    return False
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 10} 
            )
            
            # Initialize LLM with explicit API key
            with st.spinner("🤖 Initializing AI model..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-pro", 
                        temperature=0.3, 
                        max_tokens=1000,
                        google_api_key=api_key
                    )
                except Exception as e:
                    st.error(f"❌ Error initializing LLM: {str(e)}")
                    return False
            
            # Create RAG chain
            system_prompt = """
            You are an expert on Apple iPads with comprehensive knowledge from Wikipedia and structured specifications.
            Use the following pieces of retrieved context to answer the question accurately.
            If you don't know the answer based on the provided context, say that you don't know.
            Provide detailed, accurate information about iPad models, specifications, release dates, and features.
            Use clear, concise language and cite specific information when possible.
            Do not provide negative statements after a consice answer like doesn't have, doesn't support, etc.
            ***Provide source in the form of link at the end.***
            
            When answering questions about specific iPad models, use the structured information provided:
            - iPad Pro (M4): Latest model with M4 chip, Ultra Retina XDR display, price 99,990 INR
            - iPad Air (M3): Latest model with M3 chip, Liquid Retina display, price 79,990 INR
            - iPad (11th gen): Latest model with A16 Bionic chip, Liquid Retina display, price 69,990 INR
            - iPad mini (A17 Pro): Latest model with A17 Pro chip, compact 8.3-inch design,  price 59,990 INR
            
            Always mention the latest models and their key specifications when relevant.
            
            Context information:
            {context}
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            st.session_state.rag_chain = rag_chain
            st.session_state.initialized = True
            
            st.success("✅ Chatbot initialized successfully!")
            return True
            
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {str(e)}")
        return False

def ask_question(question):
    """Ask a question and return the answer"""
    if not st.session_state.initialized:
        return "Chatbot not initialized. Please initialize first."
    
    try:
        response = st.session_state.rag_chain.invoke({"input": question})
        return response["answer"]
    except Exception as e:
        return f"❌ Error: {str(e)}"

def main():
    # Header
    st.title("📱 iPad Wikipedia Chatbot")
    st.markdown("Ask questions about Apple iPads using Wikipedia knowledge")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Initialize button
        if not st.session_state.initialized:
            if st.button("🚀 Initialize Chatbot", type="primary"):
                if initialize_chatbot():
                    st.rerun()
        else:
            st.success("✅ Chatbot Ready")
            if st.button("🔄 Reinitialize"):
                st.session_state.initialized = False
                st.session_state.rag_chain = None
                st.session_state.chat_history = []
                st.rerun()
        
        st.divider()
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        
        # Information
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot uses:
        - **Wikipedia content** about iPads
        - **Google Gemini AI** for responses
        - **RAG (Retrieval-Augmented Generation)** for accuracy
        """)
    
    # Main chat interface
    if not st.session_state.initialized:
        st.info("💡 Please initialize the chatbot using the sidebar button to start chatting!")
        
        # Setup guide
        with st.expander("🔧 Setup Guide", expanded=True):
            st.markdown("""
            **To get started:**
            
            1. **Install dependencies:**
               ```bash
               pip install -r requirements.txt
               ```
            
            2. **Set up Google API key:**
               - Create a `.env` file
               - Add: `GOOGLE_API_KEY=your_api_key_here`
            
            3. **Extract Wikipedia content:**
               ```bash
               python extract_wikipedia.py
               ```
            
            4. **Initialize the chatbot** using the sidebar button
            """)
    else:
        # Chat input
        user_question = st.text_input(
            "Ask a question about iPads:",
            placeholder="e.g., What is the latest iPad Pro model?",
            key="user_input"
        )
        
        # Send button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            send_button = st.button("🚀 Send", type="primary")
        
        # Process question
        if send_button and user_question:
            # Add user message to chat history (at the top)
            st.session_state.chat_history.insert(0, {
                "role": "user",
                "content": user_question,
                "timestamp": datetime.now()
            })
            
            # Get answer with streaming effect
            with st.spinner("🤔 Thinking..."):
                answer = ask_question(user_question)
            
            # Add assistant message to chat history (at the top)
            st.session_state.chat_history.insert(0, {
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now()
            })
            
            # Clear input and rerun
            st.rerun()
        
        # Display latest response above chat history
        if st.session_state.chat_history:
            latest_message = st.session_state.chat_history[0]
            if latest_message["role"] == "assistant":
                st.markdown("### 🤖 Response")
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong> {latest_message["content"]}
                    <small style="color: #666;">{latest_message["timestamp"].strftime("%H:%M:%S")}</small>
                </div>
                """, unsafe_allow_html=True)
                st.divider()
        
        # Display chat history
        st.subheader("💬 Chat History")
        
        if not st.session_state.chat_history:
            st.info("No messages yet. Ask a question to start chatting!")
        else:
            # Display chat history (skip the first message if it's the latest response)
            start_index = 1 if st.session_state.chat_history and st.session_state.chat_history[0]["role"] == "assistant" else 0
            
            for i in range(start_index, len(st.session_state.chat_history)):
                message = st.session_state.chat_history[i]
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You:</strong> {message["content"]}
                        <small style="color: #666;">{message["timestamp"].strftime("%H:%M:%S")}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong> {message["content"]}
                        <small style="color: #666;">{message["timestamp"].strftime("%H:%M:%S")}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Quick questions
        st.divider()
        st.subheader("💡 Quick Questions")
        
        quick_questions = [
            "What is the latest iPad Pro model?",
            "What chip does the iPad Air use?",
            "When was the first iPad released?",
            "What accessories are available for iPads?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(question, key=f"quick_{i}"):
                    # Add to chat history (at the top)
                    st.session_state.chat_history.insert(0, {
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now()
                    })
                    
                    # Get answer
                    with st.spinner("🤔 Thinking..."):
                        answer = ask_question(question)
                    
                    st.session_state.chat_history.insert(0, {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now()
                    })
                    
                    st.rerun()

if __name__ == "__main__":
    main()


