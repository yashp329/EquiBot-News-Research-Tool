import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("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")

# Validate API key
if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key is missing! Please add it to your `.env` file.")
    st.stop()

# Streamlit UI
st.title("📰 News Research Tool 📈")
st.sidebar.title("🔗 Enter News URLs")

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)  # Add only non-empty URLs

# Process button
process_url_clicked = st.sidebar.button("🚀 Process URLs")

# File paths for FAISS storage
FAISS_INDEX_PATH = "faiss_index"

# Load OpenAI model
llm = OpenAI(temperature=0.3, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

if process_url_clicked:
    if not urls:
        st.warning("⚠️ Please enter at least one URL before processing.")
    else:
        st.write("🔄 Fetching and processing articles...")

        # ✅ Load articles using Playwright
        loader = WebBaseLoader(urls)
        docs = loader.load()

        if not docs:
            st.error("⚠️ No valid data retrieved from the URLs. Please check and try again.")
            st.stop()

        # ✅ Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_chunks = text_splitter.split_documents(docs)

        # ✅ Generate embeddings and store in FAISS
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(doc_chunks, embeddings)

        # ✅ Save FAISS index
        vectorstore.save_local(FAISS_INDEX_PATH)
        st.success("✅ Data indexed successfully!")

# Query Input
query = st.text_input("🔍 Ask a question based on the articles:")

if query:
    # ✅ Load FAISS index if available
    if os.path.exists(FAISS_INDEX_PATH):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

        # ✅ Setup RetrievalQA
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # ✅ Get answer
        st.write("🤖 Generating answer...")
        result = chain.invoke({"query": query})

        # ✅ Display answer
        st.header("📢 Answer:")
        st.write(result["result"])
    else:
        st.error("⚠️ No FAISS index found. Please process URLs first!")
