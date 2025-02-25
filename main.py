import os
import streamlit as st
import time
import asyncio
import sys
import requests
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document  # ✅ Import Document class
from dotenv import load_dotenv

# ✅ Fix Windows async issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ✅ Load API Key
load_dotenv()

# ✅ Streamlit UI
st.title("EquiBot : News Research Tool 📈")
st.sidebar.title("Enter News Article URLs")

# ✅ Collect URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")
faiss_index_path = "faiss_index"

main_placeholder = st.empty()
llm = ChatOpenAI(temperature=0.3, model_name="gpt-4")

def extract_text_from_url(url):
    """ Extracts main article text from a URL using BeautifulSoup """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract article text
            article_text = "\n".join([p.get_text() for p in soup.find_all("p")])
            return article_text.strip() if article_text else None
        else:
            print(f"⚠️ Unable to fetch {url}, Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error extracting {url}: {str(e)}")
        return None

if process_url_clicked:
    valid_texts = []
    
    # ✅ Extract text from URLs
    for url in urls:
        if url:
            text = extract_text_from_url(url)
            if text:
                valid_texts.append({"text": text, "source": url})
                print(f"✅ Extracted text from: {url}")
            else:
                print(f"⚠️ No readable content from {url}")

    if not valid_texts:
        st.error("❌ No valid article content found.")
    else:
        # ✅ Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        docs = []
        for item in valid_texts:
            split_texts = text_splitter.split_text(item["text"])
            for txt in split_texts:
                docs.append(Document(page_content=txt, metadata={"source": item["source"]}))  # ✅ Fix: Convert to Document

        # ✅ Store in FAISS
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        print(f"✅ FAISS Indexed {len(vectorstore_openai.docstore._dict)} chunks.")

        # ✅ Save FAISS index
        vectorstore_openai.save_local(faiss_index_path)
        main_placeholder.text("FAISS index stored successfully! ✅")

query = st.text_input("Enter your question:")
if query:
    if os.path.exists(faiss_index_path):
        # ✅ Load FAISS
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        # ✅ Retrieve context
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # ✅ Debug retrieval
        retrieved_docs = retriever.get_relevant_documents(query)
        if len(retrieved_docs) == 0:
            st.error("❌ No relevant information found.")
        else:
            print(f"🔍 Retrieved {len(retrieved_docs)} documents.")

            # ✅ Answer using RetrievalQA
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
            result = chain({"query": query})

            # ✅ Display answer
            st.header("Answer:")
            st.write(result["result"])

            # ✅ Show sources
            st.subheader("Sources:")
            for doc in retrieved_docs:
                st.write(doc.metadata.get("source", "Unknown Source"))
