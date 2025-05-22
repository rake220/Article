#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os
import pickle
import time
import numpy as np
import streamlit as st

try:
    import faiss
except ImportError:
    os.system("pip install faiss-cpu")
    import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Constants
FILE_PATH = "faiss_store.pkl"

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# User Input
urls = st.sidebar.text_area("Enter article URLs (one per line)").split("\n")
process_url_clicked = st.sidebar.button("Process URLs")

# Load Hugging Face LLM
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Load Sentence Transformer model for embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Process URLs
def process_urls(urls, embedding_model):
    if not any(urls):
        st.error("Please enter at least one valid URL.")
        return

    try:
        st.text("Fetching articles... ⏳")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        if not data:
            st.error("No content could be loaded from the URLs. Please verify they are accessible.")
            return

        st.text("Splitting text into chunks... ⏳")
        text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
        docs = text_splitter.split_documents(data)

        if not docs:
            st.error("No documents found after splitting. Check your URLs or content format.")
            return

        st.text("Generating embeddings... ⏳")
        embeddings = np.array([embedding_model.encode(doc.page_content) for doc in docs])

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        with open(FILE_PATH, "wb") as f:
            pickle.dump((index, docs), f)

        st.success("Processing complete! ✅")

    except ModuleNotFoundError:
        st.error("Missing dependencies. Please run:")
        st.code("pip install unstructured pdfminer.six unstructured-inference unstructured-pytesseract")
    except Exception as e:
        st.error(f"Error processing URLs: {e}")

# Answer Questions
def query_llm(question, generator, embedding_model):
    if not os.path.exists(FILE_PATH):
        return "No data available. Please process URLs first."

    try:
        with open(FILE_PATH, "rb") as f:
            index, docs = pickle.load(f)

        question_embedding = embedding_model.encode(question).reshape(1, -1)
        D, I = index.search(question_embedding, k=3)

        relevant_texts = " ".join([docs[i].page_content for i in I[0]])

        prompt = f"""You are an assistant that answers questions strictly using the provided context. 
If the answer is not found in the context, respond with "Not found in the article."

Context:
{relevant_texts}

Question:
{question}

Answer:"""

        response = generator(prompt, max_length=300, do_sample=False)
        return response[0]['generated_text']

    except Exception as e:
        return f"Error: {str(e)}"

# Load models
generator = load_llm()
embedding_model = load_embedding_model()

# Process URLs
if process_url_clicked:
    process_urls(urls, embedding_model)

# Question Input
query = st.text_input("Ask a question:")
if query:
    answer = query_llm(query, generator, embedding_model)
    st.subheader("Answer:")
    st.write(answer)


# In[ ]:





# In[ ]:





# In[ ]:




