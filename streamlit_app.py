import streamlit as st
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA


st.set_page_config(layout="wide", page_title="RAG")

def run_temp_dir():
    temp_dir = "temp_files"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        os.makedirs("temp_files")
    else:
        os.makedirs("temp_files")

def check_if_pdf(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        st.write("PDF file uploaded successfully")
    else:
        st.write("Uploaded file is not a PDF. Please choose a PDF file.")


template="""Use the following pieces of information to answer the user's question.
If you dont know the answer just say you don't know, don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer
"""

def main():
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    if uploaded_file is not None:
        #create a temporary dir
        run_temp_dir()

        # Save the uploaded file temporarily
        with open(os.path.join("temp_files", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Get the file path
        file_path = os.path.join("temp_files", uploaded_file.name)

        # Use the file path with PyPDFLoader
        loader = PyPDFLoader(file_path)

        st.write("Loader is done")

        documents=loader.load()

        #Step-2: Chunkin operation
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks=text_splitter.split_documents(documents)

        #Step-3: Load the Embedding Model
        embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                 model_kwargs={'device':'cpu'})
        
        #step-4: Use Vector DataBase to store the emeddings
        vector_store=FAISS.from_documents(text_chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={'k': 2})

        #Step-5: Initialize llm
        llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                        model_type="llama",
                        config={'max_new_tokens':128,
                                'temperature':0.01})
        
        #step-6: Use prompt template foruser query: Perform Q&A, using template from helper.py
        qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])

        #Step-7: creating RetrievalQA chain 
        chain = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=retriever,
                                   return_source_documents=False,
                                   chain_type_kwargs={'prompt': qa_prompt})
        

        query = st.text_input("Ask yout question")


        result=chain({'query':query })

        st.write(f"Answer:{result['result']}")



if __name__ == "__main__":
    main()