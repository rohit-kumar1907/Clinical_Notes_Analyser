# Example URL: https://www.medicalnewstoday.com/articles/315123#stages
# Example query: summaries the article. highlights the key points


import os
import streamlit as st
# import pickle5
import time
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from inference import get_ner
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings

#take environment variable(key) from .env
load_dotenv()
#Create the title of UI
st.title("Clinical Notes Analyser")

#In the UI we'll have space to enter 3 URL and PDFs and after that we can process
st.sidebar.title("Clinical Notes sources:")

option = st.sidebar.selectbox(
     'Please Select type of Clinical note?',
     ('Text file', 'PDF file','Web-page '))

if option =='Text file':
     #Input text file
     txt_file = st.sidebar.file_uploader("Upload your Text file",type='txt', help= 'Text file only')
     #load the file using textloader
     if txt_file:
          loader = TextLoader(file_path=txt_file.name)
          data = loader.load()
elif option == 'PDF file':
     # Input pdf file
     pdf_file = st.sidebar.file_uploader("Upload your PDF file",type='pdf',help="PDF files only")
     #load the file using pdfloader
     if pdf_file:
          loader = UnstructuredPDFLoader(file_path=pdf_file.name)
          data = loader.load()
else:
     #input the url
     urls=[]
     url = st.sidebar.text_input("Enter the URL of the clinical note: ")
     if url:
          urls.append(url)
          loader = UnstructuredURLLoader(urls=urls)
          data = loader.load() 

# To show progress when we'll hit process, as progess bar will appear
# for this we'll use main place holder which will be a UI element before any stage
main_place_holder = st.empty()

#Below text input there'll be button to process and we'll store the response
process_clicked = st.sidebar.button("Process Notes")

#UI element before data loading
main_place_holder.text("Data loading...started...✅✅✅")

#Create LLM
llm = OpenAI(temperature=0.7, max_tokens=1000)

#file name of FAISS index vector database
vectordb_file_path = "faiss_index"

# Create Embeddings
#1. OpenAI embeddings
# embeddings = OpenAIEmbeddings()
#2. Huggingface embeddings
embeddings = HuggingFaceEmbeddings(model_name="medical-ner-proj/bert-medical-ner-proj", model_kwargs={'device': 'cpu'})

if process_clicked:
     #2. Split the data
     text_splitter = RecursiveCharacterTextSplitter(
          separators=['\n\n','\n','.',','],
          chunk_size=1000
          )
     #UI element before splitting
     main_place_holder.text("Data Splitting...started...✅✅✅")
     #text splitter will take dcuments and return individual chunks
     docs = text_splitter.split_documents(data)

     #extract the medical entities from 'docs'
     entities_docs={}
     for doc in docs:
          entities =get_ner(doc.page_content)
          for entity in entities:
               key = entity[0]
               val = entity[1]
               entities_docs[key] = val
     st.header("Medical Entitities: ")
     #UI element before Medical entity recognition 
     main_place_holder.text("Medical Entities Recognition...started...✅✅✅")
     st.write(entities_docs)

     #UI element before Embedding vector 
     main_place_holder.text("Embedding vector ...Building...✅✅✅")
     # Create a FAISS instance for vector database from 'docs'
     vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
     # Save vector database locally
     vectordb.save_local(vectordb_file_path)

#We'll enter a question box
query = main_place_holder.text_input("Query Request")
if query:
    #loading the vector database
    if os.path.exists(vectordb_file_path):
          # Load the vector database from the local folder
          vectordb = FAISS.load_local(vectordb_file_path, embeddings)
          #create retrieval QA chain
          chain = RetrievalQAWithSourcesChain.from_llm(
               llm = llm,
               retriever = vectordb.as_retriever()
               )
          result=chain({"question":query},return_only_outputs=True)#It'll be dictionary with 2 elements
          #{"answer":"", "sources":[]}
          st.header("Answer")
          st.write(result["answer"])
          #Display sources if available
          sources = result['sources']
          if sources:
               st.subheader("Sources:")
               sources_list = sources.split("\n") #split the sources by new line
               for source in sources_list:
                    st.write(source)


