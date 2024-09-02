import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv() ## load enivronment variables
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT") 

prompt= ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Please respond to the questions asked"),
        ("user","Question:{question}")
    ]
) 
## streamlit 

st.title("LangChain Demo with LLama2")
input_text=st.text_input("what question you have in mind")

###  Ollama LLama2 model
llm=Ollama(model="gemma:2b")

output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
