import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader


st.set_page_config(page_title="LangChain: Summarize Text from YT or Website",page_icon="🦜")

st.title("Langchain : Summarize text from YT or Website")
st.subheader("summarize URL")



with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")
    
generic_url= st.text_input("URL",label_visibility="collapsed")

## Gemma model using Groq api
if groq_api_key:
    llm = ChatGroq(model="Gemma-7b-It", api_key=groq_api_key)

prompt_template="""
Provide the summary of the following content in 300 words:
Content:{text}
"""

prompt= PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information")
    elif not validators.url(generic_url):
        st.error("please enter a valid URL.It can may be a YT video url or website url")
        
    else:
        try:
            with st.spinner("Waiting..."):
                ##load the data from website or youtube
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,header={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                
                data=loader.load()
                
                ## Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                summary=chain.run(data)
                st.success(summary)
        except Exception as e:
            st.error(f"Exception:{e}")
            