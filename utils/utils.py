
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Milvus, Pinecone, Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from models.llm_model import get_openai_model, get_huggingfacehub
import pinecone
import streamlit as st
from config.templates import bot_template, user_template
from config.keys import Keys
from PyPDF2 import PdfReader


def extract_text_from_PDF(files):
    # 参考官网链接：https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
    # 加载多个PDF文件
    text = ""
    for pdf in files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_content_into_chunks(text):
    # 参考官网链接：https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/character_text_splitter
    text_spliter = CharacterTextSplitter(separator="\n",
                                         chunk_size=500,
                                         chunk_overlap=80,
                                         length_function=len)
    chunks = text_spliter.split_text(text)
    return chunks

def save_chunks_into_vectorstore(content_chunks, embedding_model):
    # 参考官网链接：https://python.langchain.com/docs/modules/data_connection/vectorstores/
    # ① FAISS
    # pip install faiss-gpu (如果没有GPU，那么 pip install faiss-cpu)
    #vectorstore = FAISS.from_texts(texts=content_chunks,
    #                                   embedding=embedding_model)

    # ② Pinecone
    # 官网链接：https://python.langchain.com/docs/integrations/vectorstores/pinecone
    # Pinecone官网链接：https://docs.pinecone.io/docs/quickstart
    # pip install pinecone-client==2.2.2
    # 初始化
    pinecone.init(api_key=Keys.PINECONE_KEY, environment="asia-southeast1-gcp")
    # 创建索引
    index_name = "pinecone-chatbot-demo"
    # 检查索引是否存在，如果不存在，则创建
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name,
                              metric="cosine",
                              dimension=1536)
    vectorstore = Pinecone.from_texts(texts=content_chunks,
                                          embedding=embedding_model,
                                          index_name=index_name)

    # ③ Milvus, pip install pymilvus
    # 官网链接：https://python.langchain.com/docs/integrations/vectorstores/milvus
    # vectorstore = Milvus.from_texts(texts=content_chunks,
    #                                     embedding=embedding_model,
    #                                     connection_args={"host": "localhost", "port": "19530"},
    # )

    return vectorstore

def get_chat_chain(vector_store):
    # ① 获取 LLM model
    llm = get_openai_model()
    #llm = get_huggingfacehub(model_name="google/flan-t5-xxl")

    # ② 存储历史记录
    # 参考官网链接：https://python.langchain.com/docs/use_cases/question_answering/how_to/chat_vector_db
    # 用于缓存或者保存对话历史记录的对象
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    # ③ 对话链
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def process_user_input(user_input):
    if st.session_state.conversation is not None:
        # 调用函数st.session_state.conversation，并把用户输入的内容作为一个问题传入，返回响应。
        response = st.session_state.conversation({'question': user_input})
        # session状态是Streamlit中的一个特性，允许在用户的多个请求之间保存数据。
        st.session_state.chat_history = response['chat_history']
        # 显示聊天记录
        # chat_history : 一个包含之前聊天记录的列表
        for i, message in enumerate(st.session_state.chat_history):
            # 用户输入
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True) # unsafe_allow_html=True表示允许HTML内容被渲染
            else:
                # 机器人响应
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
