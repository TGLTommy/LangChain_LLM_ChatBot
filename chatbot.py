# coding: utf-8
# Author: 唐国梁Tommy
# Date: 2023-08-06

import streamlit as st
from utils.utils import extract_text_from_PDF, split_content_into_chunks
from utils.utils import save_chunks_into_vectorstore, get_chat_chain, process_user_input
from models.embedding_model import get_openaiEmbedding_model, get_huggingfaceEmbedding_model


def main():
    # 配置界面
    st.set_page_config(page_title="基于PDF文档的 QA ChatBot",
                       page_icon=":robot:")

    st.header("基于LangChain+LLM实现QA ChatBot")

    # 参考官网链接：https://github.com/hwchase17/langchain-streamlit-template/blob/master/main.py
    # 初始化
    # session_state是Streamlit提供的用于存储会话状态的功能
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # 1. 提供用户输入文本框
    user_input = st.text_input("基于上传的PDF文档，请输入你的提问: ")
    # 处理用户输入，并返回响应结果
    if user_input:
        process_user_input(user_input)

    with st.sidebar:
        # 2. 设置子标题
        st.subheader("你的PDF文档")
        # 3. 上传文档
        files = st.file_uploader("上传PDF文档，然后点击'提交并处理'",
                                 accept_multiple_files=True)
        if st.button("提交并处理"):
            with st.spinner("请等待，处理中..."):
                # 4. 获取PDF文档内容（文本）
                texts = extract_text_from_PDF(files)
                # 5. 将获取到的文档内容进行切分
                content_chunks = split_content_into_chunks(texts)
                #st.write(content_chunks)
                # 6. 对每个chunk计算embedding，并存入到向量数据库
                #     6.1 根据model_type和model_name创建embedding model对象
                embedding_model = get_openaiEmbedding_model()
                #embedding_model = get_huggingfaceEmbedding_model(model_name="hkunlp/instructor-xl")
                #     6.2 创建向量数据库对象，并将文本embedding后存入到里面
                vector_store = save_chunks_into_vectorstore(content_chunks, embedding_model)
                # 7. 创建对话chain
                # 官网链接：https://python.langchain.com/docs/modules/memory/types/buffer
                st.session_state.conversation = get_chat_chain(vector_store)


if __name__ == "__main__":
    main()