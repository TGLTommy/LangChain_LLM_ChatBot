# coding: utf-8
# author: 唐国梁Tommy
# date: 2023-08-09
import gradio as gr
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

import os
import re
import fitz # 一个用于处理PDF、XPS和其他文档格式的库
from PIL import Image

# 定义2个全局变量，目的是配置、修改 openai key
enable_box = gr.Textbox.update(value=None,
                               placeholder="填写 OpenAI Key",
                               interactive=True)
disable_box = gr.Textbox.update(value="OpenAI Key 已经被设置",
                                interactive=False)


# 定义函数
def set_apikey(api_key):
    '''设置api key'''
    os.environ["OPENAI_API_KEY"] = api_key
    return disable_box

def change_api_box():
    '''修改api key'''
    return enable_box

def add_text(history, text:str):
    '''
    :param history: 用户聊天历史记录
    :param text: 用户的新输入
    :return: 更新后的history列表
    '''
    if not text:
        raise gr.Error("请输入文本")
    history = history + [(text, "")]
    return history


class chatbot:
    def __init__(self, OPENAI_API_KEY: str = None) -> None:
        self.api_key = OPENAI_API_KEY
        self.chain = None
        self.chat_history = [] # 用于存储聊天历史记录
        self.page_num: int = 0 # 当前PDF文件的页码
        self.count: int = 0 # 对话轮次

    def __call__(self, file: str):
        if self.count == 0:
            # 构建基于PDF文件的对话chain
            self.chain = self.build_conversation_chain(file)
            self.count += 1
        return self.chain

    def process_file(self, file: str):
        # 加载PDF文档
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        # 下面是正则匹配，用于找出文件名，pattern的作用：找到字符串末尾最后一个/之后的部分
        # 举例说明，假设文件路径: /home/user/documents/myfile.pdf，
        # 返回的match : /myfile.pdf ,
        # match.group() 返回/myfile.pdf， match.group(1) 返回 myfile.pdf
        pattern = r"/([^/]+)$"
        match = re.search(pattern, file.name)
        file_name = match.group(1)
        return documents, file_name

    def build_conversation_chain(self, file):
        '''核心模块：① 从pdf中提取text，② 创建embeddings，③ 将embeddings存到向量数据库，④ 创建对话chain'''
        # 1. 判断openai key是否存在
        if "OPENAI_API_KEY" not in os.environ:
            raise gr.Error("OpenAI key不存在，请上传")
        # 2. 文档预处理，提取内容、获取文件名
        documents, file_name = self.process_file(file)
        # 3. 创建embedding model，用于生成文档的embedding表示
        embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key)
        # 4. 创建向量数据库，并将数据存进去
        vectorstore = FAISS.from_documents(documents=documents,
                                           embedding=embedding_model)
        # 5. 创建对话检索链
        chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0, openai_api_key=self.api_key),
                                                      retriever=vectorstore.as_retriever(search_kwargs={"k":1}),
                                                      return_source_documents=True,)
        return chain


def generate_response(history, query, file):
    '''答案生成'''
    # 首先，判断一下是否上传了PDF文档
    if not file:
        raise gr.Error(message="上传一个PDF文档")
    # 返回对话chain
    chain = app(file)
    # 处理用户的查询并生成答案
    result = chain({"question": query,
                    "chat_history": app.chat_history},
                    return_only_outputs=True)
    # print("result = ", result.keys())
    # 将当前的查询和生成的答案添加到聊天历史中
    app.chat_history += [(query, result["answer"])]
    # 从源文档中获取当前页的页码
    app.page_num = list(result["source_documents"][0])[1][1]["page"]
    # 将生成的回答每个字符添加到history中，用于实时显示答案字符 ， 举例说明：
    # [("你好吗?", "")]
    # [("你好吗?", "我")]
    # [("你好吗?", "我很")]
    # [("你好吗?", "我很好")]
    # [("你好吗?", "我很好，你")]
    # [("你好吗?", "我很好，你呢")]
    # [("你好吗?", "我很好，你呢？")]
    for char in result["answer"]:
        history[-1][-1] += char
        yield history, ""

def render_file(file):
    # 打开PDF文档
    doc = fitz.open(file.name)
    # 根据页面获取当页的内容
    page = doc[app.page_num]
    # 将页面渲染为分辨率为300 DPI的PNG图像，从默认的72DPI转换到300DPI
    picture = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    # 从渲染的像素数据创建一个Image对象
    image = Image.frombytes("RGB", [picture.width, picture.height], picture.samples)
    # 返回渲染后的图像
    return image

def render_first(file):
    document = fitz.open(file)
    page = document[0]
    picture = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    image = Image.frombytes("RGB", [picture.width, picture.height], picture.samples)
    return image, []

app = chatbot()

# 参考官网链接：https://www.gradio.app/guides/creating-a-chatbot-fast
with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            # 占据80%宽度的列
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(
                    placeholder="请输入你的OpenAI Key",
                    show_label=False, # 不显示标签
                    interactive=True, # 当其值变化时，它会立即触发事件。
                    container=False, # 组件不会有边框
                )
            # 占据20%宽度的列
            with gr.Column(scale=0.2):
                new_api_key = gr.Button("修改Key")

        with gr.Row():
            # 参考官网链接：gradio.app/docs/chatbot
            # 创建一个聊天界面
            chatbot = gr.Chatbot(value=[], elem_id="chatbot", height=600)
            # 创建一个图像组件，供用户查看上传的PDF文件的某一页的渲染。
            show_file = gr.Image(label="上传PDF", tool="select", height=630)


    with gr.Row():
        with gr.Column(scale=0.6):
            txt = gr.Textbox(show_label=False, placeholder="请输入文本", container=False)

        with gr.Column(scale=0.2):
            submit_button = gr.Button("提交")

        with gr.Column(scale=0.2):
            button = gr.UploadButton("上传一个PDF文档", file_types=[".pdf"])


    # *** 很重要的步骤：设置事件处理器(handler) ***
    # 提交 openai key
    api_key.submit(fn=set_apikey, inputs=[api_key], outputs=[api_key])
    # 修改 openai key
    new_api_key.click(fn=change_api_box, outputs=[api_key])
    # 上传 pdf，outputs定义了哪些组件会被这个函数的返回值更新
    button.upload(fn=render_first, inputs=[button], outputs=[show_file, chatbot],)
    # 提交text，生成回答
    submit_button.click(
        fn=add_text, # 触发add_text函数
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=True # 如果同时有多个请求，这个函数应当排队执行。
    ).success(
        fn=generate_response,
        inputs=[chatbot, txt, button],
        outputs=[chatbot, txt]
    ).success(
        fn=render_file,
        inputs=[button],
        outputs=[show_file]
    )

# concurrency_count 处理并发的请求，这里设置为只处理一个请求。
demo.queue(concurrency_count=1).launch(share=False)
