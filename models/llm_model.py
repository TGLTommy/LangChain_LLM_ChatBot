
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from config.keys import Keys

def get_openai_model():
    llm_model = ChatOpenAI(openai_api_key=Keys.OPENAI_API_KEY)
    return llm_model

def get_huggingfacehub(model_name=None):
    llm_model = HuggingFaceHub(repo_id=model_name,
                               huggingfacehub_api_token=Keys.HUGGINGFACEHUB_API_TOKEN)
    return llm_model