
from config.keys import Keys
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

def get_openaiEmbedding_model():
    return OpenAIEmbeddings(openai_api_key=Keys.OPENAI_API_KEY)

def get_huggingfaceEmbedding_model(model_name):
    return HuggingFaceInstructEmbeddings(model_name=model_name)

