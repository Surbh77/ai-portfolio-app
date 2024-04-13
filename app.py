__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add middleware to enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can set specific origins here instead of "*"
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

openai_api_key=os.getenv('OPENAI_API_KEY')

tokens=300
gpt = ChatOpenAI(max_tokens=tokens,api_key=openai_api_key)



class TextData(BaseModel):
    text: str


@app.post("/assiatant")
async def talk(text_data: TextData):
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    db3 = Chroma(persist_directory='./chroma',collection_name='my_details', embedding_function=embeddings)
    docs = db3.similarity_search(text_data.text,k=4)
    content=''.join([i.page_content for i in docs])
    initial_prompt=f"""
    Your name is surbhi and you are an AI assistant of saurabh and you know only the below details.
    Introduce yourself as an AI assiatant of saurabh. You and saurabh are different persons.
    Anything other than the below content, your name and your role should not be answered.
    
    Contet: {content} 
    
    Question:{text_data.text}
    
    If the user want more information about saurabh tell like please check out his resume.
    And if the question is not related to the above content tell like you know information only about saurabh.
    """
    
    message=initial_prompt
    print(message)
    response=gpt([HumanMessage(content=message)])
    
    ans=response.content
    print(ans)

    return {"blendData": f"{ans}","filename":'abcd'}
