# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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




from langchain.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
# from dotenv import load_dotenv # Importing dotenv to get API key from .env file
from langchain.chat_models import ChatOpenAI # Import OpenAI LLM
from langchain_core.prompts import ChatPromptTemplate
import os # Importing os module for operating system functionalities

openai_api_key=os.getenv('OPENAI_API_KEY')


CHROMA_PATH = "chroma"
query_text = "What is the education of saurabh"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

def query_rag(query_text):
  """
  Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
  Args:
    - query_text (str): The text to query the RAG system with.
  Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
  """
  # YOU MUST - Use same embedding function as before
  openai_api_key=os.getenv('OPENAI_API_KEY')
  embedding_function = OpenAIEmbeddings(api_key=openai_api_key)

  # Prepare the database
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  
  # Retrieving the context from the DB using similarity search
  results = db.similarity_search_with_relevance_scores(query_text, k=3)

  # Check if there are any matching results or if the relevance score is too low
  if len(results) == 0 or results[0][1] < 0.7:
    print(f"Unable to find matching results.")

  # Combine context from matching documents
  context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
 
  # Create prompt template using context and query text
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)
  
  # Initialize OpenAI chat model
  model = ChatOpenAI(api_key=openai_api_key)

  # Generate response text based on the prompt
  response_text = model.predict(prompt)
 
   # Get sources of the matching documents
  sources = [doc.metadata.get("source", None) for doc, _score in results]
 
  # Format and return response including generated text and sources
  formatted_response = f"Response: {response_text}\nSources: {sources}"
  return formatted_response, response_text

# Let's call our function we have defined
formatted_response, response_text = query_rag(query_text)
# and finally, inspect our final response!
print(response_text)




app = FastAPI()

# Add middleware to enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can set specific origins here instead of "*"
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)



tokens=300
gpt = ChatOpenAI(max_tokens=tokens,api_key=openai_api_key)



class TextData(BaseModel):
    text: str


@app.post("/assiatant")
async def talk(text_data: TextData):
    # embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    # db3 = Chroma(persist_directory='./chroma',collection_name='my_details', embedding_function=embeddings)
    # docs = db3.similarity_search(text_data.text,k=4)
    # content=''.join([i.page_content for i in docs])
    # initial_prompt=f"""
    # Your name is surbhi and you are an AI assistant of saurabh and you know only the below details.
    # Introduce yourself as an AI assiatant of saurabh. You and saurabh are different persons.
    # Anything other than the below content, your name and your role should not be answered.
    
    # Contet: {content} 
    
    # Question:{text_data.text}
    
    # If the user want more information about saurabh tell like please check out his resume.
    # And if the question is not related to the above content tell like you know information only about saurabh.
    # """
    
    # message=initial_prompt
    # response=gpt([HumanMessage(content=message)])
    
    # ans=response.content

    formatted_response, response_text = query_rag(query_text)
    
    return {"blendData": f"{response_text}","filename":'abcd'}
