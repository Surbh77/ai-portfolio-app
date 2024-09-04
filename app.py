from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores.chroma import Chroma 
from langchain.chat_models import ChatOpenAI # Import OpenAI LLM
from langchain_core.prompts import ChatPromptTemplate
import os # Importing os module for operating system functionalities

openai_api_key=os.getenv('OPENAI_API_KEY')


CHROMA_PATH = "chroma"


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

  openai_api_key=os.getenv('OPENAI_API_KEY')
  embedding_function = OpenAIEmbeddings(api_key=openai_api_key)


  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  

  results = db.similarity_search_with_relevance_scores(query_text, k=3)


  if len(results) == 0 or results[0][1] < 0.7:
    print(f"Unable to find matching results.")


  context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
 

  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)
  

  model = ChatOpenAI(api_key=openai_api_key)


  response_text = model.predict(prompt)
 

  sources = [doc.metadata.get("source", None) for doc, _score in results]
 

  formatted_response = f"Response: {response_text}\nSources: {sources}"
  return formatted_response, response_text






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

    formatted_response, response_text = query_rag(text_data.text)
    
    return {"blendData": f"{response_text}","filename":'abcd'}
