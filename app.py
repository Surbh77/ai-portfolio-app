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
You are an AI assistant of Saurabh. Your task is to answer the question asked by the user.
For answering only refer the content provided below. 
User doesnt about know any content provided to you
If the question is not relevent to the content apologize to the user and give a polite answer that you only have information related to saurabh.
Your response should only contain the answer.
Your answer should be short and maximum of 2 sentences.

Content:- 
SAURABH SHINDE 
Consultant – II (Generative AI) 
Pune, Maharashtra 
PROFILE SUMMARY 
• AI & Machine Learning professional with 3.5+ years of experience, specializing in Generative AI, NLP, 
Deep Learning, and Machine Learning. Proficient in Python, Azure Cloud, Azure DevOps, and Azure 
OpenAI, with expertise in LLMs, LangChain, Chatbots, and RAG-based systems. Skilled in building and 
deploying AI solutions using TensorFlow, PyTorch, Scikit-learn, and NLP frameworks. Experienced in 
Azure AI services, data science, and cloud-based AI automation. 
STRENGTHS AND EXPERTISE 
• Generative AI : LLM, prompt engineering, Langchain, LLAMA_2, Chatbot, explainable AI, Shapash, Lime, 
chainlit, streamlit, Ragas. 
• Deep Learning: Skilled in implementing models using frameworks such as Scikit-learn, TensorFlow, 
and Keras. opencv, torch. 
• Machine Learning: Strong understanding of various machine learning algorithms, including regression, 
classification, clustering, decision trees, random forests. 
• Natural Language Processing: Expert in text analysis, sentiment analysis, and language generation 
with NLTK, spaCy, and Transformers. 
• Cloud Services: Azure AI search, Azure WebApp, Azure Functions, Azure key vault, Azure cognitive 
services, Azure OpenAI, Microsoft Graph API, AWS Secrets Manager. 
• DevOps: Azure DevOps, CI/CD Pipelines, Terraform Scripts, Webapp, Azure functions. 
• Data Analysis and Visualization: Proficient in Python and SQL to extract, clean, and analyze large 
datasets. Experienced in using libraries such as Pandas, NumPy, and Matplotlib for data manipulation, 
exploration, and visualization. 
• Communication and Collaboration: Strong verbal and written communication skills, with the ability to 
present technical findings to both technical and non-technical stakeholders. Collaborative team player 
with excellent problem-solving and interpersonal skills. 
PROFESSIONAL EXPERIENCE 
CAPCO- Pune                      
Consultant-II (Generative AI) 
Roles and Responsibilities: 
June 2024 – Present 
• Developed an intelligent chatbot solution using Python, Azure Cloud, Azure OpenAI, and Azure DevOps. 
• Integrated Model Monitoring to monitor the model performance with parameters like accuracy, 
completeness, hallucination, robustness, efficiency, latency, harmfulness, bad-actor, etc. 
• Implementing CI/CD pipelines to automate deployment and streamline development. 
Prodapt Solutions- Chennai            
Senior Software Engineer (Machine Learning) 
Roles and Responsibilities: 
Nov 2022 – June 2024 
• Develop and implement machine learning models to predict customer behaviour, optimize pricing 
strategies, and improve marketing campaign effectiveness, resulting in a 15% increase in customer 
acquisition and a 10% boost in revenue. 
• Automate data extraction and pre-processing tasks using Python scripting, reducing data processing 
time by 30% and improving data quality. 
TVSSCS – Pune                    June 2021 – Apr 2022 
Data Analyst 
Roles and Responsibilities: 
• Conducted data cleaning, transformation, and analysis of customer data, resulting in the identification 
of key customer segments and the development of targeted marketing strategies. 
• Developed and maintained SQL queries and scripts to extract data from relational databases, enabling 
efficient data retrieval and analysis. 
 
INDUSTRY PROJECTS 
Generative AI (Frontline Assistant)              June 2024 – Present 
Business Idea: 
• The AI HR Assistant with prompt engineering is an innovative solution aimed at enhancing HR 
processes by leveraging advanced AI technologies. 
 
Generative AI (HR Assistant)            Aug 2023 – June 2024 
Business Idea: 
• The AI HR Assistant with prompt engineering is an innovative solution aimed at enhancing HR 
processes by leveraging advanced AI technologies. 
 
Sales Win Prediction             Dec 2023 – June 2024 
Business Idea: 
• Sales Win Prediction offers AI-driven analytics to forecast and optimize sales outcomes, empowering 
businesses to make data-driven decisions and increase their win rates. 
 
Employee Performance and Attrition Prediction             Dec 2022 - Aug 2023 
Business Idea: 
• Developing an AI-powered platform that utilizes employee data and machine learning algorithms to 
predict performance and attrition, enabling organizations to proactively identify at-risk employees and 
take strategic actions to improve retention and productivity. 
 
PERSONAL PROJECTS 
   AI Teacher       Talk to documents      Log Anomaly Detection 
  RAG Chatbot         OCR Cheque Data Extraction 
 
AWARDS AND RECOGNITION 
   Hackathon Winner – 2024      Spotlight Award - 2023    Team Cheer Award – 2023 
 
EDUCATION 
ME – (First Class Distinction)                  Completed in 2022 
AISSMS College of Engineering, Pune. 
 
BE – (First Class)                   Completed in 2019 
NBN Sinhgad School of Engineering, Pune. 
 
HSC – (First Class)                   Completed in 2015 
Bharat Children's Academy & Jr. College, Walchandnagar. 
 
SSC– (First Class)                   Completed in 2013 
Bharat Children's Academy & Jr. College, Walchandnagar. 
 
PROFESSIONAL COURCES 
 
• DataCamp Certified: - Intermediate Python for Data Science. 
• DataCamp Certified: - Introduction to Python for Data Science. 

Hobbies:
I love to play different types of musical instruments like guitar, flute, Mouthorgan and Tabla.
I am also a health consious and love to hit gym 5 times a week.
I like to travel and explore new places.

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


  # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  

  # results = db.similarity_search_with_relevance_scores(query_text, k=3)


  # if len(results) == 0 or results[0][1] < 0.7:
  #   print(f"Unable to find matching results.")


  # context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
 

  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format( question=query_text)
  

  model = ChatOpenAI(api_key=openai_api_key,model="gpt-4")


  response_text = model.predict(prompt)
 

  # sources = [doc.metadata.get("source", None) for doc, _score in results]
 

  # formatted_response = f"Response: {response_text}\nSources: {sources}"
  return  response_text






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

    response_text = query_rag(text_data.text)
    
    return {"blendData": f"{response_text}","filename":'abcd'}
