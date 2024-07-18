# import streamlit as st

# st.title("Mental Health Chatbot")

# user_input = st.text_input("How can I help you today?")
# if user_input:
#     response = get_response(user_input)
#     st.write(response)


from fastapi import FastAPI, Request
from weaviate_schema import create_schema
from semantic_search import perform_semantic_search
from Models.llm_model import load_model
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    context: str

# Initialize and load models
llm_model = load_model()

@app.on_event("startup")
async def startup_event():
    # Create schema in Weaviate
    create_schema()

@app.post("/search")
async def search(request: Request):
    query = await request.json()
    result = perform_semantic_search(query['question'])
    return {"result": result}

@app.post("/generate")
async def generate(request: Request):
    query = await request.json()
    response = llm_model.generate(query['question'])
    return {"response": response}
