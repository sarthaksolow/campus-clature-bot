from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.rag_pipeline import get_qa_chain

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    qa_chain = get_qa_chain(user_id=request.user_id)
    response = qa_chain.invoke({"question": request.message})
    return {"answer": response["answer"]}
