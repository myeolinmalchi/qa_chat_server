from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from typing import List, Optional, Dict
from pydantic import BaseModel
from starlette.status import HTTP_400_BAD_REQUEST
from db import auth

import openai
import json

load_dotenv()

app = FastAPI()

@app.get('/', status_code=status.HTTP_200_OK)
def healthcheck():
    return { 'message': 'Everything OK!' }

class Message(BaseModel):
    role: str
    content: str

class ClassificationRequest(BaseModel):
    messages: List[Message]

class ClassificationResponse(BaseModel):
    type: str
    answer: str

@app.get("api/v1/classification", dependencies=[Depends(auth)])
def classification(body: ClassificationRequest):
    completion = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=body.messages, 
        temperature=0.3
    )
    if isinstance(completion, Dict):
        jsonstring = {**completion}['choices'][0]['messages']['content']
        _json = json.load(jsonstring)
        return ClassificationResponse(type=_json["type"], answer=_json["answer"])

    else:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, 
            detail={
                "message": "Error has occurred while completion."
            }
        )

    
