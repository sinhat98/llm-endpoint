from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import google.generativeai as genai
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import os
from pathlib import Path

def read_secret(file_name):
    file_path = Path('.secret') / file_name
    with open(file_path, "r") as file:
        return file.read().strip()

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    model: str  # モデルを指定するためのフィールドを追加

@app.post("/process-text/")
async def process_text(request: TextRequest):
    try:
        if request.model == 'openai':
            # OpenAI APIキーを設定
            api_key = read_secret('openai.txt')
            openai.api_key = api_key
            # OpenAI APIを叩く
            response = openai.Completion.create(
                engine="gpt-3.5", # または他のモデル
                prompt=request.text,
                temperature=0.7,
                max_tokens=100
            )
            return {"processed_text": response.choices[0].text.strip()}
        elif request.model == 'google':
            api_key = os.getenv('GOOGLE_API_KEY')
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(request.text)
            return {"processed_text": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
