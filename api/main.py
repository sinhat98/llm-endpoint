from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import google.generativeai as genai

import os
from pathlib import Path

def read_secret(file_name):
    file_path = Path('.secret') / file_name
    with open(file_path, "r") as file:
        return file.read().strip()

app = FastAPI()
try:
    from algo.bert import get_bert_model
    bert = get_bert_model()
except:
    bert = None



class TextRequest(BaseModel):
    text: str
    model: str  # モデルを指定するためのフィールドを追加

class TextResponse(BaseModel):
    processed_text: str
    sentiment_label: int
    sentiment_score: float

@app.post("/process-text/", response_model=TextResponse)
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
            if bert is not None:    
                sentiment_output = bert(request.text)
                sentiment_label = sentiment_output.label
                sentiment_score = sentiment_output.score
            else:
                sentiment_label = 1
                sentiment_score = 1.
            if sentiment_label == 1:
                prompt = '''次の文章は自然言語処理モデルによって有害と判断されました。以下の文章を修正してください。
                {}
                '''
                response = model.generate_content(prompt.format(request.text))
                output_text = response.text
            else:
                output_text = request.text
            return {"processed_text": output_text, "sentiment_label": sentiment_label, "sentiment_score": sentiment_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
