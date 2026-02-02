from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
app = FastAPI()

class RequestBody(BaseModel):
    tokens: List[str]

@app.post("/lemmatize")
def lemmatize(data: RequestBody):
    result: Dict[str, str] = {}
    for token in data.tokens:
        t = token.lower().replace("ё", "е")
        if not t:
            continue
        result[token] = morph.parse(t)[0].normal_form
    return {"lemmas": result}
