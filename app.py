from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import pymorphy2

from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc


morph = pymorphy2.MorphAnalyzer()
app = FastAPI()

# --- NER (Natasha) init: создаём один раз, не внутри функции ---
segmenter = Segmenter()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)


class RequestBody(BaseModel):
    tokens: List[str]


class NerRequestBody(BaseModel):
    text: str


@app.post("/lemmatize")
def lemmatize(data: RequestBody):
    result: Dict[str, str] = {}
    for token in data.tokens:
        t = token.lower().replace("ё", "е")
        if not t:
            continue
        result[token] = morph.parse(t)[0].normal_form
    return {"lemmas": result}


@app.post("/ner")
def ner(data: NerRequestBody):
    doc = Doc(data.text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    entities = []
    for s in doc.spans:
        s.normalize()
        entities.append({
            "text": s.text,
            "type": s.type,   # PER / ORG / LOC
            "start": s.start,
            "end": s.stop
        })

    return {"entities": entities}
