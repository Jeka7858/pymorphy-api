from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
app = FastAPI()

# --------- модели запросов ---------
class LemmaRequestBody(BaseModel):
    tokens: List[str]

class NerTextRequestBody(BaseModel):
    text: str

# --------- NER lazy init (чтобы не падал старт) ---------
_segmenter = None
_emb = None
_ner_tagger = None

def get_ner():
    global _segmenter, _emb, _ner_tagger
    if _ner_tagger is None:
        try:
            from natasha import Segmenter, NewsEmbedding, NewsNERTagger
            _segmenter = Segmenter()
            _emb = NewsEmbedding()
            _ner_tagger = NewsNERTagger(_emb)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"NER init failed: {type(e).__name__}: {e}")
    return _segmenter, _ner_tagger


@app.post("/lemmatize")
def lemmatize(data: LemmaRequestBody):
    result: Dict[str, str] = {}
    for token in data.tokens:
        t = token.lower().replace("ё", "е")
        if not t:
            continue
        result[token] = morph.parse(t)[0].normal_form
    return {"lemmas": result}


@app.post("/ner")
def ner(data: NerTextRequestBody):
    try:
        from natasha import Doc

        segmenter, ner_tagger = get_ner()

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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NER failed: {type(e).__name__}: {e}")
