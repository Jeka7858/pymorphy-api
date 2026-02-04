from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import re
import pymorphy2

morph = pymorphy2.MorphAnalyzer()
app = FastAPI()

# -----------------------------
# Request models
# -----------------------------

class LemmaRequestBody(BaseModel):
    tokens: List[str] = Field(..., description="List of tokens (words) to lemmatize")

class NerTextRequestBody(BaseModel):
    text: str = Field(..., description="Text to run NER on")

class LemmaTextRequestBody(BaseModel):
    text: str = Field(..., description="Full text to tokenize and lemmatize")
    window: int = Field(40, ge=0, le=500, description="Context window size (chars) for quotes around each token")


# -----------------------------
# NER lazy init
# -----------------------------

_segmenter = None
_emb = None
_ner_tagger = None
_vocab = None

def get_ner():
    global _segmenter, _emb, _ner_tagger, _vocab
    if _ner_tagger is None:
        try:
            from natasha import Segmenter, NewsEmbedding, NewsNERTagger, MorphVocab
            _segmenter = Segmenter()
            _emb = NewsEmbedding()
            _ner_tagger = NewsNERTagger(_emb)
            _vocab = MorphVocab()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"NER init failed: {type(e).__name__}: {e}")
    return _segmenter, _ner_tagger, _vocab


# -----------------------------
# Helpers
# -----------------------------

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)

def normalize_token(token: str) -> str:
    """
    Normalize token for morphological parsing:
    - lower
    - replace 'ё' -> 'е'
    """
    return token.lower().replace("ё", "е").strip()

def token_quote(text: str, start: int, end: int, window: int) -> str:
    """
    Build a 'quote' snippet around token positions.
    """
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right]


# -----------------------------
# Endpoints
# -----------------------------

@app.post("/lemmatize")
def lemmatize(data: LemmaRequestBody):
    """
    Lemmatize a list of tokens. Returns dict: original_token -> lemma
    (If the same token repeats, last one wins — this keeps your current behavior.)
    """
    result: Dict[str, str] = {}
    for token in data.tokens:
        t = normalize_token(token)
        if not t:
            continue
        try:
            result[token] = morph.parse(t)[0].normal_form
        except Exception as e:
            # If something goes wrong with parsing a token, we skip it rather than fail the entire request.
            # You can change this to raise HTTPException if you prefer strict behavior.
            continue
    return {"lemmas": result}


@app.post("/lemmatize_text")
def lemmatize_text(data: LemmaTextRequestBody):
    """
    Tokenize + lemmatize full text and return items with:
    token, lemma, start, end, quote (context snippet).
    This format is convenient for "finding quotes" later and highlighting in original text.
    """
    text = data.text
    window = int(data.window)

    items: List[Dict[str, Any]] = []

    for m in _WORD_RE.finditer(text):
        token = m.group(0)
        start = m.start()
        end = m.end()

        t = normalize_token(token)
        if not t:
            continue

        try:
            lemma = morph.parse(t)[0].normal_form
        except Exception:
            # If parsing fails, still return token with lemma=None (or token itself if you prefer)
            lemma = None

        items.append({
            "token": token,
            "lemma": lemma,
            "start": start,
            "end": end,
            "quote": token_quote(text, start, end, window),
        })

    return {"items": items}


@app.post("/ner")
def ner(data: NerTextRequestBody):
    try:
        from natasha import Doc

        segmenter, ner_tagger, vocab = get_ner()

        doc = Doc(data.text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)

        entities = []
        for s in doc.spans:
            s.normalize(vocab)
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
