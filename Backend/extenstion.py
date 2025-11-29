from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
import os
import httpx
import json
import re

router = APIRouter(tags=["Extension"])

# Load key securely
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Small constant
EPS = 1e-6


# ============================================================
#  Request Schema
# ============================================================

class QueryPayload(BaseModel):
    query: str
    response: str


# ============================================================
#  Semantic Hash Embeddings (Deterministic)
# ============================================================

def deterministic_hash_embedding(text, dims=512):
    vec = np.zeros(dims)
    text = text.lower()

    for i in range(len(text) - 2):
        gram = text[i:i+3]
        h = abs(hash(gram)) % dims
        vec[h] += 1

    norm = np.linalg.norm(vec)
    return vec / (norm + EPS)


def cosine_sim(a, b):
    return float(np.dot(a, b))


# ============================================================
#  Factual Extraction (Rule-Based)
# ============================================================

def extract_facts(text):
    sentences = [s.strip() for s in re.split(r"[.?!]", text) if s.strip()]
    facts = []

    for s in sentences:
        if any(k in s.lower() for k in [" is ", " was ", " has ", " contains "]):
            facts.append(s)
        elif re.search(r"\d", s):  
            facts.append(s)
        elif re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", s):  
            facts.append(s)

    return facts


def factual_accuracy_grounded(query, response):
    facts = extract_facts(response)
    if not facts:
        return 0.35, 0, 0

    query_words = set(query.lower().split())

    matches = 0
    for f in facts:
        f_words = set(f.lower().split())
        if len(query_words.intersection(f_words)) >= 2:
            matches += 1

    return matches / len(facts), len(facts), matches


# ============================================================
#  Evidence Alignment (Semantic)
# ============================================================

def evidence_alignment(query, response):
    emb_q = deterministic_hash_embedding(query)
    emb_r = deterministic_hash_embedding(response)
    return (1 + cosine_sim(emb_q, emb_r)) / 2


# ============================================================
#  Reasoning Integrity (Causal Keyword Detection)
# ============================================================

def reasoning_integrity(response):
    keywords = [
        "because", "therefore", "hence", "thus", "as a result",
        "implies", "so", "consequently", "which means"
    ]

    steps = sum(1 for k in keywords if k in response.lower())

    if steps == 0:
        return 0.3, 1, 0

    total = steps + 1
    return steps / total, total, steps


# ============================================================
#  Consistency (Contradiction Pattern Detection)
# ============================================================

def consistency_score(response):
    patterns = [
        ("always", "never"),
        ("true", "false"),
        ("yes", "no"),
        ("can", "cannot"),
        ("did", "did not"),
        ("is", "is not")
    ]

    text = response.lower()
    contradictions = sum(1 for a, b in patterns if a in text and b in text)

    if contradictions == 0:
        return 0.9

    return max(0.1, 1 / (contradictions + 1))


# ============================================================
#  Fake News Likelihood (S + U + C)
# ============================================================

def fake_news_likelihood(response):
    text = response.lower()

    sensational = ["shocking", "exposed", "unbelievable", "secret", "breaking"]
    unverifiable = ["allegedly", "some say", "it is said", "rumor", "might be"]
    conspiracy = ["agenda", "cover up", "fake", "hoax"]

    S = min(1, sum(w in text for w in sensational) / 2)
    U = min(1, sum(w in text for w in unverifiable) / 2)
    C = min(1, sum(w in text for w in conspiracy) / 2)

    FNLS = (S + U + C) / 3

    return FNLS, {"Sensational": S, "Unverifiable": U, "Conspiracy": C}


# ============================================================
#  OpenAI 60% Engine
# ============================================================

async def openai_score(query: str, response: str):

    if not OPENAI_KEY:
        return {
            "factual_accuracy": 0.5,
            "consistency": 0.5,
            "reasoning": 0.5,
            "alignment": 0.5,
            "fake_news": 0.5,
        }

    prompt = f"""
Evaluate the following AI response on a scale 0–1. 
Return ONLY valid JSON with numeric values.

Query: {query}
Response: {response}

JSON Format:
{{
 "factual_accuracy": float,
 "consistency": float,
 "reasoning": float,
 "alignment": float,
 "fake_news": float
}}
"""

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0
                }
            )

        raw = r.json()
        content = raw["choices"][0]["message"]["content"].strip()
        return json.loads(content)

    except Exception:
        return {
            "factual_accuracy": 0.5,
            "consistency": 0.5,
            "reasoning": 0.5,
            "alignment": 0.5,
            "fake_news": 0.5,
        }


# ============================================================
#  Weighted Merge (60% OpenAI + 40% Grounded)
# ============================================================

def weighted(oai, grounded):
    return (0.6 * oai) + (0.4 * grounded)


# ============================================================
#  Final Endpoint
# ============================================================

@router.post("/score")
async def score_ai(payload: QueryPayload):

    query = payload.query
    response = payload.response

    # Grounded (40%)
    F_g, fact_total, fact_match = factual_accuracy_grounded(query, response)
    R_g, step_total, valid_steps = reasoning_integrity(response)
    E_g = evidence_alignment(query, response)
    C_g = consistency_score(response)
    FNLS_g, fn_detail = fake_news_likelihood(response)

    # OpenAI (60%)
    oai = await openai_score(query, response)

    F = weighted(oai["factual_accuracy"], F_g)
    R = weighted(oai["reasoning"], R_g)
    E = weighted(oai["alignment"], E_g)
    C = weighted(oai["consistency"], C_g)
    FNLS = weighted(oai["fake_news"], FNLS_g)

    CARS = (0.40 * F) + (0.25 * R) + (0.20 * E) + (0.15 * C)
    TCS = (1 - FNLS) * CARS
    HP = 1 - TCS

    return {
        "query": query,
        "response": response,
        "scores": {
            "Factual_Accuracy": round(F * 100, 2),
            "Reasoning_Integrity": round(R * 100, 2),
            "Evidence_Alignment": round(E * 100, 2),
            "Consistency": round(C * 100, 2),
            "Fake_News_Likelihood": round(FNLS * 100, 2),
            "CARS": round(CARS * 100, 2),
            "Trust_Confidence_Score": round(TCS * 100, 2),
            "Hallucination_Probability": round(HP * 100, 2)
        },
        "details": {
            "Facts_Extracted": fact_total,
            "Facts_Matched": fact_match,
            "Reasoning_Steps": step_total,
            "Valid_Steps": valid_steps,
            "FakeNews_Breakdown": fn_detail,
            "OpenAI_Raw": oai,
            "Grounded_Raw": {
                "F": F_g,
                "R": R_g,
                "E": E_g,
                "C": C_g,
                "FNLS": FNLS_g
            }
        }
    }
