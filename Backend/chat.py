# chatapi_with_grounding.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx
import asyncio
import time
import os
from dotenv import load_dotenv
import math
import typing
from typing import List, Dict, Any
import re

UNCERTAINTY = [
    "not aware", "no widely", "not familiar", "fictional",
    "obscure", "does not appear", "no information",
    "might be", "cannot confirm", "unclear"
]

def count_named_entities(text):
    # Simple heuristic — plug your NER model here
    patterns = [
        r"\b[A-Z][a-z]+\b",               # Names / cities
        r"\b(19|20)\d{2}\b",              # Year
        r"\b(police|court|arrested|FIR)\b",
        r"\bcase\b"
    ]
    hits = 0
    for p in patterns:
        hits += len(re.findall(p, text))
    return hits

def contains_uncertainty(text):
    t = text.lower()
    return any(p in t for p in UNCERTAINTY)

def calculate_structure_score(text: str) -> float:
    """
    Heuristic for answer structure quality (0.0 - 1.0).
    - Penalizes very short answers.
    - Rewards proper capitalization and punctuation.
    """
    if not text or len(text.strip()) < 10:
        return 0.1
    
    score = 0.5  # Start neutral
    
    # Length bonus (up to 0.2)
    length = len(text.split())
    if length > 20:
        score += 0.1
    if length > 50:
        score += 0.1
        
    # Formatting bonus (up to 0.2)
    if "\n" in text: # Has paragraphs/lists
        score += 0.1
    if text[0].isupper() and text.strip()[-1] in ".!?":
        score += 0.1
        
    return min(1.0, score)

load_dotenv()
router = APIRouter()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY") 
WIKI_ENABLED = os.getenv("USE_WIKIPEDIA", "true").lower() in ("1", "true", "yes")

# WARNINGS (non-blocking)
if not GEMINI_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables")

if not GROQ_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables")

if not OPENAI_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables")

# Request Model
# -------------------------
class ChatRequest(BaseModel):
    query: str
    
    def sanitize_query(self) -> str:
        """Remove invalid control characters from query"""
        # Remove control characters except newline, tab, and carriage return
        import unicodedata
        sanitized = ''.join(
            char for char in self.query 
            if unicodedata.category(char)[0] != 'C' or char in '\n\r\t'
        )
        return sanitized.strip()

# -------------------------
# Math & Embedding Utilities
# -------------------------
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def semantic_distance(vec_a: List[float], vec_b: List[float]) -> float:
    sim = cosine_similarity(vec_a, vec_b)
    return 1.0 - sim

async def get_embedding(text: str, api_key: str) -> List[float]:
    """Fetch embedding for a given text using Gemini API. Returns [] on failure."""
    if not api_key:
        return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={api_key}"
    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(url, json=payload)
            data = resp.json()
            if "embedding" in data:
                return data["embedding"]["values"]
            # Some Gemini versions respond differently; attempt common alternatives
            if "candidates" in data and data["candidates"]:
                # try to find embedding-like object
                return data["candidates"][0].get("embedding", {}).get("values", []) or []
            return []
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

# -------------------------
# Variance / Consistency / CACE
# -------------------------
def calculate_variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)

def calculate_consistency(embeddings: List[List[float]], current_index: int) -> float:
    """
    Calculate consistency score for a response based on similarity to other responses.
    Handles edge cases and embedding failures properly.
    """
    if not embeddings or len(embeddings) <= 1:
        return 0.1  # Low consistency when few responses for comparison
    
    current_embedding = embeddings[current_index]
    if not current_embedding:
        return 0.05  # Very low if no embedding available
    
    similarities = []
    valid_comparisons = 0
    
    for i, other_embedding in enumerate(embeddings):
        if i == current_index:  # Skip self-comparison
            continue
        if not other_embedding:  # Skip invalid embeddings
            continue
            
        similarity = cosine_similarity(current_embedding, other_embedding)
        similarities.append(similarity)
        valid_comparisons += 1
    
    if valid_comparisons < 2:
        # Need at least 2 valid comparisons for meaningful consistency
        return 0.15 if valid_comparisons == 1 else 0.1
    
    # Calculate variance and convert to consistency score
    variance = calculate_variance(similarities)
    
    # Convert variance to consistency (low variance = high consistency)
    # Use inverse relationship with smoothing
    consistency = 1.0 / (1.0 + (variance * 10))  # Scale variance appropriately
    
    # Ensure reasonable bounds
    return max(0.05, min(0.95, consistency))

def calculate_cace(distances: List[float]) -> float:
    epsilon = 0.05
    if not distances:
        return epsilon
    exps = [math.exp(-(d + epsilon)) for d in distances]
    sum_exps = sum(exps)
    if sum_exps == 0:
        return epsilon
    probs = [e / sum_exps for e in exps]
    entropy = 0.0
    for p in probs:
        entropy -= p * math.log(p + epsilon)
    return max(0.0, entropy)

# -------------------------
# External Fact-Checking Layer (Grounded Data Layer)
# -------------------------
async def external_fact_check_serp(query_text: str) -> float:
    """
    Returns a confidence score [0.0, 1.0] that the text is supported by search results.
    Uses SERP_API_KEY if available; otherwise returns a conservative default (0.2).
    """
    if not SERP_API_KEY:
        return 0.2  # conservative fallback when no API key

    # Example: generic SERP API endpoint (adapt to your actual provider)
    # NOTE: You should adapt to your SERP provider's request/response format.
    url = "https://api.serply.io/v1/search"  # placeholder — replace with your actual provider
    payload = {"q": query_text, "num": 5}
    headers = {"Authorization": f"Bearer {SERP_API_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(url, json=payload, headers=headers)
            data = r.json()
            results = data.get("results") or data.get("organic") or data.get("items") or []
            if not results:
                return 0.2
            # crude matching: count how many snippets contain a major claim phrase
            lower_text = query_text.lower()
            matches = 0
            for item in results[:5]:
                snippet = (item.get("snippet") or item.get("title") or item.get("description") or "").lower()
                if not snippet:
                    continue
                # If many words overlap, consider it supporting evidence
                overlap = sum(1 for w in lower_text.split() if len(w) > 3 and w in snippet)
                if overlap >= max(1, len(lower_text.split()) // 6):  # heuristic
                    matches += 1
            confidence = matches / min(len(results[:5]), 5)
            # Bound and smooth
            return max(0.05, min(0.99, confidence))
    except Exception as e:
        print(f"SERP fact-check error: {e}")
        return 0.2

async def external_fact_check_wikipedia(claim_text: str) -> float:
    """
    Quick Wikipedia lookup: returns 0.0-1.0 support.
    This is a very permissive check: looks for exact presence in top page extracts.
    """
    if not WIKI_ENABLED:
        return 0.2
    try:
        # Use MediaWiki API search
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": claim_text,
            "format": "json",
            "srlimit": 3
        }
        async with httpx.AsyncClient(timeout=6) as client:
            resp = await client.get(search_url, params=params)
            data = resp.json()
            hits = data.get("query", {}).get("search", []) or []
            if not hits:
                return 0.1
            # fetch the top hit extract
            top = hits[0]
            pageid = top.get("pageid")
            if not pageid:
                return 0.1
            extract_params = {
                "action": "query",
                "prop": "extracts",
                "pageids": pageid,
                "explaintext": True,
                "format": "json",
            }
            resp2 = await client.get(search_url, params=extract_params)
            d2 = resp2.json()
            extracts = d2.get("query", {}).get("pages", {}).get(str(pageid), {}).get("extract", "") or ""
            lower_extract = extracts.lower()
            lower_claim = claim_text.lower()
            # crude check: count overlap of long words
            overlap = sum(1 for w in lower_claim.split() if len(w) > 4 and w in lower_extract)
            if overlap >= max(1, len(lower_claim.split()) // 8):
                return 0.9  # strong support from Wikipedia
            return 0.2
    except Exception as e:
        print(f"Wikipedia fact-check error: {e}")
        return 0.2

async def external_fact_check_combined(text: str) -> float:
    """
    Combined external fact check:
    - Uses SERP provider (preferred)
    - Uses Wikipedia as secondary check
    Returns smoothed confidence in [0.05, 0.99]
    """
    try:
        serp = await external_fact_check_serp(text)
        wiki = await external_fact_check_wikipedia(text) if WIKI_ENABLED else 0.2
        # Weighted average: prefer SERP evidence but boost if wiki agrees strongly
        combined = (0.7 * serp) + (0.3 * wiki)
        return max(0.05, min(0.99, combined))
    except Exception as e:
        print(f"Combined fact-check error: {e}")
        return 0.2

# -------------------------
# Evaluation (Gemini judge) — unchanged but kept safe
# -------------------------
async def evaluate_response_metrics(query: str, answer: str) -> Dict[str, float]:
    """
    Uses Gemini to estimate factual accuracy, reasoning depth, and verification.
    Returns normalized scores (0-1). If Gemini fails, returns conservative defaults.
    """
    if not GEMINI_KEY:
        return {"factual_accuracy": 0.05, "reasoning_depth": 0.05, "source_verification": 0.05}

    prompt = f"""
    Analyze the following AI response to the query: "{query}"

    Response: "{answer}"

    Provide a JSON object with:
    - factual_accuracy_score: Float (0.0 to 1.0)
    - reasoning_step_count: Integer (estimated number of reasoning steps/nodes)
    - verified_fact_count: Integer
    - total_fact_count: Integer

    Return ONLY the JSON object.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with httpx.AsyncClient(timeout=12) as client:
        try:
            resp = await client.post(url, json=payload)
            data = resp.json()
            text_resp = data["candidates"][0]["content"]["parts"][0]["text"]
            # sanitize
            text_resp = text_resp.replace("```json", "").replace("```", "").strip()
            import json
            metrics = json.loads(text_resp)
            raw_acc = float(metrics.get("factual_accuracy_score", 0.0))
            nodes = int(metrics.get("reasoning_step_count", 0))
            verified = int(metrics.get("verified_fact_count", 0))
            total_facts = int(metrics.get("total_fact_count", 1))
            if total_facts < 1:
                total_facts = 1
            epsilon = 0.05
            factual_smoothed = (raw_acc + epsilon) / (1.0 + epsilon)
            effective_nodes = max(1, nodes)
            r_val = math.log2(1 + effective_nodes)
            reasoning_smoothed = min(1.0, r_val / 5.0)
            verification_smoothed = (verified + epsilon) / (total_facts + epsilon)
            return {
                "factual_accuracy": factual_smoothed,
                "reasoning_depth": reasoning_smoothed,
                "source_verification": verification_smoothed
            }
        except Exception as e:
            print(f"Evaluation error: {e}")
            epsilon = 0.05
            return {
                "factual_accuracy": epsilon / (1 + epsilon),
                "reasoning_depth": 0.05,
                "source_verification": epsilon / (1 + epsilon)
            }

async def evaluate_response_metrics_openai(query: str, answer: str) -> Dict[str, float]:
    """
    Uses OpenAI GPT-4o to estimate factual accuracy, reasoning depth, and verification.
    Returns normalized scores (0-1).
    """
    if not OPENAI_KEY:
        return {"factual_accuracy": 0.05, "reasoning_depth": 0.05, "source_verification": 0.05}

    prompt = f"""
    Analyze the following AI response to the query: "{query}"

    Response: "{answer}"

    Provide a JSON object with:
    - factual_accuracy_score: Float (0.0 to 1.0)
    - reasoning_step_count: Integer (estimated number of reasoning steps/nodes)
    - verified_fact_count: Integer
    - total_fact_count: Integer

    Return ONLY the JSON object.
    """
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    
    async with httpx.AsyncClient(timeout=12) as client:
        try:
            resp = await client.post(url, json=payload, headers=headers)
            data = resp.json()
            if "choices" not in data:
                return {"factual_accuracy": 0.05, "reasoning_depth": 0.05, "source_verification": 0.05}
                
            text_resp = data["choices"][0]["message"]["content"]
            # sanitize
            text_resp = text_resp.replace("```json", "").replace("```", "").strip()
            import json
            metrics = json.loads(text_resp)
            
            raw_acc = float(metrics.get("factual_accuracy_score", 0.0))
            nodes = int(metrics.get("reasoning_step_count", 0))
            verified = int(metrics.get("verified_fact_count", 0))
            total_facts = int(metrics.get("total_fact_count", 1))
            
            if total_facts < 1:
                total_facts = 1
            epsilon = 0.05
            
            factual_smoothed = (raw_acc + epsilon) / (1.0 + epsilon)
            effective_nodes = max(1, nodes)
            r_val = math.log2(1 + effective_nodes)
            reasoning_smoothed = min(1.0, r_val / 5.0)
            verification_smoothed = (verified + epsilon) / (total_facts + epsilon)
            
            return {
                "factual_accuracy": factual_smoothed,
                "reasoning_depth": reasoning_smoothed,
                "source_verification": verification_smoothed
            }
        except Exception as e:
            print(f"OpenAI Evaluation error: {e}")
            epsilon = 0.05
            return {
                "factual_accuracy": epsilon / (1 + epsilon),
                "reasoning_depth": 0.05,
                "source_verification": epsilon / (1 + epsilon)
            }

def compute_CARS(metrics: Dict[str, float], answer_text: str) -> float:
    """
    Compute CARS score with dynamic minimum based on available evidence
    """
    # ------------------------
    # LAYER 1: Base weighted score
    # ------------------------
    base = (
        0.25 * metrics.get("factualAccuracy_judge", 0) +
        0.20 * metrics.get("factualAccuracy_grounded", 0) +
        0.15 * metrics.get("reasoningDepth", 0) +
        0.10 * metrics.get("external_confidence", 0) +
        0.10 * metrics.get("consistency", 0) +
        0.10 * metrics.get("sourceVerification", 0) +
        0.10 * metrics.get("structure_score", 0)
    )

    # ------------------------
    # LAYER 2: Low factual accuracy penalty
    # ------------------------
    factual_grounded = metrics.get("factualAccuracy_grounded", 0)
    if factual_grounded < 0.15:
        # Scale penalty based on how bad it is
        penalty_factor = max(0.1, factual_grounded)  # Worse scores get heavier penalties
        base *= penalty_factor

    # ------------------------
    # LAYER 3: Specificity boost (NER)
    # ------------------------
    ner_count = count_named_entities(answer_text)
    if ner_count >= 2:
        base += 0.15
    elif ner_count == 1:
        base += 0.05

    # ------------------------
    # LAYER 4: Uncertainty penalty
    # ------------------------
    if contains_uncertainty(answer_text):
        base -= 0.25

    # ------------------------
    # LAYER 5: Calculate dynamic minimum based on structure and basic metrics
    # ------------------------
    structure_val = metrics.get("structure_score", 0.1)
    consistency_val = metrics.get("consistency", 0.05)
    
    # Dynamic minimum: even poor answers get some credit for structure and consistency
    dynamic_min = (structure_val * 0.3 + consistency_val * 0.2) * 0.5
    
    base = max(dynamic_min, min(base, 1.0))
    
    return base


# -------------------------
# Callers for Gemini & Groq (kept safe)
# -------------------------
async def call_gemini(query: str):
    if not GEMINI_KEY:
        return {"model": "gemini-2.5-flash", "answer": "[ERROR] Gemini failed → Missing GOOGLE_API_KEY"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_KEY}"
    payload = {"contents": [{"parts": [{"text": query}]}]}
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(url, json=payload)
            data = r.json()
            ans = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"model": "gemini-2.5-flash", "answer": ans}
        except Exception as e:
            return {"model": "gemini-2.5-flash", "answer": f"[ERROR] Gemini failed → {str(e)}"}

async def get_groq_models():
    if not GROQ_KEY:
        return [
            "llama-3.1-70b-versatile",
            "llama3-8b-instant",
            "mixtral-8x7b",
            "gemma-7b-it",
            "llama-3.1-8b-instant",
            "llama-3.1-405b-reasoning",
        ]
    url = "https://api.groq.com/openai/v1/models"
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.get(url, headers={"Authorization": f"Bearer {GROQ_KEY}"})
            data = r.json()
            model_list = [m["id"] for m in data.get("data", []) if "whisper" not in m["id"]]
            return model_list
        except Exception:
            return [
                "llama-3.1-70b-versatile",
                "llama3-8b-instant",
                "mixtral-8x7b",
                "gemma-7b-it",
                "llama-3.1-8b-instant",
                "llama-3.1-405b-reasoning",
            ]

async def call_single_groq_model(model_name: str, query: str):
    if not GROQ_KEY:
        return {"model": model_name, "answer": f"[ERROR] Groq failed → Missing GROQ_API_KEY"}
    payload = {"model": model_name, "messages": [{"role": "user", "content": query}]}
    headers = {"Authorization": f"Bearer {GROQ_KEY}"}
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            res = await client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
            data = res.json()
            if "choices" not in data:
                return {"model": model_name, "answer": f"[ERROR] Missing 'choices' → {data}"}
            ans = data["choices"][0]["message"]["content"]
            return {"model": model_name, "answer": ans}
        except Exception as e:
            return {"model": model_name, "answer": f"[ERROR] Groq failed → {str(e)}"}

async def call_all_groq_models(query: str):
    models = await get_groq_models()
    tasks = [call_single_groq_model(m, query) for m in models]
    results = await asyncio.gather(*tasks)
    return results

async def call_openai_gpt4(query: str):
    if not OPENAI_KEY:
        return {"model": "gpt-4o", "answer": "[ERROR] OpenAI failed → Missing OPENAI_API_KEY"}
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(url, json=payload, headers=headers)
            data = resp.json()
            if "choices" not in data:
                return {"model": "gpt-4o", "answer": f"[ERROR] OpenAI response missing choices: {data}"}
            ans = data["choices"][0]["message"]["content"]
            return {"model": "gpt-4o", "answer": ans}
        except Exception as e:
            return {"model": "gpt-4o", "answer": f"[ERROR] OpenAI failed → {str(e)}"}

# -------------------------
# Consolidation & Ranking (with grounding)
# -------------------------
# async def process_and_score_responses(responses, query):
#     # Filter out total API errors but keep partial successes
#     valid_responses = [r for r in responses if not r["answer"].startswith("[ERROR]")]
#     if not valid_responses:
#         return (
#             "All models failed to generate a response. Please check API keys and try again.",
#             responses,
#             "No successful models."
#         )

#     # 1. Get embeddings in parallel
#     embedding_tasks = [get_embedding(r["answer"], GEMINI_KEY) for r in valid_responses]
#     embeddings = await asyncio.gather(*embedding_tasks)

#     for i, r in enumerate(valid_responses):
#         r["embedding"] = embeddings[i]

# # 2. Calculate consistency scores properly
#     # 2. Calculate consistency scores properly
#     valid_embeddings = [r.get("embedding") for r in valid_responses]
    
#     # Calculate mean embedding for CACE
#     valid_vecs = [e for e in valid_embeddings if e]
#     mean_embedding = None
#     if valid_vecs:
#         # Calculate element-wise mean
#         dim = len(valid_vecs[0])
#         mean_embedding = [sum(col) / len(valid_vecs) for col in zip(*valid_vecs)]
    
#     cace_distances = []

#     for i, r in enumerate(valid_responses):
#         # Hard-fail: empty answer
#         if not r["answer"] or not r["answer"].strip():
#             r["hard_fail"] = True
#             r["consistency_score"] = 0.05
#             r["cace_distance"] = 1.0
#             cace_distances.append(1.0)
#             continue

#         # Calculate consistency using all valid embeddings
#         r["consistency_score"] = calculate_consistency(valid_embeddings, i)
        
#         # Distance to consolidated embedding
#         if mean_embedding and r.get("embedding"):
#             dist = semantic_distance(r["embedding"], mean_embedding)
#             r["cace_distance"] = dist
#             cace_distances.append(dist)
#         else:
#             r["cace_distance"] = 1.0
#             cace_distances.append(1.0)

#     query_cace = calculate_cace(cace_distances)

#     # 3. LLM judge (Gemini + OpenAI) AND external fact-check (SERP / Wiki)
#     eval_tasks_gemini = [evaluate_response_metrics(query, r["answer"]) for r in valid_responses]
#     eval_tasks_openai = [evaluate_response_metrics_openai(query, r["answer"]) for r in valid_responses]
#     fact_tasks = [external_fact_check_combined(r["answer"]) for r in valid_responses]
    
#     eval_results_gemini, eval_results_openai, fact_results = await asyncio.gather(
#         asyncio.gather(*eval_tasks_gemini),
#         asyncio.gather(*eval_tasks_openai),
#         asyncio.gather(*fact_tasks)
#     )

#     # 4. Final scoring with grounding & hard-fail gating
#     ranked = []
#     for i, r in enumerate(valid_responses):
#         metrics_gemini = eval_results_gemini[i]
#         metrics_openai = eval_results_openai[i]
#         external_conf = fact_results[i]  # [0.05 - 0.99]

#         # Merge Gemini judge factual_accuracy with external evidence (weighted)
#         # Weighted: 40% OpenAI, 20% Gemini, 40% external fact-check
#         merged_factual = (0.4 * metrics_openai.get("factual_accuracy", 0.05)) + \
#                          (0.2 * metrics_gemini.get("factual_accuracy", 0.05)) + \
#                          (0.4 * external_conf)
        
#         # Use average of other metrics for now, or prefer OpenAI?
#         # Let's average reasoning and verification for robustness
#         avg_reasoning = (metrics_gemini.get("reasoning_depth", 0.05) + metrics_openai.get("reasoning_depth", 0.05)) / 2
#         avg_verification = (metrics_gemini.get("source_verification", 0.05) + metrics_openai.get("source_verification", 0.05)) / 2
        
#         metrics = {
#             "factual_accuracy": merged_factual, # Use merged as the main factual accuracy
#             "reasoning_depth": avg_reasoning,
#             "source_verification": avg_verification
#         }
        
#         metrics["factual_accuracy_grounded"] = merged_factual

#         # Prepare metrics for CARS
#         structure_val = calculate_structure_score(r["answer"])
#         cars_metrics = {
#             "factualAccuracy_judge": metrics.get("factual_accuracy", 0.05),
#             "factualAccuracy_grounded": merged_factual,
#             "reasoningDepth": metrics.get("reasoning_depth", 0.05),
#             "external_confidence": external_conf,
#             "consistency": r.get("consistency_score", 0.05),
#             "sourceVerification": metrics.get("source_verification", 0.05),
#             "structure_score": structure_val
#         }

#         # Calculate CARS
#         cars_score = compute_CARS(cars_metrics, r["answer"])

#         # Only eliminate if CARS returned -1 (garbage filter failed)
#         # Don't eliminate just because embeddings are missing - that's too harsh
#         eliminated = False
#         if cars_score < 0:
#             eliminated = True
#             cars_score = 0.0 # Set to 0 for display purposes, but mark as eliminated

#         # If hard_fail (empty answer), eliminate
#         if r.get("hard_fail"):
#             eliminated = True
#             cars_score = 0.0

#         r_out = {
#             "modelName": r["model"],
#             "answer": r["answer"],
#             "factualAccuracy_judge": metrics.get("factual_accuracy", 0.0),
#             "factualAccuracy_grounded": merged_factual,
#             "external_confidence": external_conf,
#             "reasoningDepth": metrics.get("reasoning_depth", 0.0),
#             "consistency": r.get("consistency_score", 0.0),
#             "sourceVerification": metrics.get("source_verification", 0.0),
#             "CARS": cars_score,
#             "cacePerQuery": query_cace,
#             "had_embedding": bool(r.get("embedding")),
#             "hard_fail": bool(r.get("hard_fail", False)),
#             "eliminated": eliminated
#         }
#         ranked.append(r_out)

#     # Sort and return best
#     # Filter out eliminated candidates for selection
#     candidates = [r for r in ranked if not r["eliminated"]]
    
#     if not candidates:
#         # Fallback: if all eliminated, pick the one with highest grounded accuracy, or just the first one
#         # But user said "Zero-value garbage models are eliminated".
#         # We will try to pick the "least bad" one if all are eliminated, or return a specific message.
#         # For now, let's pick the one with highest factualAccuracy_grounded from the eliminated ones.
#         if ranked:
#              candidates = ranked # Fallback to all if none passed
#              candidates.sort(key=lambda x: x["factualAccuracy_grounded"], reverse=True)
#              best = candidates[0]
#              best_answer_text = f"[WARNING: All models failed quality checks. Best available shown]\\n\\n{best['answer']}"
#         else:
#              return "No valid responses generated.", [], "No successful models."
#     else:
#         # Sort candidates by CARS
#         candidates.sort(key=lambda x: x["CARS"], reverse=True)
#         best = candidates[0]
#         best_answer_text = f"Most Reliable Synthesized Answer (Meta-AI - CARS {best['CARS']:.2f}):\\n\\n{best['answer']}"

#     # We still return 'ranked' (all of them) for the frontend to show, 
#     # but we might want to indicate which were eliminated.
#     # The frontend might need to know which one is 'best'.
#     # We'll ensure the 'ranked' list is sorted by CARS descending, with eliminated ones at the bottom.
#     ranked.sort(key=lambda x: (not x["eliminated"], x["CARS"]), reverse=True)

#     summary = f"""
# Best Model: {best['modelName']}

# Scores:
# - CARS: {best['CARS']:.2f}
# - Factual Accuracy (grounded): {best['factualAccuracy_grounded']:.2f}
# - External Confidence: {best['external_confidence']:.2f}
# - Reasoning Depth: {best['reasoningDepth']:.2f}
# - Consistency: {best['consistency']:.2f}
# - CACE (Query Entropy): {best['cacePerQuery']:.2f}
# """

#     return best_answer_text, ranked, summary

# # -------------------------
# # Main Endpoint
# # -------------------------
# @router.post("/chatapi")
# async def chat_api(request: ChatRequest):
#     query = request.sanitize_query()  # Sanitize input
#     try:
#         start = time.time()
#         # Run Gemini + Groq + OpenAI models in parallel
#         gemini_task = call_gemini(query)
#         groq_task = call_all_groq_models(query)
#         openai_task = call_openai_gpt4(query)
        
#         gemini_resp, groq_resps, openai_resp = await asyncio.gather(gemini_task, groq_task, openai_task)
        
#         all_outputs = [gemini_resp] + groq_resps + [openai_resp]
#         best_answer, ranked, summary = await process_and_score_responses(all_outputs, query)
#         return {
#             "query": query,
#             "best_answer": best_answer,
#             "ranked_models": ranked,
#             "summary": summary,
#             "llm_count": len(all_outputs),
#             "time_taken": round(time.time() - start, 2),
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# class EvaluateRequest(BaseModel):
#     query: str
#     response: str


# # -----------------------------------------
# # OpenAI Evaluator (60%)
# # -----------------------------------------

# async def evaluate_with_openai(query: str, response: str) -> Dict[str, float]:
#     """
#     Evaluate hallucination, consistency, fake news, factual accuracy using OpenAI /v1/responses.
#     Returns scores in 0.0 - 1.0 range.
#     """

#     # Fallback if key not set
#     if not OPENAI_KEY:
#         return {
#             "hallucination": 0.5,
#             "consistency": 0.5,
#             "fake_news_probability": 0.5,
#             "factual_accuracy": 0.5
#         }

#     prompt = f"""
#     Evaluate the following AI response to the query: "{query}"

#     Response: "{response}"

#     Provide a JSON object with these fields:
#     - hallucination_score (0.0 = no hallucination, 1.0 = pure hallucination)
#     - consistency_score (1.0 = consistent, 0.0 = contradictory)
#     - fake_news_probability (0.0 = true, 1.0 = fake)
#     - factual_accuracy_score (1.0 = accurate, 0.0 = wrong)

#     Return ONLY the JSON. No explanation.
#     """

#     url = "https://api.openai.com/v1/responses"

#     headers = {
#         "Authorization": f"Bearer {OPENAI_KEY}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": "gpt-4o-mini",
#         "input": prompt,
#         "temperature": 0.2
#     }

#     async with httpx.AsyncClient(timeout=15) as client:
#         try:
#             resp = await client.post(url, json=payload, headers=headers)
#             data = resp.json()

#             # Extract text content from new schema
#             text_raw = data["output"][0]["content"][0]["text"]

#             # Cleanup
#             text_clean = text_raw.replace("```json", "").replace("```", "").strip()

#             # Parse JSON
#             metrics = json.loads(text_clean)

#             return {
#                 "hallucination": float(metrics.get("hallucination_score", 0.5)),
#                 "consistency": float(metrics.get("consistency_score", 0.5)),
#                 "fake_news_probability": float(metrics.get("fake_news_probability", 0.5)),
#                 "factual_accuracy": float(metrics.get("factual_accuracy_score", 0.5)),
#             }

#         except Exception as e:
#             print("OpenAI Evaluation Error:", e)
#             return {
#                 "hallucination": 0.5,
#                 "consistency": 0.5,
#                 "fake_news_probability": 0.5,
#                 "factual_accuracy": 0.5
#             }


# # -----------------------------------------
# # Grounded Dummy Model (40%)
# # -----------------------------------------

# def evaluate_with_grounded_dummy(query: str, response: str) -> Dict[str, float]:
#     """
#     Deterministic heuristic-based grounded evaluator.
#     """

#     seed_val = sum(ord(c) for c in response[:50])

#     # Length-based
#     length_score = min(1.0, len(response) / 500)

#     # Keyword overlap
#     q_words = set(query.lower().split())
#     r_words = set(response.lower().split())
#     overlap = len(q_words.intersection(r_words))
#     keyword_score = min(1.0, overlap / max(1, len(q_words)))

#     consistency = 0.7 + (0.2 * length_score)
#     fake_news = 0.3 - (0.2 * keyword_score)
#     factual = 0.5 + (0.4 * keyword_score)
#     hallucination = 1.0 - factual

#     noise = (seed_val % 10) / 100.0  # 0.00 – 0.09

#     return {
#         "hallucination": max(0.0, min(1.0, hallucination + noise)),
#         "consistency": max(0.0, min(1.0, consistency - noise)),
#         "fake_news_probability": max(0.0, min(1.0, fake_news + noise)),
#         "factual_accuracy": max(0.0, min(1.0, factual - noise)),
#     }


# # -----------------------------------------
# # Hybrid Endpoint
# # -----------------------------------------

# @router.post("/evaluate")
# async def evaluate_response(request: EvaluateRequest):
#     """
#     Hybrid evaluation:
#     - 60% OpenAI
#     - 40% Grounded Dummy
#     """

#     query = request.query
#     response = request.response

#     # Run OpenAI evaluation async
#     openai_task = evaluate_with_openai(query, response)
#     grounded_scores = evaluate_with_grounded_dummy(query, response)

#     openai_scores = await openai_task

#     # Weighted combine
#     def combine(key):
#         return round(
#             (0.6 * openai_scores.get(key, 0.5)) +
#             (0.4 * grounded_scores.get(key, 0.5)),
#             2
#         )

#     final_hallucination = combine("hallucination")
#     final_consistency = combine("consistency")
#     final_fake_news = combine("fake_news_probability")
#     final_factual = combine("factual_accuracy")

#     # Composite AI Reliability Score (CARS)
#     cars_score = (
#         0.4 * final_factual +
#         0.3 * final_consistency +
#         0.3 * (1.0 - final_fake_news)
#     )
#     cars_score = round(max(0.0, min(1.0, cars_score)), 2)

#     return {
#         "cars": cars_score,
#         "consistency": final_consistency,
#         "fake_news_probability": final_fake_news,
#         "factual_accuracy": final_factual,
#         "hallucination": final_hallucination,
#         "details": {
#             "query": query,
#             "response": response[:150] + "..." if len(response) > 150 else response,
#             "openai_raw": openai_scores,
#             "grounded_raw": grounded_scores
#         }
#     }