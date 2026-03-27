"""
Natural Language Generation (NLG)
===================================
Demonstrates multiple NLG approaches:
1. Template-based generation (rule-driven)
2. N-gram language model (Markov chain)
3. GPT-2 fine-tuning demo (requires transformers)
4. Text summarisation with extractive methods
"""

import re
import random
import math
import numpy as np
from collections import defaultdict, Counter

random.seed(42)
np.random.seed(42)

print("=" * 60)
print("  NATURAL LANGUAGE GENERATION (NLG)")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# 1. TEMPLATE-BASED NLG
# ══════════════════════════════════════════════════════════════
print("\n[1] Template-Based NLG")

REPORT_TEMPLATE = """
Sales Report — {month} {year}
{'=' * 40}
Total Revenue   : ${revenue:,.2f}
Units Sold      : {units:,}
Top Product     : {top_product}
Growth vs Last Month : {growth:+.1f}%

{summary_sentence}
"""

def generate_sales_report(data):
    growth = data["growth"]
    if growth > 10:
        sentiment = "an exceptional month, significantly exceeding targets"
    elif growth > 0:
        sentiment = "a positive month with steady growth"
    elif growth > -5:
        sentiment = "a relatively flat month; slight optimisation is recommended"
    else:
        sentiment = "a challenging month; immediate strategic review is advised"

    summary = (f"{data['month']} {data['year']} was {sentiment}. "
               f"The top-selling product, {data['top_product']}, "
               f"contributed significantly to the ${data['revenue']:,.0f} revenue total.")
    return REPORT_TEMPLATE.format(**data, summary_sentence=summary)

sample_data = {
    "month": "March", "year": 2026,
    "revenue": 1_245_670.50, "units": 8_432,
    "top_product": "CloudSync Pro", "growth": 14.3,
}
print(generate_sales_report(sample_data))

# ══════════════════════════════════════════════════════════════
# 2. N-GRAM LANGUAGE MODEL (Markov chain)
# ══════════════════════════════════════════════════════════════
print("[2] N-Gram Language Model (Markov Chain)")

CORPUS = """
The quick brown fox jumps over the lazy dog.
A machine learning model learns patterns from data to make predictions.
Natural language processing enables computers to understand human language.
Deep learning uses neural networks with many layers to solve complex tasks.
Data science combines statistics mathematics and programming to extract insights.
The future of artificial intelligence is both exciting and challenging.
Language models predict the next word based on previous context.
Text generation creates new content by sampling from learned distributions.
Transformers revolutionised natural language processing with attention mechanisms.
Large language models demonstrate emergent capabilities at scale.
""".strip()

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_ngram_model(tokens, n=2):
    model = defaultdict(Counter)
    for i in range(len(tokens) - n):
        context = tuple(tokens[i:i + n - 1])
        next_w  = tokens[i + n - 1]
        model[context][next_w] += 1
    return model

def sample_next(model, context):
    options = model.get(context)
    if not options:
        return None
    words  = list(options.keys())
    counts = list(options.values())
    total  = sum(counts)
    probs  = [c / total for c in counts]
    return random.choices(words, weights=probs, k=1)[0]

def generate_text(model, seed_words, n=2, length=30):
    tokens  = list(seed_words)
    context = tuple(tokens[-(n - 1):])
    for _ in range(length):
        nxt = sample_next(model, context)
        if nxt is None:
            break
        tokens.append(nxt)
        context = tuple(tokens[-(n - 1):])
    return " ".join(tokens)

tokens   = tokenize(CORPUS)
bigram   = build_ngram_model(tokens, n=2)
trigram  = build_ngram_model(tokens, n=3)

print("\nBigram generation (seed: 'the'):")
print(" ", generate_text(bigram,  ["the"], n=2, length=20))

print("\nTrigram generation (seed: 'deep learning'):")
print(" ", generate_text(trigram, ["deep", "learning"], n=3, length=20))

print("\nTrigram generation (seed: 'natural language'):")
print(" ", generate_text(trigram, ["natural", "language"], n=3, length=20))

# ── Perplexity ────────────────────────────────────────────────────────────────
def perplexity(model, tokens, n=2):
    log_prob = 0
    count    = 0
    for i in range(n - 1, len(tokens)):
        ctx  = tuple(tokens[i - (n - 1):i])
        word = tokens[i]
        dist = model.get(ctx)
        if dist:
            total = sum(dist.values())
            prob  = (dist.get(word, 0) + 1) / (total + len(dist))  # Laplace smoothing
            log_prob += math.log(prob)
            count += 1
    return math.exp(-log_prob / count) if count > 0 else float("inf")

print(f"\nBigram perplexity  : {perplexity(bigram,  tokens, 2):.2f}")
print(f"Trigram perplexity : {perplexity(trigram, tokens, 3):.2f}")

# ══════════════════════════════════════════════════════════════
# 3. EXTRACTIVE SUMMARISATION
# ══════════════════════════════════════════════════════════════
print("\n[3] Extractive Summarisation (TF-IDF scoring)")

ARTICLE = """
Artificial intelligence has transformed numerous industries over the past decade.
Machine learning algorithms now power recommendation systems, fraud detection, and medical diagnosis.
Natural language processing allows computers to understand and generate human language with remarkable accuracy.
Recent advances in deep learning, particularly transformer architectures, have led to large language models
that can write essays, answer questions, and even generate code.
However, these systems also raise important ethical questions about bias, privacy, and the future of work.
Researchers are actively working on making AI systems more transparent and interpretable.
The development of responsible AI requires collaboration between technologists, policymakers, and the public.
As AI capabilities continue to grow, society must carefully consider how to harness these tools for human benefit.
The coming years will be critical in determining how humanity chooses to develop and deploy artificial intelligence.
"""

def extractive_summarise(text, n_sentences=3):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]

    # TF score
    word_freq = Counter(re.findall(r"\b\w+\b", text.lower()))
    stop = {"the","a","an","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","and","or","but","in",
            "on","at","to","for","of","with","by","from","as","that","this"}
    word_freq = {w: f for w, f in word_freq.items() if w not in stop}
    max_freq   = max(word_freq.values()) if word_freq else 1

    def score(sentence):
        words = re.findall(r"\b\w+\b", sentence.lower())
        return sum(word_freq.get(w, 0) / max_freq for w in words)

    scored = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
    selected = sorted(scored[:n_sentences], key=lambda x: x[0])
    return " ".join(s for _, s in selected)

print("\nOriginal article length:", len(ARTICLE.split()), "words")
summary = extractive_summarise(ARTICLE, n_sentences=3)
print(f"Summary  ({len(summary.split())} words):\n  {summary}")

# ══════════════════════════════════════════════════════════════
# 4. GPT-2 DEMO (optional)
# ══════════════════════════════════════════════════════════════
print("\n[4] GPT-2 Text Generation (requires transformers library)")
try:
    from transformers import pipeline
    print("Loading GPT-2 pipeline…")
    generator = pipeline("text-generation", model="gpt2", max_new_tokens=60, truncation=True)
    prompt = "Artificial intelligence will"
    result = generator(prompt, num_return_sequences=2, do_sample=True, temperature=0.8)
    for i, r in enumerate(result, 1):
        print(f"\n  Sample {i}: {r['generated_text']}")
except ImportError:
    print("  [INFO] transformers not installed. Run: pip install transformers torch")
    print("  GPT-2 generation requires these libraries.\n")

print("\nNLG project complete!")