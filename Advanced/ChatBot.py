"""
Chatbot Development
====================
A retrieval-augmented generative chatbot with:
- Intent classification (rule-based + ML)
- Named Entity Recognition (spaCy-lite)
- Context window management
- Sentiment-aware responses
- Knowledge base (FAQ retrieval via TF-IDF)
- Fallback to generative templates
"""

import re
import json
import random
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

random.seed(42)

# ── Knowledge base ────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "greeting":    ["Hello! How can I help you today?",
                    "Hi there! What can I assist you with?",
                    "Hey! Great to see you. What's on your mind?"],
    "farewell":    ["Goodbye! Have a wonderful day!",
                    "See you later! Feel free to come back anytime.",
                    "Take care! It was great chatting with you."],
    "thanks":      ["You're welcome!", "Happy to help!", "Anytime! 😊"],
    "hours":       ["We're open Monday–Friday, 9 AM – 6 PM.",
                    "Our business hours are 9 AM to 6 PM, Mon–Fri."],
    "location":    ["We're located at 123 Main Street, Tech City.",
                    "Our office is at 123 Main Street. See you there!"],
    "price":       ["Our pricing starts at $29/month. Visit our website for details.",
                    "Plans start from $29/month. Would you like a demo?"],
    "refund":      ["Refunds are processed within 5–7 business days.",
                    "We offer a 30-day money-back guarantee!"],
    "contact":     ["You can reach us at support@example.com or call 555-0100.",
                    "Email us: support@example.com. We respond within 24 hours."],
}

FAQ_QA = [
    ("What are your business hours?",         "hours"),
    ("When are you open?",                    "hours"),
    ("Where is your office?",                 "location"),
    ("How much does it cost?",                "price"),
    ("What is the pricing?",                  "price"),
    ("Can I get a refund?",                   "refund"),
    ("How do I contact support?",             "contact"),
    ("How can I reach you?",                  "contact"),
    ("What is your return policy?",           "refund"),
    ("Do you have a free trial?",             "price"),
]

SMALL_TALK = {
    r"how are you": "I'm doing great, thanks for asking! How about you?",
    r"what('s| is) your name": "I'm ChatBot 3000, your friendly assistant!",
    r"are you (a |an )?ai|are you (a |an )?robot": "Yes, I'm an AI chatbot. But I do my best to be helpful!",
    r"what can you do|help": ("I can answer FAQs, tell you about our services, hours, "
                               "pricing, and more. Just ask!"),
    r"(joke|funny|laugh)": random.choice([
        "Why don't scientists trust atoms? Because they make up everything! 😄",
        "I told my computer I needed a break... now it won't stop sending me Kit-Kat ads. 😂",
        "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
    ]),
}

# ── Intent classifier ─────────────────────────────────────────────────────────
INTENT_PATTERNS = {
    "greeting": r"\b(hi|hello|hey|good morning|good afternoon|howdy|sup)\b",
    "farewell":  r"\b(bye|goodbye|see you|later|exit|quit|farewell)\b",
    "thanks":    r"\b(thanks|thank you|thx|ty|appreciate)\b",
    "hours":     r"\b(hour|open|close|time|when)\b",
    "location":  r"\b(where|location|address|office|place)\b",
    "price":     r"\b(price|cost|how much|plan|fee|charge|pay|money)\b",
    "refund":    r"\b(refund|return|money back|cancel|guarantee)\b",
    "contact":   r"\b(contact|email|phone|call|reach|support)\b",
}

# ── TF-IDF retriever ──────────────────────────────────────────────────────────
faq_questions = [q for q, _ in FAQ_QA]
faq_intents   = [i for _, i in FAQ_QA]
vectorizer    = TfidfVectorizer()
faq_matrix    = vectorizer.fit_transform(faq_questions)

def retrieve_intent(text, threshold=0.3):
    vec   = vectorizer.transform([text])
    sims  = cosine_similarity(vec, faq_matrix).flatten()
    best  = np.argmax(sims)
    if sims[best] >= threshold:
        return faq_intents[best], sims[best]
    return None, 0.0

# ── Sentiment (simple rule-based) ────────────────────────────────────────────
POS_WORDS = {"good","great","love","excellent","awesome","wonderful","nice","happy","thanks","fantastic"}
NEG_WORDS = {"bad","hate","terrible","awful","worst","angry","frustrated","upset","poor","horrible"}

def get_sentiment(text):
    words = set(text.lower().split())
    pos   = len(words & POS_WORDS)
    neg   = len(words & NEG_WORDS)
    if pos > neg: return "positive"
    if neg > pos: return "negative"
    return "neutral"

# ── Context manager ───────────────────────────────────────────────────────────
class ConversationContext:
    def __init__(self, window=5):
        self.history  = []
        self.window   = window
        self.user_name = None
        self.last_intent = None

    def add(self, role, text, intent=None):
        self.history.append({"role": role, "text": text, "time": datetime.now(), "intent": intent})
        if len(self.history) > self.window * 2:
            self.history = self.history[-self.window * 2:]
        if intent:
            self.last_intent = intent

    def summary(self):
        return [(m["role"], m["text"]) for m in self.history[-4:]]

# ── Response generator ────────────────────────────────────────────────────────
def generate_response(user_input, ctx: ConversationContext):
    text_lower = user_input.lower().strip()

    # Small talk
    for pattern, response in SMALL_TALK.items():
        if re.search(pattern, text_lower):
            return response, "small_talk"

    # Name detection
    m = re.search(r"my name is (\w+)", text_lower)
    if m:
        ctx.user_name = m.group(1).capitalize()
        return f"Nice to meet you, {ctx.user_name}! How can I help?", "greeting"

    # Rule-based intent
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, text_lower):
            reply = random.choice(KNOWLEDGE_BASE[intent])
            if ctx.user_name and intent == "greeting":
                reply = f"Hello again, {ctx.user_name}! " + reply
            return reply, intent

    # TF-IDF retrieval
    intent, score = retrieve_intent(text_lower)
    if intent:
        return random.choice(KNOWLEDGE_BASE[intent]) + f" (confidence: {score:.2f})", intent

    # Sentiment-based fallback
    sentiment = get_sentiment(text_lower)
    if sentiment == "negative":
        return ("I'm sorry to hear that. Let me connect you with a human agent — "
                "please email support@example.com"), "fallback"
    if sentiment == "positive":
        return "That's great to hear! Is there anything else I can help you with?", "fallback"

    return ("I'm not sure I understood that. Could you rephrase? "
            "You can ask about our hours, location, pricing, or contact info."), "fallback"

# ── Main chat loop ─────────────────────────────────────────────────────────────
def run_chatbot():
    ctx = ConversationContext()
    print("\n" + "=" * 60)
    print("  CHATBOT 3000  |  Type 'quit' to exit")
    print("=" * 60)
    print("Bot: Hello! I'm ChatBot 3000. How can I assist you today?\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! 👋")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot:", random.choice(KNOWLEDGE_BASE["farewell"]))
            break

        response, intent = generate_response(user_input, ctx)
        ctx.add("user", user_input, intent)
        ctx.add("bot", response, intent)
        print(f"Bot: {response}\n")

    print("\n── Conversation Summary ──")
    for role, text in ctx.summary():
        print(f"  [{role.upper()}] {text}")

# ── Demo (non-interactive) ────────────────────────────────────────────────────
def run_demo():
    ctx = ConversationContext()
    demo_inputs = [
        "Hi there!",
        "My name is Alex",
        "What are your business hours?",
        "How much does it cost?",
        "Can I get a refund?",
        "How can I contact support?",
        "Tell me a joke",
        "Thanks!",
        "Bye",
    ]
    print("\n── Demo Mode ──")
    for inp in demo_inputs:
        response, intent = generate_response(inp, ctx)
        ctx.add("user", inp, intent)
        ctx.add("bot", response, intent)
        print(f"  You : {inp}")
        print(f"  Bot : {response}  [intent={intent}]\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        run_demo()           # default to demo for non-interactive environments
        print("\n[Tip] For interactive mode run: python chatbot.py")
