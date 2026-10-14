"""
ML features for the WhatsApp Chat Analyzer
  1. Sentiment analysis   -> TF-IDF (word + char n-grams) + Logistic Regression
  2. Toxicity detection   -> TF-IDF + Logistic Regression
  3. Spam detection       -> TF-IDF + Logistic Regression
  4. Next-word prediction -> Trigram language model with stupid-backoff
"""
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

import helper
import ml_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "ml_data")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
SAFE_DOMAINS = ("youtube.com", "youtu.be", "google.com", "forms.gle", "instagram.com", "github.com",
                "wikipedia.org", "zoom.us", "linkedin.com", "whatsapp.com", "facebook.com", "twitter.com", "x.com")


# =============================================================== text cleaning
def _url_token(match):
    url = match.group(0)
    return " safeurl " if any(d in url for d in SAFE_DOMAINS) else " urltoken "


def clean_text(text: str) -> str:
    text = URL_RE.sub(_url_token, str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


# =============================================================== generic classifier
def _make_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced")
    return Pipeline([("tfidf", features), ("clf", clf)])


def _train(texts, labels, seed_count=0):
    """Train on all data, evaluate on a held-out 20% split. Returns (model, metrics)."""
    texts = [clean_text(t) for t in texts]
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)
    eval_model = _make_pipeline().fit(X_tr, y_tr)
    pred = eval_model.predict(X_te)
    avg = "binary" if len(set(labels)) == 2 else "macro"
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average=avg, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(p), "recall": float(r), "f1": float(f),
        "confusion": confusion_matrix(y_te, pred, labels=sorted(set(labels))).tolist(),
        "average": avg,
        "train_size": len(X_tr), "test_size": len(X_te), "total_size": len(texts),
        "seed_count": seed_count,
    }
    final_model = _make_pipeline().fit(texts, labels)  # refit on everything for deployment
    return final_model, metrics


def _repeat(items, n):
    return list(items) * n


# =============================================================== dataset builders
def _sentiment_data():
    texts, labels = [], []
    path = os.path.join(DATA_DIR, "tweets_sentiment.csv")
    if os.path.exists(path):
        d = pd.read_csv(path).dropna()
        texts += d["text"].tolist(); labels += (d["label"] * 2).tolist()  # 0 = negative, 2 = positive
    # neutral class = everyday SMS/chat messages (ham) + hand-written neutral Hinglish chat
    sms = os.path.join(DATA_DIR, "sms_spam.tsv")
    if os.path.exists(sms):
        s = pd.read_csv(sms, sep="\t", header=None, names=["label", "text"]).dropna()
        ham = s[s["label"] == "ham"].sample(n=min(2500, (s["label"] == "ham").sum()), random_state=1)["text"].tolist()
        texts += ham; labels += [1] * len(ham)
    seed_pos = _repeat(ml_data.POSITIVE_CHAT, 6)
    seed_neg = _repeat(ml_data.NEGATIVE_CHAT, 6) + _repeat(ml_data.TOXIC_EXTRA, 2)
    seed_neu = _repeat(ml_data.NORMAL_CHAT, 6) + _repeat(ml_data.SAFE_LINK_CHAT, 3)
    texts += seed_pos + seed_neg + seed_neu
    labels += [2] * len(seed_pos) + [0] * len(seed_neg) + [1] * len(seed_neu)
    return texts, labels, len(seed_pos) + len(seed_neg) + len(seed_neu)


def _toxicity_data():
    toxic = [t.format(w=w) for t in ml_data.TOXIC_TEMPLATES for w in ml_data.TOXIC_WORDS]
    toxic += ml_data.TOXIC_EXTRA * 3
    random.Random(7).shuffle(toxic)
    toxic = toxic[:400]
    nice = [t.format(w=w) for t in ml_data.TOXIC_TEMPLATES for w in ml_data.NICE_WORDS]
    random.Random(8).shuffle(nice)
    non_toxic = (_repeat(ml_data.NORMAL_CHAT, 5) + _repeat(ml_data.POSITIVE_CHAT, 3)
                 + _repeat(ml_data.NEGATIVE_CHAT, 3) + nice[:300] + _repeat(ml_data.SAFE_LINK_CHAT, 3))
    texts = toxic + non_toxic
    labels = [1] * len(toxic) + [0] * len(non_toxic)
    return texts, labels, len(texts)


SPAM_SEED = [
    "Jio free recharge ke liye is link par click karo http://bit.ly/free-recharge",
    "Ghar baithe paise kamao daily 5000 rupay, abhi join karo http://earn-now.xyz",
    "Congratulations! You have won a free iPhone. Click here to claim http://claim-prize.xyz",
    "Join our telegram group for guaranteed profit, limited seats http://t.me/profit",
    "Forwarded many times: share this message with 10 groups to get free data",
    "Lucky draw winner! Aapne 10 lakh jeete hain, apna bank details bhejo",
    "Urgent: your account will be blocked, verify now http://secure-bank-login.xyz",
    "Loan approved in 5 minutes without documents, call now 9999999999",
    "Big sale 90% off today only, order now http://shop-deal.xyz",
    "Earn money online from mobile, no investment, click link to start",
    "Free Netflix subscription for 1 year, click this link to get http://free-netflix.xyz",
    "Hurry! Offer ends tonight. Buy 1 get 3 free, whatsapp us now",
]


def _spam_data():
    texts, labels = [], []
    path = os.path.join(DATA_DIR, "sms_spam.tsv")
    if os.path.exists(path):
        d = pd.read_csv(path, sep="\t", header=None, names=["label", "text"]).dropna()
        texts += d["text"].tolist(); labels += (d["label"] == "spam").astype(int).tolist()
    spam = _repeat(SPAM_SEED, 8)
    ham = _repeat(ml_data.NORMAL_CHAT, 6) + _repeat(ml_data.SAFE_LINK_CHAT, 8) + _repeat(ml_data.POSITIVE_CHAT, 2)
    texts += spam + ham
    labels += [1] * len(spam) + [0] * len(ham)
    return texts, labels, len(spam) + len(ham)


# =============================================================== public: train all
def train_all():
    models = {}
    for name, builder in [("sentiment", _sentiment_data), ("toxicity", _toxicity_data), ("spam", _spam_data)]:
        texts, labels, seed_n = builder()
        model, metrics = _train(texts, labels, seed_n)
        models[name] = {"model": model, "metrics": metrics}
    return models


# =============================================================== predictions
SENTIMENT_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
_HAS_SIGNAL = re.compile(r"[^\W\d_]|[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def predict_sentiment_texts(models, texts):
    """Returns (score, label). score = P(positive) - P(negative), in [-1, +1]."""
    cleaned = [clean_text(t) for t in texts]
    proba = models["sentiment"]["model"].predict_proba(cleaned)  # columns: 0=neg, 1=neu, 2=pos
    scores = proba[:, 2] - proba[:, 0]
    labels = [SENTIMENT_NAMES[i] for i in proba.argmax(axis=1)]
    for i, t in enumerate(cleaned):   # numbers-only / empty messages carry no sentiment
        if not _HAS_SIGNAL.search(t):
            scores[i], labels[i] = 0.0, "Neutral"
    return scores, labels


def predict_prob(models, kind, texts):
    cleaned = [clean_text(t) for t in texts]
    return models[kind]["model"].predict_proba(cleaned)[:, 1]


def analyze_chat(models, selected_user, df, toxic_th=0.6, spam_th=0.7):
    """Runs sentiment + toxicity + spam on every real text message."""
    temp = helper._text_df(selected_user, df)[["user", "date", "year", "month_num", "month", "message"]].copy()
    if temp.empty:
        return temp
    msgs = temp["message"].tolist()
    temp["sentiment_score"], temp["sentiment"] = predict_sentiment_texts(models, msgs)
    temp["toxic_prob"] = predict_prob(models, "toxicity", msgs)
    temp["is_toxic"] = temp["toxic_prob"] >= toxic_th
    temp["spam_prob"] = predict_prob(models, "spam", msgs)
    temp["is_spam"] = temp["spam_prob"] >= spam_th
    return temp.reset_index(drop=True)


# =============================================================== next-word model
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?|[^\w\s]", re.UNICODE)
BOS, EOS = "<s>", "</s>"


def tokenize(text):
    text = URL_RE.sub(" ", str(text).lower())
    return _TOKEN_RE.findall(text)


class NextWordModel:
    """Trigram language model with stupid-backoff (Brants et al., 2007)."""

    def __init__(self, backoff=0.4):
        self.backoff = backoff
        self.uni, self.bi, self.tri = Counter(), defaultdict(Counter), defaultdict(Counter)
        self.total = 0
        self.n_messages = 0

    def fit(self, messages):
        for m in messages:
            toks = tokenize(m)
            if not toks:
                continue
            self.n_messages += 1
            toks = [BOS, BOS] + toks + [EOS]
            for i in range(2, len(toks)):
                w = toks[i]
                self.uni[w] += 1
                self.total += 1
                self.bi[toks[i - 1]][w] += 1
                self.tri[(toks[i - 2], toks[i - 1])][w] += 1
        self.vocab = len(self.uni)
        return self

    def _scores(self, context):
        c1 = context[-1] if context else BOS
        c2 = context[-2] if len(context) > 1 else BOS
        scores = {}
        tri = self.tri.get((c2, c1))
        if tri:
            denom = sum(tri.values())
            for w, c in tri.items():
                scores[w] = c / denom
        bi = self.bi.get(c1)
        if bi:
            denom = sum(bi.values())
            for w, c in bi.items():
                scores.setdefault(w, self.backoff * c / denom)
        for w, c in self.uni.most_common(30):
            scores.setdefault(w, self.backoff ** 2 * c / self.total)
        scores.pop(BOS, None)
        return scores

    def predict(self, text, k=5):
        scores = self._scores(tokenize(text))
        top = sorted(scores.items(), key=lambda x: -x[1])[: k + 1]
        top = [(w, s) for w, s in top if w != EOS][:k]
        total = sum(s for _, s in top) or 1.0
        return [(w, s / total) for w, s in top]

    def generate(self, text="", max_words=15, seed=None):
        rng = random.Random(seed)
        toks = tokenize(text)
        out = []
        for _ in range(max_words):
            scores = self._scores(toks + out)
            top = sorted(scores.items(), key=lambda x: -x[1])[:8]
            words, weights = zip(*top)
            nxt = rng.choices(words, weights=weights, k=1)[0]
            if nxt == EOS:
                break
            out.append(nxt)
        return " ".join(toks + out)

    def evaluate(self, messages, max_messages=2000):
        """Top-1 / Top-3 accuracy of predicting each next word."""
        hit1 = hit3 = n = 0
        for m in list(messages)[:max_messages]:
            toks = tokenize(m)
            for i in range(1, len(toks)):
                preds = [w for w, _ in self.predict(" ".join(toks[:i]), k=3)]
                n += 1
                hit1 += toks[i] == preds[0] if preds else 0
                hit3 += toks[i] in preds
        return {"top1": hit1 / n if n else 0.0, "top3": hit3 / n if n else 0.0, "tested_words": n}


def build_next_word_model(selected_user, df, test_size=0.2):
    temp = helper._text_df(selected_user, df)
    msgs = temp["message"].tolist()
    if len(msgs) < 30:
        return None, None
    train, test = train_test_split(msgs, test_size=test_size, random_state=42)
    stats = NextWordModel().fit(train).evaluate(test)
    final = NextWordModel().fit(msgs)  # final model uses all messages
    stats.update({"messages": len(msgs), "vocab": final.vocab})
    return final, stats