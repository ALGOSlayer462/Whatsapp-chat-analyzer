"""
Ek baar chalao:  python download_data.py
Ye ml_data/ folder banata hai aur 2 dataset files download karta hai:
  - ml_data/sms_spam.tsv         (SMS Spam Collection)
  - ml_data/tweets_sentiment.csv (NLTK twitter_samples: 5000 positive + 5000 negative tweets)
"""
import csv
import os
import re
import urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_data")
os.makedirs(OUT, exist_ok=True)

# 1) SMS spam dataset
SMS_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
urllib.request.urlretrieve(SMS_URL, os.path.join(OUT, "sms_spam.tsv"))
print("sms_spam.tsv ready")

# 2) Tweets sentiment dataset (via nltk)
import nltk
nltk.download("twitter_samples", quiet=True)
from nltk.corpus import twitter_samples

rows = []
for label, fid in [(1, "positive_tweets.json"), (0, "negative_tweets.json")]:
    for t in twitter_samples.strings(fid):
        t = re.sub(r"https?://\S+", "", t)
        t = re.sub(r"@\w+", "", t).replace("#", "")
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) > 2:
            rows.append((t, label))

with open(os.path.join(OUT, "tweets_sentiment.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text", "label"])
    w.writerows(rows)
print("tweets_sentiment.csv ready:", len(rows), "rows")