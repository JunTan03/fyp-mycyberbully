# =============================
# 🚀 Optimized model.py
# =============================
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import re, joblib, os
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from concurrent.futures import ThreadPoolExecutor

from model.slangdict import slangdict as eng_slang
from model.malayslangdict import malayslangdict as malay_slang

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'model'))

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models + Tokenizers + Label Encoders
sentiment_model = DistilBertForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, 'sentiment_model')).to(device)
sentiment_tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(MODEL_DIR, 'sentiment_tokenizer'))

cyberbullying_model = BertForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, 'cyberbullying_model')).to(device)
cyberbullying_tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_DIR, 'cyberbullying_tokenizer'))

label_encoder1 = joblib.load(os.path.join(MODEL_DIR, 'cyberbullying_model/cyber_label_encoder.pkl'))
label_encoder2 = joblib.load(os.path.join(MODEL_DIR, 'sentiment_label_encoder.pkl'))

sentiment_model.eval()
cyberbullying_model.eval()

# Ekphrasis Preprocessor
text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date', 'number'],
    annotate=set(),
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)

# Text utilities
def normalize_slang(text):
    words = text.split()
    words = [malay_slang.get(w, w) for w in words]
    words = [eng_slang.get(w, w) for w in words]
    return " ".join(words)

def clean_text_basic(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text.strip()).lower()

# Fast simple Malay-English detection
def fast_language_filter(text):
    malay_keywords = ["saya", "awak", "tidak", "boleh", "akan", "dengan", "pergi", "datang"]
    if any(w in text.lower() for w in malay_keywords):
        return "ms"
    return "en"

# Preprocessing Main
def preprocess_text(text):
    if not text.strip():
        return text
    text = clean_text_basic(text)
    lang = fast_language_filter(text)
    if lang == "ms":
        return normalize_slang(text)
    text = " ".join(text_processor.pre_process_doc(text))
    return normalize_slang(text)

# Parallel Preprocessing
def preprocess_batch(text_list):
    with ThreadPoolExecutor(max_workers=8) as executor:
        cleaned_texts = list(executor.map(preprocess_text, text_list))
    return cleaned_texts

# Label decoding
def decode_cyberbullying_label(pred):
    return label_encoder1.inverse_transform([pred])[0]

def decode_sentiment_label(pred):
    return label_encoder2.inverse_transform([pred])[0]

# Batch tweet classifier (optimized)
def classify_tweet_batch(text_list, sentiment_model, sentiment_tokenizer, cyber_model, cyber_tokenizer):
    cleaned_texts = preprocess_batch(text_list)

    sentiment_inputs = sentiment_tokenizer(cleaned_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    sentiment_inputs = {k: v.to(device) for k, v in sentiment_inputs.items()}

    with torch.no_grad():
        sentiment_logits = sentiment_model(**sentiment_inputs).logits
    sentiment_preds = torch.argmax(sentiment_logits, axis=1).tolist()

    sentiment_labels = [decode_sentiment_label(pred) for pred in sentiment_preds]
    non_positive_indices = [i for i, label in enumerate(sentiment_labels) if label != "Positive"]

    if not non_positive_indices:
        return [(sentiment_labels[i], "Not applicable") for i in range(len(sentiment_labels))]

    non_positive_texts = [cleaned_texts[i] for i in non_positive_indices]

    cyber_inputs = cyberbullying_tokenizer(non_positive_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    cyber_inputs = {k: v.to(device) for k, v in cyber_inputs.items()}

    with torch.no_grad():
        cyber_logits = cyber_model(**cyber_inputs).logits
    cyber_preds = torch.argmax(cyber_logits, axis=1).tolist()
    cyber_labels = [decode_cyberbullying_label(pred) for pred in cyber_preds]

    results = []
    non_positive_counter = 0
    for i in range(len(cleaned_texts)):
        if sentiment_labels[i] == "Positive":
            results.append((sentiment_labels[i], "Not applicable"))
        else:
            results.append((sentiment_labels[i], cyber_labels[non_positive_counter]))
            non_positive_counter += 1

    return results

# Exports
__all__ = [
    "sentiment_model", "sentiment_tokenizer",
    "cyberbullying_model", "cyberbullying_tokenizer",
    "classify_tweet_batch", "preprocess_text"
]
