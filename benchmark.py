import os
import re
import json
import html
from html.parser import HTMLParser

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","of","in","on","at","by",
    "for","from","with","to","into","onto","upon","is","are","was","were","be",
    "been","being","do","does","did","have","has","had","this","that","these",
    "those","it","its","as","than","so","such","because","while","although",
    "about","against","between","during","before","after","above","below",
    "again","further","once",
    "you","your","we","our","i","my","me","us","will","can","no","not","all",
    "one","out","just","may","here","more","any","get","now","new","only","please",
    "enron","ect","hou","vince","kaminski","daren","hpl","corp",
    "escapenumber","escapelong","escapenumbermg",
    "td","tr","font","bgcolor","nbsp","href","img","width","height",
    "style","pt","cc","subject","per", "nextpart", "mime", "multipart", "content", "charset",
    "transfer", "encoding", "quoted", "printable", "boundary",
    "listmaster", "mailman", "listinfo", "unsubscribe","linux", "ilug", "listmaster", "irish", "users", "group",
    "multipart", "format", "plain", "windows", "type", "text",
    "base", "tbit", "tab", "decoration", "none", "multi", "part", "legal", "notice", "iso", "instead",
    "go", "format", "type", "plain", "windows", "text"
}


# -----------------------------
# HTML STRIPPING
# -----------------------------
class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts)


def strip_html(text):
    stripper = HTMLStripper()
    try:
        stripper.feed(html.unescape(text))
        return stripper.get_text()
    except:
        return text


def is_garbage_token(token):
    vowels = set("aeiou")
    alpha_chars = [c for c in token if c.isalpha()]
    digit_chars = [c for c in token if c.isdigit()]

    if len(token) > 1 and len(digit_chars) / len(token) > 0.3:
        return True

    if len(token) <= 2 and any(c.isdigit() for c in token):
        return True

    if len(alpha_chars) >= 3 and not any(c in vowels for c in alpha_chars):
        return True

    return False


# -----------------------------
# LOAD MODEL
# -----------------------------
with open("spam_model.json", "r", encoding="latin-1") as f:
    model = json.load(f)

vocabulary = set(model["vocabulary"])
log_prior_ham = model["priors"]["ham"]
log_prior_spam = model["priors"]["spam"]
ham_log_likelihoods = model["log_likelihoods"]["ham"]
spam_log_likelihoods = model["log_likelihoods"]["spam"]

# -----------------------------
# LOAD BENCHMARK EMAILS
# -----------------------------
benchmark_folder = "datasets/benchmark-testing"

benchmark_files = [
    os.path.join(benchmark_folder, f)
    for f in os.listdir(benchmark_folder)
    if f.endswith(".txt")
]

emails = []
true_labels = []

for file_path in benchmark_files:
    with open(file_path, "r", encoding="latin-1") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and line.lower().startswith("email"):
                continue
            parts = re.split(r"\s+", line)
            label = parts[-1]
            if label not in {"0", "1"}:
                raise ValueError(f"Invalid label detected: {line}")
            email_text = " ".join(parts[:-1])
            emails.append(email_text)
            true_labels.append(label)


# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize_word(word):
    substitutions = {
        "0":"o","1":"i","3":"e","4":"a",
        "$":"s","@":"a","5":"s","7":"t"
    }
    for k, v in substitutions.items():
        word = word.replace(k, v)
    return word


# -----------------------------
# MERGE LETTER SEQUENCES
# -----------------------------
def merge_letter_sequences(tokens):
    merged = []
    buffer = []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buffer.append(t)
        else:
            if len(buffer) > 1:
                merged.append("".join(buffer))
            elif buffer:
                merged.extend(buffer)
            buffer = []
            merged.append(t)
    if len(buffer) > 1:
        merged.append("".join(buffer))
    else:
        merged.extend(buffer)
    return merged


def add_ngrams(tokens, n=4):
    ngrams = []
    for k in range(2, n + 1):
        for i in range(len(tokens) - k + 1):
            ngram = "_".join(tokens[i:i + k])
            ngrams.append(ngram)
    return tokens + ngrams


# -----------------------------
# TOKENIZATION PIPELINE
# -----------------------------
def tokenize_email(email):
    url_pattern = r"(http\S+|www\.\S+)"
    token_pattern = r"[a-zA-Z0-9$@\.]+"

    email = email.lower()
    email = strip_html(email)
    email = re.sub(url_pattern, "<URL>", email)
    email = re.sub(r"(.)\1{2,}", r"\1\1", email)
    tokens = re.findall(token_pattern, email)
    tokens = [t.strip("._-") for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [t for t in tokens if not ("." in t and not t.replace(".", "").isalpha())]  # ← ADD HERE
    tokens = [t for t in tokens if not is_garbage_token(t)]
    tokens = [normalize_word(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = merge_letter_sequences(tokens)
    tokens = add_ngrams(tokens, n=4)
    tokens = [t for t in tokens if not any(
        is_garbage_token(part) for part in t.split("_")
    )]
    return tokens


# -----------------------------
# PREDICTION
# -----------------------------
SPAM_BIAS = 0.0  # tune: negative = catch more spam, positive = fewer false positives

def predict_email(tokens):
    score_ham = log_prior_ham
    score_spam = log_prior_spam

    for token in tokens:
        if token in vocabulary:
            score_ham += ham_log_likelihoods.get(token, 0)
            score_spam += spam_log_likelihoods.get(token, 0)

    return "spam" if score_spam > score_ham + SPAM_BIAS else "ham"


# -----------------------------
# RUN BENCHMARK
# -----------------------------
correct = 0
tp = 0  # spam correctly detected
tn = 0  # ham correctly detected
fp = 0  # ham marked as spam
fn = 0  # spam missed

for email, true_label in zip(emails, true_labels):
    tokens = tokenize_email(email)
    prediction = predict_email(tokens)
    true_label_str = "spam" if true_label == "1" else "ham"

    if prediction == true_label_str:
        correct += 1

    if prediction == "spam" and true_label_str == "spam":
        tp += 1
    elif prediction == "ham" and true_label_str == "ham":
        tn += 1
    elif prediction == "spam" and true_label_str == "ham":
        fp += 1
    elif prediction == "ham" and true_label_str == "spam":
        fn += 1

accuracy = correct / len(emails) * 100
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Total benchmark emails: {len(emails)}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")

print("\n--- Confusion Matrix ---")
print(f"TP (Spam detected): {tp}")
print(f"TN (Ham detected): {tn}")
print(f"FP (Ham marked spam): {fp}")
print(f"FN (Spam missed): {fn}")

print("\n--- Metrics ---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

top = sorted(spam_log_likelihoods.items(), key=lambda x: x[1], reverse=True)[:50]
print("\n--- Top spam features ---")
for w, v in top:
    print(w)