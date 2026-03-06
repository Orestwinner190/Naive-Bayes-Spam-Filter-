import os
import re
import json

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

        for line in f:

            line = line.strip()
            if not line:
                continue

            if " " in line:
                email_text, label = line.rsplit(" ", 1)

                emails.append(email_text.strip())
                true_labels.append(label.strip())

            else:
                emails.append(line)
                true_labels.append("0")

# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize_word(word):

    substitutions = {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "$": "s",
        "@": "a",
        "5": "s",
        "7": "t"
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


# -----------------------------
# TOKENIZATION PIPELINE
# -----------------------------
def tokenize_email(email):

    url_pattern = r"(http\S+|www\.\S+)"
    token_pattern = r"[a-zA-Z0-9$@\.]+"

    email = email.lower()

    email = re.sub(url_pattern, "<URL>", email)

    email = re.sub(r"(.)\1{2,}", r"\1\1", email)

    tokens = re.findall(token_pattern, email)

    tokens = [normalize_word(t) for t in tokens]

    tokens = merge_letter_sequences(tokens)

    return tokens


# -----------------------------
# PREDICTION
# -----------------------------
def predict_email(tokens):

    score_ham = log_prior_ham
    score_spam = log_prior_spam

    for token in tokens:

        if token in vocabulary:

            score_ham += ham_log_likelihoods.get(token, 0)
            score_spam += spam_log_likelihoods.get(token, 0)

    return "spam" if score_spam > score_ham else "ham"


# -----------------------------
# RUN BENCHMARK
# -----------------------------
correct = 0

for email, true_label in zip(emails, true_labels):

    tokens = tokenize_email(email)

    prediction = predict_email(tokens)

    true_label_str = "spam" if true_label == "1" else "ham"

    if prediction == true_label_str:
        correct += 1


accuracy = correct / len(emails) * 100

print(f"Total benchmark emails: {len(emails)}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")


# -----------------------------
# MODEL STATS
# -----------------------------
print("\n--- Model Stats ---")
print(f"Vocabulary size: {len(vocabulary)}")
print(f"Ham log-likelihood entries: {len(ham_log_likelihoods)}")
print(f"Spam log-likelihood entries: {len(spam_log_likelihoods)}")