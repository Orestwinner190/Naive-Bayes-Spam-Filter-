import os
import re
import json

with open("spam_model.json", "r", encoding="latin-1") as f:
    model = json.load(f)

vocabulary = set(model["vocabulary"])
log_prior_ham = model["priors"]["ham"]
log_prior_spam = model["priors"]["spam"]
ham_log_likelihoods = model["log_likelihoods"]["ham"]
spam_log_likelihoods = model["log_likelihoods"]["spam"]


testing_folder = "filter-testing"
testing_files = [os.path.join(testing_folder, f) for f in os.listdir(testing_folder) if f.endswith(".txt")]

emails = []
for file_path in testing_files:
    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            email_text = line.strip()
            if email_text:
                emails.append(email_text)

def tokenise_email(email):
    url_pattern = r"(http\S+|www\.\S+)"
    word_pattern = r"\b\w+\b"

    email = email.lower()
    email = re.sub(url_pattern, "<URL>", email)
    email = re.sub(r"(.)\1{2,}", r"\1\1", email)
    tokens = re.findall(word_pattern, email, flags=re.UNICODE)
    return tokens


def predict_email(tokens):
    score_ham = log_prior_ham
    score_spam = log_prior_spam

    for token in tokens:
        if token in vocabulary:
            score_ham += ham_log_likelihoods.get(token, 0)
            score_spam += spam_log_likelihoods.get(token, 0)
        # tokens not in vocabulary are ignored for now

    return "spam" if score_spam > score_ham else "ham"

for i, email in enumerate(emails, start=1):
    tokens = tokenise_email(email)
    prediction = predict_email(tokens)
    print(f"Email {i}: {prediction}")
