import re
import os
import math
import json

ham_folder = "datasets/hams-data"
spam_folder = "datasets/spams-data"

ham_files = [os.path.join(ham_folder, f) for f in os.listdir(ham_folder) if f.endswith(".txt")]
spam_files = [os.path.join(spam_folder, f) for f in os.listdir(spam_folder) if f.endswith(".txt")]

STOPWORDS = {
    "the","is","a","an","to","of","and","for","in","on","with","that","this",
    "it","you","your","we","our","be","are","was","were","as","at","by","from",
    "or","but","if","then","than","so","because","about","into","over","after",
    "before","between","during","without","within","also","can","will","just",
    "do","does","did","done","have","has","had","having","i","me","my","mine",
    "he","him","his","she","her","they","them","their","theirs"
}


class EmailTrainer:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.emails = []
        self.tokens = []

    # ----------------------------
    # Load dataset
    # ----------------------------
    def load_emails(self):
        self.emails = []

        for file_path in self.file_paths:
            with open(file_path, "r", encoding="latin-1") as f:
                for line in f:
                    email_text = line.strip()
                    if email_text:
                        self.emails.append(email_text)

    # ----------------------------
    # Character normalization
    # ----------------------------
    def normalize_word(self, word):

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

    # ----------------------------
    # Merge sequences like "f r e e"
    # ----------------------------
    def merge_letter_sequences(self, tokens):

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

    # ----------------------------
    # Tokenization pipeline
    # ----------------------------
    def tokenize(self):

        self.tokens = []

        url_pattern = r"(http\S+|www\.\S+)"
        token_pattern = r"[a-zA-Z0-9$@\.]+"

        for email in self.emails:

            email = email.lower()

            email = re.sub(url_pattern, "<URL>", email)

            email = re.sub(r"(.)\1{2,}", r"\1\1", email)

            tokens = re.findall(token_pattern, email)

            tokens = [self.normalize_word(t) for t in tokens]
            tokens = [t for t in tokens if t not in STOPWORDS]

            tokens = self.merge_letter_sequences(tokens)

            self.tokens.append(tokens)

    # ----------------------------
    # Count words
    # ----------------------------
    def train_counts(self):

        word_counts = {}
        total_words = 0

        for email_tokens in self.tokens:
            for token in email_tokens:

                word_counts[token] = word_counts.get(token, 0) + 1
                total_words += 1

        return word_counts, total_words

    # ----------------------------
    # Remove rare tokens
    # ----------------------------
    def prune_counts(self, word_counts, min_count=2):

        return {
            word: count
            for word, count in word_counts.items()
            if count >= min_count
        }

    # ----------------------------
    # Compute log likelihoods
    # ----------------------------
    def compute_log_likelihoods(self, word_counts, total_words, vocabulary, alpha=1):

        log_likelihood = {}
        V = len(vocabulary)

        for word in vocabulary:

            count = word_counts.get(word, 0)

            prob = (count + alpha) / (total_words + alpha * V)

            log_likelihood[word] = math.log(prob)

        return log_likelihood


# ----------------------------
# Train HAM model
# ----------------------------

ham_trainer = EmailTrainer(ham_files)

ham_trainer.load_emails()
ham_trainer.tokenize()

ham_counts, _ = ham_trainer.train_counts()

ham_counts = ham_trainer.prune_counts(ham_counts, min_count=2)

ham_total = sum(ham_counts.values())


# ----------------------------
# Train SPAM model
# ----------------------------

spam_trainer = EmailTrainer(spam_files)

spam_trainer.load_emails()
spam_trainer.tokenize()

spam_counts, _ = spam_trainer.train_counts()

spam_counts = spam_trainer.prune_counts(spam_counts, min_count=2)

spam_total = sum(spam_counts.values())


# ----------------------------
# Vocabulary
# ----------------------------

vocabulary = set(ham_counts.keys()) | set(spam_counts.keys())


# ----------------------------
# Likelihoods
# ----------------------------

ham_log_likelihoods = ham_trainer.compute_log_likelihoods(
    ham_counts,
    ham_total,
    vocabulary
)

spam_log_likelihoods = spam_trainer.compute_log_likelihoods(
    spam_counts,
    spam_total,
    vocabulary
)


# ----------------------------
# Priors
# ----------------------------

total_emails = len(ham_trainer.emails) + len(spam_trainer.emails)

log_prior_ham = math.log(len(ham_trainer.emails) / total_emails)
log_prior_spam = math.log(len(spam_trainer.emails) / total_emails)


# ----------------------------
# Save model
# ----------------------------

model = {
    "vocabulary": list(vocabulary),
    "priors": {
        "ham": log_prior_ham,
        "spam": log_prior_spam
    },
    "log_likelihoods": {
        "ham": ham_log_likelihoods,
        "spam": spam_log_likelihoods
    }
}

with open("spam_model.json", "w", encoding="latin-1") as f:
    json.dump(model, f, ensure_ascii=False, indent=2)

print("Training complete. Model saved as spam_model.json")