# Naive Bayes Spam Filter

A spam classifier built entirely from scratch in Python — no scikit-learn, no shortcuts. Trained on 22,000+ emails from the Enron corpus and SpamAssassin dataset.

**Live demo:** [spam-filters.up.railway.app](https://spam-filters.up.railway.app)

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 89.4% |
| Precision | 0.6607 |
| Recall | 0.8060 |
| F1 Score | 0.7261 |

Benchmarked on 3,000 held-out emails (500 spam / 2500 ham).

---

## How It Works

### The Math

Naive Bayes classifies an email by computing the probability it belongs to each class (spam or ham) and picking the most likely one. For each class C:

```
score(C) = log P(C) + Σ cₙ · log [ (N(w,C) + α) / (N(C) + α|V|) ]
```

Where:
- `P(C)` is the prior probability of the class (how much of the training data is spam vs ham)
- `N(w,C)` is how many times word `w` appears in class `C`
- `N(C)` is the total word count in class `C`
- `α` is Laplace smoothing (prevents zero probabilities for unseen words)
- `|V|` is the vocabulary size

We use log probabilities to avoid floating point underflow when multiplying many small numbers together.

### Why "Naive"?

The algorithm assumes all words are independent of each other — that the presence of "free" doesn't affect the probability of "prize" appearing. This is obviously false in reality, but the simplification works surprisingly well in practice and makes the math tractable.

---

## Tokenization Pipeline

Raw email text goes through several preprocessing steps before reaching the classifier:

```
raw email
  → lowercase
  → HTML stripping
  → URL replacement (<URL>)
  → repeated character collapsing (freeee → free)
  → tokenization
  → dot/dash stripping
  → garbage token filtering (high digit ratio, no vowels)
  → leet speak normalization (0→o, 1→i, $→s, @→a)
  → stopword removal
  → short token filtering (len ≤ 3)
  → letter sequence merging
  → n-gram generation (bigrams through 4-grams)
  → post n-gram garbage filtering
```

N-grams are particularly useful here — `free_money` is a much stronger spam signal than `free` and `money` individually.

### Garbage Token Filtering

Tokens are discarded if:
- More than 30% of characters are digits
- Length ≤ 2 and contains any digit
- Has 3+ alphabetic characters but no vowels (likely an encoding artifact)

---

## Feature Selection

After tokenization, chi-square feature selection picks the 150,000 tokens most correlated with spam vs ham. This removes noise words that appear equally in both classes and reduces model size.

A `min_ratio=0.3` threshold ensures only tokens with meaningful class imbalance are kept.

---

## Dataset

- **Ham:** 18,153 emails from the Enron corpus (real business emails from Enron employees)
- **Spam:** 4,088 emails from SpamAssassin's `20021010_spam` archive
- **Benchmark:** 3,000 held-out emails not used in training

Several dataset expansion attempts were made (additional SpamAssassin archives, Kaggle datasets, labeled Enron spam CSVs) but all introduced label contamination or domain mismatch that hurt accuracy. The current dataset combination is the best found.

---

## Project Structure

```
├── training.py          # Model training — tokenization, counting, chi-square, serialization
├── benchmark.py         # Evaluation on held-out test set
├── diagnose.py          # Inspect top spam/ham features and suspicious tokens
├── spamham-sorter.py    # Converts CSV datasets to .txt email files
├── backend/
│   ├── app.py           # Flask REST API
│   ├── predictor.py     # Tokenization + prediction (shared module)
│   ├── requirements.txt
│   └── Procfile
└── datasets/
    ├── ham/
    ├── spam/
    └── benchmark-testing/
```

---

## Running Locally

**Train the model:**
```bash
python training.py
```
This reads emails from `datasets/ham/` and `datasets/spam/`, trains the classifier, and saves `spam_model.json`.

**Run the benchmark:**
```bash
python benchmark.py
```

**Start the API:**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

The API runs on `http://localhost:5000`.

**Predict an email:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"email": "Congratulations! You have won a free prize..."}'
```

Response:
```json
{
  "label": "spam",
  "confidence": 94.3
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Classify an email. Body: `{"email": "..."}` |
| GET | `/health` | Health check |

---

## Limitations

- **Enron bias:** The ham corpus is corporate business email, so the model is tuned to that vocabulary. Generic phishing phrases that don't overlap with typical spam training vocabulary may be misclassified.
- **Naive independence assumption:** Words are treated as independent, which isn't true. Logistic regression or SVM would handle feature correlation better.
- **~90% ceiling:** This appears to be near the Naive Bayes ceiling for this dataset combination. Breaking through would require either a different algorithm or a broader ham corpus.

---

## Next

Building a logistic regression classifier from scratch to compare against this baseline. Gradient descent, sigmoid activation, binary cross-entropy loss — no libraries.