import json

with open("spam_model.json", "r", encoding="latin-1") as f:
    model = json.load(f)

ham_ll = model["log_likelihoods"]["ham"]
spam_ll = model["log_likelihoods"]["spam"]
vocabulary = set(model["vocabulary"])

# Compute log likelihood RATIO (spam score - ham score) for each word
# Positive = more spammy, Negative = more hammy
ratios = {}
for word in vocabulary:
    s = spam_ll.get(word, None)
    h = ham_ll.get(word, None)
    if s is not None and h is not None:
        ratios[word] = s - h

print("=== TOP 50 MOST SPAMMY WORDS (by ratio) ===")
top_spam = sorted(ratios.items(), key=lambda x: x[1], reverse=True)[:50]
for w, r in top_spam:
    print(f"  {w:40s}  ratio={r:+.3f}")

print("\n=== TOP 50 MOST HAMMY WORDS (by ratio) ===")
top_ham = sorted(ratios.items(), key=lambda x: x[1])[:50]
for w, r in top_ham:
    print(f"  {w:40s}  ratio={r:+.3f}")

print("\n=== SUSPICIOUS: High spam likelihood but LOW ratio (noise words) ===")
top_raw = sorted(spam_ll.items(), key=lambda x: x[1], reverse=True)[:100]
for w, s in top_raw:
    ratio = ratios.get(w, 0)
    if abs(ratio) < 0.1:
        print(f"  {w:40s}  spam_ll={s:.3f}  ratio={ratio:+.4f}")