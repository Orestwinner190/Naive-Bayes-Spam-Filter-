import os

# --------------------------
# INPUT / OUTPUT FILES
# --------------------------
mixed_file = "datasets/benchmark-testing/benchmark_emails.txt"  # path to your mixed dataset
ham_file = "datasets/hams-data/ham5.txt"
spam_file = "datasets/spams-data/spam5.txt"

# Ensure output directories exist
os.makedirs(os.path.dirname(ham_file), exist_ok=True)
os.makedirs(os.path.dirname(spam_file), exist_ok=True)

# --------------------------
# PROCESSING
# --------------------------
with open(mixed_file, "r", encoding="latin-1") as f, \
     open(ham_file, "w", encoding="latin-1") as ham_out, \
     open(spam_file, "w", encoding="latin-1") as spam_out:

    for line in f:
        line = line.strip()
        if not line:
            continue

        # Assume label is the last character after a space
        if line[-1] in ("0", "1") and line[-2].isspace():
            email_text = line[:-2].strip()  # everything before the space+label
            label = line[-1]

            if label == "0":
                ham_out.write(email_text + "\n")
            elif label == "1":
                spam_out.write(email_text + "\n")
        else:
            # fallback for malformed lines
            continue

print("Done! Emails sorted into ham5.txt and spam5.txt")