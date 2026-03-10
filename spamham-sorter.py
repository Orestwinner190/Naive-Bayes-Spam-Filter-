import csv
import os

input_file = r"C:\Users\ortso\Desktop\enron_spam_data.csv"
spam_output = r"C:\Users\ortso\Desktop\Git projects\Spam filter\datasets\spams-data\spam7-trial.txt"
ham_output = r"C:\Users\ortso\Desktop\Git projects\Spam filter\datasets\hams-data\ham7-trial.txt"

csv.field_size_limit(10_000_000)

spam_count = 0
ham_count = 0
skipped = 0

with open(input_file, "r", encoding="utf-8", errors="ignore") as f, \
     open(spam_output, "w", encoding="latin-1", errors="ignore") as spam_out, \
     open(ham_output, "w", encoding="latin-1", errors="ignore") as ham_out:

    reader = csv.reader(f)
    next(reader)  # skip header

    for row in reader:
        if len(row) < 4:
            skipped += 1
            continue

        subject = row[1].strip()
        message = row[2].strip()
        label = row[3].strip().lower()

        # combine subject and message
        body = " ".join((subject + " " + message).split())

        if len(body) < 50:
            skipped += 1
            continue

        if label == "spam":
            spam_out.write(body + "\n")
            spam_count += 1
        elif label == "ham":
            ham_out.write(body + "\n")
            ham_count += 1

print(f"Done — {spam_count} spam, {ham_count} ham written, {skipped} skipped")