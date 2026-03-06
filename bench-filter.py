import os

# --------------------------
# INPUT / OUTPUT FILES
# --------------------------
mixed_file = "datasets/benchmark-testing/benchmarkemail.csv"  # path to your new benchmark dataset
bench_file = "datasets/benchmark-testing/newbenchmark_emails.txt"

# Ensure output folder exists
os.makedirs(os.path.dirname(bench_file), exist_ok=True)

# --------------------------
# PROCESSING
# --------------------------
with open(mixed_file, "r", encoding="utf-8", errors="replace") as f, \
     open(bench_file, "w", encoding="utf-8") as out_file:

    for line in f:
        line = line.strip()
        if not line:
            continue

        # Split on the last comma to separate email text and label
        if "," in line:
            email_text, label = line.rsplit(",", 1)
            email_text = email_text.strip()
            label = label.strip()
            out_file.write(f"{email_text}\t{label}\n")
        else:
            # fallback: skip malformed lines
            continue

print(f"Done! Benchmark emails written to '{bench_file}'")