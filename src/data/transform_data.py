import csv

INPUT_FILE = "input.csv"
OUTPUT_FILE = "data.csv"

def is_truthy(value):
    """Return True if a cell represents a 'truthy' blacklist value."""
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}

def main():
    with open(INPUT_FILE, mode="r", newline="", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)

        # Write new header
        writer.writerow(["ttl_bucket", "prompt", "tool_call_response"])

        for row in reader:
            # Skip rows that are blacklisted
            if is_truthy(row.get("blacklisted")):
                continue

            ttl_bucket = row.get("ttl_bucket", "")
            prompt = row.get("prompt", "")

            # Gather non-null r1–r5 values
            responses = []
            for key in ["r1", "r2", "r3", "r4", "r5"]:
                val = row.get(key)
                if val and val.strip():
                    responses.append(val.strip())

            writer.writerow([ttl_bucket, prompt, str(responses)])

if __name__ == "__main__":
    main()
