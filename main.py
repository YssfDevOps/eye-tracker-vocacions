import csv, pathlib, textwrap

path = pathlib.Path("src/data/positions.csv")

# Count *physical* lines
physical = sum(1 for _ in path.open('rb'))

# Count *logical* rows via the csv module (same rules Excel uses)
with path.open(newline='', encoding='utf‑8') as f:
    logical = sum(1 for _ in csv.reader(f))

print(f"physical={physical}, logical={logical}, gap={physical-logical}")
