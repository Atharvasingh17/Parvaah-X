import re

with open(r'C:\Users\acer\.gemini\antigravity-ide\brain\af181db6-a9ac-4102-a074-5e1b026b9ac9\.system_generated\logs\transcript_full.jsonl', encoding='utf-8') as f:
    text = f.read()

# Look for the start of the CSV and the end
match = re.search(r'(sl_no,ministry,sector.*?\n1775,.*?\n)', text, flags=re.DOTALL)
if match:
    # Fix the newlines inside the JSON encoded string if any
    raw_csv = match.group(1)
    raw_csv = raw_csv.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\"', '"')
    with open(r'data\mospi_real_data.csv', 'w', encoding='utf-8') as out:
        out.write(raw_csv)
    print("Found and wrote CSV!")
else:
    print("CSV data not found by regex.")
