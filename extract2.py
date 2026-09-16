import re
import json

text = open(r'C:\Users\acer\.gemini\antigravity-ide\brain\af181db6-a9ac-4102-a074-5e1b026b9ac9\.system_generated\logs\transcript_full.jsonl', encoding='utf-8').read()
# Find the start of the CSV
match = re.search(r'(sl_no,ministry,sector.*?1775,.*?152)', text, flags=re.DOTALL)
if match:
    # unescape json newlines
    raw_csv = match.group(1)
    raw_csv = raw_csv.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\"', '"')
    open(r'data\mospi_real_data.csv', 'w', encoding='utf-8').write(raw_csv)
    print("Found and wrote!")
else:
    print("Not found")
