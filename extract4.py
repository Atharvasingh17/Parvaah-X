import json
import codecs

text = open(r'C:\Users\acer\.gemini\antigravity-ide\brain\af181db6-a9ac-4102-a074-5e1b026b9ac9\.system_generated\logs\transcript_full.jsonl', encoding='utf-8').read()
idx1 = text.find('sl_no,ministry,sector')
idx2 = text.find('1775,', idx1)

# Find the next newline after 1775
idx3 = text.find('\\n', idx2)
if idx3 == -1:
    idx3 = text.find('\n', idx2)
    
csv_raw = text[idx1:idx3]
# The text is inside a JSON string, so we can decode the escapes
csv_raw = csv_raw.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\"', '"')

with open(r'data\mospi_real_data.csv', 'w', encoding='utf-8') as f:
    f.write(csv_raw)

print(f"Successfully wrote {len(csv_raw)} characters.")
