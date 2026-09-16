import json
import os

in_file = r"C:\Users\acer\.gemini\antigravity-ide\brain\af181db6-a9ac-4102-a074-5e1b026b9ac9\.system_generated\logs\transcript_full.jsonl"
out_file = r"data\mospi_real_data.csv"

with open(in_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data.get("type") == "USER_INPUT":
            content = data.get("content", "")
            if isinstance(content, str) and "sl_no,ministry,sector,project_name" in content:
                idx = content.find("sl_no,ministry,sector,project_name")
                csv_data = content[idx:]
                with open(out_file, "w", encoding="utf-8") as out:
                    out.write(csv_data)
                print("Extracted successfully!")
                break
