import csv
import json

with open('output_3_5_1242.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]

with open('output_formatted.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Command', 'Response'])
    for item in data:
        writer.writerow([item['command'], item['response'].replace('\n', '\\n')])