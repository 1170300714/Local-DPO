import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, 'prompt.txt')
output_path = os.path.join(base_dir, 'prompt.json')

with open(input_path, 'r', encoding='utf-8') as f:
    lines = [line.rstrip('\n') for line in f if line.strip()]

data = [{'long': line} for line in lines]

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f'转换完成，共 {len(data)} 条记录，输出文件：{output_path}')
