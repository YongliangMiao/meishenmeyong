import glob
import json

# 1. 读取所有目标文件
file_list = glob.glob('data/output_allheads_temp_scores.rank*.json')
all_data = []

# 2. 合并所有文件的数据
for filename in file_list:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 有的文件内容可能是一个对象而不是列表，这里统一处理成列表
        if isinstance(data, dict):
            data = [data]
        all_data.extend(data)

# 3. 按照 reception_variance_score 从高到低排序
all_data_sorted = sorted(
    all_data, 
    key=lambda x: x.get("reception_variance_score", 0), 
    reverse=True
)

# 4. 截取前19652条
top_data = all_data_sorted[:19652]

# 5. 只保留 instruction, input, output 三个字段
filtered_data = [
    {
        "instruction": item.get("instruction", ""),
        "input": item.get("input", ""),
        "output": item.get("output", "")
    }
    for item in top_data
]

# 6. 写入输出文件
with open('data/output.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)
