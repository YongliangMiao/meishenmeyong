python -c "
import json, random, collections
from datasets import load_dataset

random.seed(42)
dataset_path = 'open-r1/OpenR1-Math-220k'
split = 'train'

print(f'>>> 正在加载数据集 {dataset_path} split={split} ...')

def _as_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, bool): return [x]
    try: return [bool(x)]
    except Exception: return []

def _bool_any(x):
    return any(bool(v) for v in _as_list(x))

def is_high_quality(example):
    return _bool_any(example.get('correctness_math_verify'))

try:
    ds = load_dataset(dataset_path, 'all', split=split)
except Exception:
    ds = load_dataset(dataset_path, split=split)

n_total = len(ds)
print(f'>>> 原始样本数：{n_total}')

print('>>> 正在过滤：[correctness_math_verify 有 True] ...')
filtered_ds = ds.filter(is_high_quality)
n_filtered = len(filtered_ds)
print(f'>>> 过滤后样本数：{n_filtered} （占比 {n_filtered / n_total:.2%}）')

# 分组，并取 q/a = messages[0/1]['content']
groups = collections.defaultdict(list)
drop_msgs = 0
for it in filtered_ds:
    msgs = it.get('messages') or []
    if not isinstance(msgs, list) or len(msgs) < 2:
        drop_msgs += 1
        continue
    m0, m1 = msgs[0], msgs[1]
    q = m0.get('content') if isinstance(m0, dict) else None
    a = m1.get('content') if isinstance(m1, dict) else None
    if not q or a is None:
        drop_msgs += 1
        continue
    pt = it.get('problem_type') or 'unknown'
    groups[pt].append({'q': q, 'a': a})

K = len(groups) or 1
avail_total = sum(len(v) for v in groups.values())
target_total = int(n_filtered * 0.10)      # 10% 向下取整，与你之前一致
if avail_total < target_total:
    print(f'>>> 警告：可用样本 {avail_total} < 目标 {target_total}，将以可用上限为准')
    target_total = avail_total

base = target_total // K                    # 先均分
alloc = {pt: min(base, len(items)) for pt, items in groups.items()}
remaining = target_total - sum(alloc.values())

# 回填：随机顺序轮转，把剩余名额填到仍有容量的类别
keys = list(groups.keys())
random.shuffle(keys)
while remaining > 0:
    progressed = False
    for pt in keys:
        cap = len(groups[pt]) - alloc[pt]
        if cap > 0 and remaining > 0:
            alloc[pt] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
    if not progressed:                      # 没有可填容量了
        break

print(f'>>> problem_type 种类：{K} | 目标(10%)：{target_total} | 均分基数：{base} | 回填后总配额：{sum(alloc.values())}')
print(f'>>> 因 messages 约束被丢弃：{drop_msgs}')
print('>>> 每类分配（pt | 可用 | 分配）:')
for pt in sorted(groups.keys()):
    print(f' - {pt}: {len(groups[pt])} | {alloc[pt]}')

INSTRUCT = 'Please reason step by step, and put your final answer within \\boxed{}.'  # 注意转义

selected = []
for pt, items in groups.items():
    k = alloc.get(pt, 0)
    if k <= 0: continue
    pick = items if k >= len(items) else random.sample(items, k)
    for it in pick:
        selected.append({'instruction': INSTRUCT, 'input': it['q'], 'output': it['a']})

out_path = 'diversity.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)

print(f'>>> 已保存 {out_path} | 实际写入：{len(selected)} 条（应等于目标或受可用上限限制）')
"