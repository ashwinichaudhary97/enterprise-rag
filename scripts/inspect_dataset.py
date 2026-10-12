from datasets import load_dataset
from collections import Counter

print('=== QUESTIONS ===')
q = load_dataset('onyx-dot-app/EnterpriseRAG-Bench', 'questions', split='test')
print('Total:', len(q))
print('Fields:', q.column_names)
print()
print('Question categories:')
types = Counter(q['question_type'])
for t, c in types.most_common():
    print(f'  {t}: {c}')
print()
print('First question full record:')
first_q = q[0]
for k, v in first_q.items():
    s = str(v)
    print(f'  {k}: {s[:120]}')

print()
print('=== DOCUMENTS ===')
d = load_dataset('onyx-dot-app/EnterpriseRAG-Bench', 'documents', split='test')
print('Total:', len(d))
print('Fields:', d.column_names)
print()
print('Source type distribution:')
sources = Counter(d['source_type'])
for s, c in sources.most_common():
    print(f'  {s}: {c}')
print()
print('First document:')
first_d = d[0]
for k, v in first_d.items():
    s = str(v)
    print(f'  {k}: {s[:200]}')
