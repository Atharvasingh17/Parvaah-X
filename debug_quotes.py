with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

tq = 0
for i, line in enumerate(lines, 1):
    count = line.count('"""')
    if count:
        tq += count
        print(f'Line {i}: +{count} tq={tq}  {line.rstrip()[:100]}')

print(f'\nTotal triple-quotes: {tq}  ({"OK - even" if tq % 2 == 0 else "ODD - MISMATCH!"})')
