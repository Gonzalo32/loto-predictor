with open('generador_cobertura_ml.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if line.strip() in ('print(f"', 'print("'):
        next_line = lines[i+1] if i + 1 < len(lines) else ''
        if '===' in next_line:
            new_lines.append('    print("\\n==================================================")\n')
            i += 2
            continue
    new_lines.append(line)
    i += 1

with open('generador_cobertura_ml.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Fixed lines cleanly')

