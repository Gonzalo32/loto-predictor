with open('generador_cobertura_ml.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.strip() == 'if resonance_detected:':
        if i + 1 < len(lines) and 'print' in lines[i+1]:
            lines[i+1] = '        print("\\n==================================================")\n'

with open('generador_cobertura_ml.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Fixed indentation cleanly')

