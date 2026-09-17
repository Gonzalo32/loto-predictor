import csv
import os

nuevos_sorteos = [
    {
        'sorteo': 3409, 'fecha': '2026-09-16',
        'Tradicional': ['13','14','16','19','20','29'],
        'Segunda': ['02','06','09','31','36','42'],
        'Revancha': ['02','09','20','30','35','36'],
        'SiempreSale': ['11','25','27','38','40','44']
    },
    {
        'sorteo': 3408, 'fecha': '2026-09-13',
        'Tradicional': ['05','08','22','29','34','37'],
        'Segunda': ['04','20','25','31','39','42'],
        'Revancha': ['16','19','21','35','38','42'],
        'SiempreSale': ['11','15','33','36','43','44']
    }
]

def actualizar_completo():
    path = 'c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv'
    with open(path, 'r', encoding='utf-8') as f:
        existing_lines = f.readlines()

    header = existing_lines[0]
    filtered_lines = []
    for line in existing_lines[1:]:
        parts = line.strip().split(',')
        if parts and parts[0].isdigit() and int(parts[0]) >= 3408:
            continue
        filtered_lines.append(line)

    new_rows = []
    for s in nuevos_sorteos:
        for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
            nums = s[mod]
            row_str = f"{s['sorteo']},{s['fecha']},{mod}," + ",".join(nums) + "\n"
            new_rows.append(row_str)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(new_rows)
        f.writelines(filtered_lines)
    print(f"[OK] historico_quini_completo.csv actualizado con sorteos 3408 y 3409.")

def actualizar_limpio():
    path = 'c:/Users/Administrador/Desktop/lot/historico_quini_limpio.csv'
    with open(path, 'r', encoding='utf-8') as f:
        existing_lines = f.readlines()

    header = "Sorteo,Fecha,B1,B2,B3,B4,B5,B6,Plus\n"
    filtered_lines = []
    for line in existing_lines[1:]:
        parts = line.strip().split(',')
        if parts and parts[0].isdigit() and int(parts[0]) >= 3408:
            continue
        filtered_lines.append(line)

    new_rows = []
    for s in nuevos_sorteos:
        nums = s['Tradicional']
        row_str = f"{s['sorteo']},{s['fecha']}," + ",".join(nums) + ",00\n"
        new_rows.append(row_str)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(new_rows)
        f.writelines(filtered_lines)
    print(f"[OK] historico_quini_limpio.csv actualizado con sorteos 3408 y 3409.")

if __name__ == '__main__':
    actualizar_completo()
    actualizar_limpio()
