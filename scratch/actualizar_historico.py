import csv
import os
from typing import TypedDict, List

class SorteoDict(TypedDict):
    sorteo: int
    fecha: str
    Tradicional: List[str]
    Segunda: List[str]
    Revancha: List[str]
    SiempreSale: List[str]

nuevos_sorteos: List[SorteoDict] = [
    {
        'sorteo': 3411, 'fecha': '2026-09-23',
        'Tradicional': ['05','07','18','25','30','41'],
        'Segunda': ['00','06','08','14','40','44'],
        'Revancha': ['14','15','31','34','35','41'],
        'SiempreSale': ['13','24','27','32','42','44']
    },
    {
        'sorteo': 3410, 'fecha': '2026-09-20',
        'Tradicional': ['01','06','09','25','30','33'],
        'Segunda': ['02','06','07','12','18','44'],
        'Revancha': ['05','07','12','20','24','31'],
        'SiempreSale': ['14','15','29','31','32','40']
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
        if parts and parts[0].isdigit() and int(parts[0]) >= 3410:
            continue
        filtered_lines.append(line)

    new_rows = []
    for s in nuevos_sorteos:
        for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
            nums = s.get(mod)
            if not isinstance(nums, list) or len(nums) != 6:
                continue
            row_str = f"{s['sorteo']},{s['fecha']},{mod}," + ",".join(nums) + "\n"
            new_rows.append(row_str)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(new_rows)
        f.writelines(filtered_lines)
    print(f"[OK] historico_quini_completo.csv actualizado con sorteos 3410 y 3411.")

def actualizar_limpio():
    path = 'c:/Users/Administrador/Desktop/lot/historico_quini_limpio.csv'
    with open(path, 'r', encoding='utf-8') as f:
        existing_lines = f.readlines()

    header = "Sorteo,Fecha,B1,B2,B3,B4,B5,B6,Plus\n"
    filtered_lines = []
    for line in existing_lines[1:]:
        parts = line.strip().split(',')
        if parts and parts[0].isdigit() and int(parts[0]) >= 3410:
            continue
        filtered_lines.append(line)

    new_rows = []
    for s in nuevos_sorteos:
        nums = s['Tradicional']
        if not isinstance(nums, list) or len(nums) != 6:
            continue
        row_str = f"{s['sorteo']},{s['fecha']}," + ",".join(nums) + ",00\n"
        new_rows.append(row_str)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(new_rows)
        f.writelines(filtered_lines)
    print(f"[OK] historico_quini_limpio.csv actualizado con sorteos 3410 y 3411.")

if __name__ == '__main__':
    actualizar_completo()
    actualizar_limpio()
