import csv
import os

nuevos_sorteos = [
    {
        'sorteo': 3405, 'fecha': '2026-09-02',
        'Tradicional': ['00','05','10','22','26','45'],
        'Segunda': ['02','03','16','22','24','44'],
        'Revancha': ['02','07','14','25','34','38'],
        'SiempreSale': ['02','05','08','10','31','38']
    },
    {
        'sorteo': 3404, 'fecha': '2026-08-30',
        'Tradicional': ['07','14','22','23','40','45'],
        'Segunda': ['05','11','17','36','38','40'],
        'Revancha': ['00','05','10','20','26','28'],
        'SiempreSale': ['06','13','14','21','36','37']
    },
    {
        'sorteo': 3403, 'fecha': '2026-08-26',
        'Tradicional': ['05','09','11','16','23','26'],
        'Segunda': ['10','14','25','30','37','39'],
        'Revancha': ['08','10','22','30','36','37'],
        'SiempreSale': ['20','23','24','28','35','39']
    },
    {
        'sorteo': 3402, 'fecha': '2026-08-23',
        'Tradicional': ['00','21','24','26','27','42'],
        'Segunda': ['19','24','25','27','29','36'],
        'Revancha': ['02','09','21','29','36','39'],
        'SiempreSale': ['01','04','05','20','37','43']
    },
    {
        'sorteo': 3401, 'fecha': '2026-08-19',
        'Tradicional': ['01','02','06','10','17','25'],
        'Segunda': ['03','10','12','16','26','33'],
        'Revancha': ['02','12','14','24','29','40'],
        'SiempreSale': ['07','21','27','30','32','37']
    },
    {
        'sorteo': 3400, 'fecha': '2026-08-16',
        'Tradicional': ['02','09','15','18','28','31'],
        'Segunda': ['02','18','28','34','43','44'],
        'Revancha': ['12','16','29','30','37','43'],
        'SiempreSale': ['02','06','09','10','32','35']
    },
    {
        'sorteo': 3399, 'fecha': '2026-08-12',
        'Tradicional': ['23','24','25','30','31','44'],
        'Segunda': ['04','08','16','20','24','36'],
        'Revancha': ['03','07','22','24','25','42'],
        'SiempreSale': ['13','21','23','26','32','33']
    },
    {
        'sorteo': 3398, 'fecha': '2026-08-09',
        'Tradicional': ['06','09','10','18','22','31'],
        'Segunda': ['11','13','23','25','26','30'],
        'Revancha': ['01','03','18','23','24','31'],
        'SiempreSale': ['01','09','16','18','30','40']
    },
    {
        'sorteo': 3397, 'fecha': '2026-08-05',
        'Tradicional': ['03','04','14','17','19','35'],
        'Segunda': ['05','22','28','31','32','41'],
        'Revancha': ['08','11','14','29','33','44'],
        'SiempreSale': ['05','10','20','24','25','38']
    },
    {
        'sorteo': 3396, 'fecha': '2026-08-02',
        'Tradicional': ['00','12','23','29','43','45'],
        'Segunda': ['01','13','18','29','31','44'],
        'Revancha': ['02','03','19','20','23','25'],
        'SiempreSale': ['00','01','09','22','27','29']
    },
    {
        'sorteo': 3395, 'fecha': '2026-07-29',
        'Tradicional': ['00','02','27','34','37','38'],
        'Segunda': ['00','01','03','06','10','35'],
        'Revancha': ['06','24','28','32','39','42'],
        'SiempreSale': ['15','17','30','35','36','39']
    },
    {
        'sorteo': 3394, 'fecha': '2026-07-26',
        'Tradicional': ['08','09','12','21','28','37'],
        'Segunda': ['00','01','03','04','05','22'],
        'Revancha': ['03','05','08','22','25','45'],
        'SiempreSale': ['06','10','13','40','42','43']
    },
    {
        'sorteo': 3393, 'fecha': '2026-07-22',
        'Tradicional': ['00','05','10','36','41','42'],
        'Segunda': ['06','24','31','34','42','43'],
        'Revancha': ['01','08','18','30','38','43'],
        'SiempreSale': ['02','07','20','23','44','45']
    },
    {
        'sorteo': 3392, 'fecha': '2026-07-19',
        'Tradicional': ['05','17','27','28','31','45'],
        'Segunda': ['08','21','23','33','36','41'],
        'Revancha': ['04','10','14','17','25','32'],
        'SiempreSale': ['04','14','18','19','28','29']
    },
    {
        'sorteo': 3391, 'fecha': '2026-07-15',
        'Tradicional': ['19','22','25','30','37','42'],
        'Segunda': ['01','05','16','21','33','34'],
        'Revancha': ['04','18','22','31','34','39'],
        'SiempreSale': ['03','21','31','34','37','42']
    },
    {
        'sorteo': 3390, 'fecha': '2026-07-12',
        'Tradicional': ['01','03','16','24','25','31'],
        'Segunda': ['02','03','20','31','36','43'],
        'Revancha': ['11','18','23','31','38','41'],
        'SiempreSale': ['02','10','22','23','32','40']
    }
]

def actualizar_completo():
    path = 'c:/Users/Administrador/Desktop/lot/historico_quini_completo.csv'
    with open(path, 'r', encoding='utf-8') as f:
        existing_lines = f.readlines()

    header = existing_lines[0]
    # Filter out any lines for sorteos >= 3390 if present to avoid duplication
    filtered_lines = []
    for line in existing_lines[1:]:
        parts = line.strip().split(',')
        if parts and parts[0].isdigit() and int(parts[0]) >= 3390:
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
    print(f"[OK] historico_quini_completo.csv actualizado con {len(nuevos_sorteos)} sorteos nuevos.")

def actualizar_limpio():
    path = 'c:/Users/Administrador/Desktop/lot/historico_quini_limpio.csv'
    with open(path, 'r', encoding='utf-8') as f:
        existing_lines = f.readlines()

    header = "Sorteo,Fecha,B1,B2,B3,B4,B5,B6,Plus\n"
    filtered_lines = []
    for line in existing_lines[1:]:
        parts = line.strip().split(',')
        if parts and parts[0].isdigit() and int(parts[0]) >= 3390:
            continue
        # Also clean up corrupt headers in B1 if present
        if len(parts) >= 9 and parts[2] == 'Tradicional':
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
    print(f"[OK] historico_quini_limpio.csv actualizado con {len(nuevos_sorteos)} sorteos nuevos.")

if __name__ == '__main__':
    actualizar_completo()
    actualizar_limpio()
