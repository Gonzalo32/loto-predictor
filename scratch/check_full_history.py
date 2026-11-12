import os, sys
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_DIR)
from generador_cobertura_ml import load_data

draws = load_data()
trad = [d for d in draws if d['modalidad'] == 'Tradicional']
sorteos = sorted(list(set(d['sorteo'] for d in draws)))

print(f"Total sub-sorteos en base de datos: {len(draws)}")
print(f"Total fechas (sorteos unicos): {len(sorteos)}")
print(f"Total sorteos con modalidad Tradicional: {len(trad)}")
print(f"Primer sorteo: #{sorteos[0]} ({draws[0]['fecha']})")
print(f"Ultimo sorteo: #{sorteos[-1]} ({draws[-1]['fecha']})")

