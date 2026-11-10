"""
Aplica la Mejora 1 (Cobertura Complementaria de Cuadruplas en B3)
y la Mejora 3 (Auto-Auditoria de Resultados) a generador_cobertura_ml.py
"""
import re

file_path = 'generador_cobertura_ml.py'

with open(file_path, 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Insertar funcion de auditoria antes de main()
audit_func = '''def auditar_prediccion_previa(draws):
    pred_file = 'proxima_prediccion.json'
    if not os.path.exists(pred_file):
        return
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            pred = json.load(f)
        last_sorteo = max(d['sorteo'] for d in draws)
        group_last = [d for d in draws if d['sorteo'] == last_sorteo]
        real_by_mod = {d['modalidad']: set(d['numbers']) for d in group_last}
        pozo_extra = real_by_mod.get('Tradicional', set()) | real_by_mod.get('Segunda', set()) | real_by_mod.get('Revancha', set())
        
        tickets = pred.get('tickets', [])
        if not tickets:
            return
            
        print("\\n==================================================")
        print(f"AUDITORÍA DE LA JUGADA ANTERIOR (SORTEO #{last_sorteo}):")
        print("==================================================")
        for idx, t in enumerate(tickets):
            nums = set(t.get('numeros', []))
            t_name = t.get('nombre', f'Boleto {idx+1}')
            print(f" {t_name}: {sorted(list(nums))}")
            for mod in ['Tradicional', 'Segunda', 'Revancha', 'SiempreSale']:
                if mod in real_by_mod:
                    hits = nums & real_by_mod[mod]
                    print(f"   -> {mod:<12}: {len(hits)} aciertos {sorted(list(hits)) if hits else ''}")
            pe_hits = nums & pozo_extra
            print(f"   -> Pozo Extra  : {len(pe_hits)} aciertos {sorted(list(pe_hits)) if pe_hits else ''}\\n")
        print("==================================================\\n")
    except Exception:
        pass

def estimar_probabilidad_exito_conjunto('''

code = code.replace('def estimar_probabilidad_exito_conjunto(', audit_func)

# 2. Llamar auditar_prediccion_previa en main() despues de cargar datos
old_load_section = '''    print("Cargando datos...")
    draws = load_data()
    n_draws = len(draws)
    if n_draws == 0:
        print("Error al cargar los datos.")
        return
        
    print(f"OK: {n_draws} sub-sorteos cargados.")
    print(f"Último sorteo registrado: Nro {draws[-1]['sorteo']} ({draws[-1]['fecha']}) [{draws[-1]['modalidad']}] -> {draws[-1]['numbers']}\\n")'''

new_load_section = '''    print("Cargando datos...")
    draws = load_data()
    n_draws = len(draws)
    if n_draws == 0:
        print("Error al cargar los datos.")
        return
        
    print(f"OK: {n_draws} sub-sorteos cargados.")
    print(f"Último sorteo registrado: Nro {draws[-1]['sorteo']} ({draws[-1]['fecha']}) [{draws[-1]['modalidad']}] -> {draws[-1]['numbers']}")
    
    # Auto-Auditoría de la jugada anterior
    auditar_prediccion_previa(draws)'''

code = code.replace(old_load_section, new_load_section)

# 3. Aplicar Cobertura Complementaria de Cuadruplas en B3
old_b3_section = '''    b1 = t_base[0]
    b2 = t_day[0] if t_day[0] != b1 else t_day[1]
    b3 = t_hib[0] if (t_hib[0] != b1 and t_hib[0] != b2) else t_base[1]
    tickets_optimos = [b1, b2, b3]'''

new_b3_section = '''    b1 = t_base[0]
    b2 = t_day[0] if t_day[0] != b1 else t_day[1]
    
    # Mejora: Selección de Boleto 3 con Cobertura Complementaria de Cuádruplas
    q12 = set(itertools.combinations(b1, 4)) | set(itertools.combinations(b2, 4))
    best_b3 = None
    best_b3_score = -float('inf')
    for c in viable_hib:
        if c == b1 or c == b2:
            continue
        q_c = set(itertools.combinations(c, 4))
        new_quads = len(q_c - q12)
        c_score = sum(borda_hib[n] for n in c) + 350.0 * sum(cooccur_exp_mat[u, v] for u, v in itertools.combinations(c, 2)) + 80.0 * new_quads
        if c_score > best_b3_score:
            best_b3_score = c_score
            best_b3 = c
            
    b3 = best_b3 if best_b3 is not None else (t_hib[0] if (t_hib[0] != b1 and t_hib[0] != b2) else t_base[1])
    tickets_optimos = [b1, b2, b3]'''

code = code.replace(old_b3_section, new_b3_section)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(code)

print("generador_cobertura_ml.py actualizado con auditoria y cuadruplas complementarias.")

