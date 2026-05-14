import pandas as pd
from datetime import datetime

# Load the raw CSV
try:
    # Skip the first 2 metadata lines if they exist
    df = pd.read_csv('historico_quini.csv', skiprows=2)
except:
    df = pd.read_csv('historico_quini.csv')

# Standardize columns
# CSV has: #, Fecha, N°1, N°2, N°3, N°4, N°5, N°6, Suma
df = df.rename(columns={
    '#': 'Sorteo',
    'Fecha': 'Fecha',
    'N°1': 'B1',
    'N°2': 'B2',
    'N°3': 'B3',
    'N°4': 'B4',
    'N°5': 'B5',
    'N°6': 'B6'
})

# Filter out rows where Sorteo is not a number
df = df[pd.to_numeric(df['Sorteo'], errors='coerce').notnull()]

# Add dummy Plus column (Quini doesn't have one like Loto)
df['Plus'] = '00'

# Reorder columns
df = df[['Sorteo', 'Fecha', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'Plus']]

# Data to add (New draws)
new_data = [
    ["3367", "22/04/2026", "00", "10", "12", "20", "30", "37", "00"],
    ["3366", "19/04/2026", "05", "06", "32", "33", "35", "38", "00"],
    ["3365", "15/04/2026", "04", "06", "07", "09", "24", "35", "00"],
    ["3364", "12/04/2026", "01", "14", "33", "38", "41", "44", "00"],
    ["3363", "08/04/2026", "01", "16", "31", "32", "34", "44", "00"],
    ["3362", "05/04/2026", "02", "04", "05", "13", "17", "40", "00"],
    ["3361", "01/04/2026", "20", "22", "27", "29", "34", "43", "00"],
    ["3360", "29/03/2026", "01", "09", "15", "18", "19", "38", "00"],
    ["3359", "25/03/2026", "10", "13", "18", "28", "34", "37", "00"],
    ["3358", "22/03/2026", "05", "11", "18", "19", "24", "36", "00"],
    ["3357", "18/03/2026", "03", "18", "22", "26", "34", "44", "00"],
    ["3356", "15/03/2026", "04", "08", "14", "15", "19", "31", "00"],
    ["3355", "11/03/2026", "09", "11", "12", "14", "18", "20", "00"],
    ["3354", "08/03/2026", "00", "06", "10", "13", "21", "37", "00"],
]

df_new = pd.DataFrame(new_data, columns=['Sorteo', 'Fecha', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'Plus'])

# Combine
df_final = pd.concat([df_new, df], ignore_index=True)

# Format dates to YYYY-MM-DD
def fix_date(d):
    try:
        # Check if already in YYYY-MM-DD
        datetime.strptime(str(d), '%Y-%m-%d')
        return d
    except:
        try:
            return datetime.strptime(str(d), '%d/%m/%Y').strftime('%Y-%m-%d')
        except:
            return d

df_final['Fecha'] = df_final['Fecha'].apply(fix_date)

# Save
df_final.to_csv('historico_quini_limpio.csv', index=False)
print("CSV de Quini 6 procesado y completado.")
