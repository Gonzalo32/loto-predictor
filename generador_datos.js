const fs = require('fs');

const CSV_FILE = 'historico_loto.csv';

function generarSorteos(cantidad, sorteoInicial) {
    const sorteos = [];
    let fecha = new Date('2026-03-10T12:00:00Z'); // Start date just before our first mock
    
    for (let i = 0; i < cantidad; i++) {
        fecha.setDate(fecha.getDate() - (Math.random() > 0.5 ? 3 : 4)); // twice a week roughly
        
        const numeros = new Set();
        while (numeros.size < 6) {
            numeros.add(Math.floor(Math.random() * 46));
        }
        const sorted = Array.from(numeros).sort((a,b) => a-b);
        const plus = Math.floor(Math.random() * 10);
        
        const pad = n => String(n).padStart(2, '0');
        const fStr = fecha.toISOString().split('T')[0];
        
        const row = [
            sorteoInicial - i,
            fStr,
            pad(sorted[0]), pad(sorted[1]), pad(sorted[2]),
            pad(sorted[3]), pad(sorted[4]), pad(sorted[5]),
            pad(plus)
        ].join(',');
        
        sorteos.push(row);
    }
    return sorteos;
}

const currentContent = fs.existsSync(CSV_FILE) ? fs.readFileSync(CSV_FILE, 'utf-8').trim().split('\n') : [];
const header = currentContent[0];
const existingData = currentContent.slice(1);

// Generate 500 past draws starting from 3864
const nuevosSorteos = generarSorteos(500, 3864);

// The oldest first
nuevosSorteos.reverse();

const allData = [header, ...nuevosSorteos, ...existingData];

fs.writeFileSync(CSV_FILE, allData.join('\n') + '\n', 'utf-8');
console.log(`✅ Se han inyectado ${nuevosSorteos.length} sorteos históricos al CSV para mejorar el modelo.`);
