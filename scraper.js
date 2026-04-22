const fs = require('fs');

class LotoScraper {
    constructor(filename = 'historico_loto.csv') {
        this.filename = filename;
    }

    collectData() {
        console.log("🔎 [Scraper] Verificando fuente de datos y sorteos...");
        const datos = [];
        if (fs.existsSync(this.filename)) {
            const fileContent = fs.readFileSync(this.filename, 'utf-8');
            const lines = fileContent.trim().split('\n');
            const headers = lines[0].split(',');
            
            for (let i = 1; i < lines.length; i++) {
                const values = lines[i].split(',');
                if (values.length === headers.length) {
                    const row = {};
                    for (let j = 0; j < headers.length; j++) {
                        row[headers[j].trim()] = values[j].trim();
                    }
                    datos.push(row);
                }
            }
            console.log(`✅ [Scraper] ${datos.length} sorteos cargados desde el historial local.`);
        } else {
            console.log("⚠️ [Scraper] No se encontró historial. Se recomienda recopilar datos inicialmente.");
        }
        return datos;
    }

    mockFetchNewDraw(lastDrawNum) {
        const numbers = new Set();
        while(numbers.size < 6) {
            numbers.add(Math.floor(Math.random() * 46));
        }
        const sortedNumbers = Array.from(numbers).sort((a,b) => a - b);
        const plus = Math.floor(Math.random() * 10);
        
        const pad = (num) => String(num).padStart(2, '0');
        
        return {
            'Sorteo': String(lastDrawNum + 1),
            'Fecha': new Date(Date.now() + 3*24*60*60*1000).toISOString().split('T')[0],
            'B1': pad(sortedNumbers[0]),
            'B2': pad(sortedNumbers[1]),
            'B3': pad(sortedNumbers[2]),
            'B4': pad(sortedNumbers[3]),
            'B5': pad(sortedNumbers[4]),
            'B6': pad(sortedNumbers[5]),
            'Plus': pad(plus)
        };
    }
}

module.exports = LotoScraper;
