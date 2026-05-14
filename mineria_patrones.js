const fs = require('fs');

class PatternMiner {
    constructor(filename = 'historico_loto.csv') {
        this.datos = this._loadData(filename);
    }

    _loadData(filename) {
        if (!fs.existsSync(filename)) return [];
        const content = fs.readFileSync(filename, 'utf-8').trim().split('\n');
        const headers = content[0].split(',');
        return content.slice(1).map(line => {
            const values = line.split(',');
            const row = {};
            headers.forEach((h, i) => row[h.trim()] = values[i].trim());
            return row;
        });
    }

    _extraerBolillas(row) {
        return [
            parseInt(row['B1']), parseInt(row['B2']), parseInt(row['B3']),
            parseInt(row['B4']), parseInt(row['B5']), parseInt(row['B6'])
        ].sort((a, b) => a - b);
    }

    analizarRepeticiones() {
        console.log("\n--- ANALIZANDO REPETICIÓN DEL SORTEO ANTERIOR ---");
        const conteoRepeticiones = { 0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0 };
        
        for (let i = 1; i < this.datos.length; i++) {
            const actual = this._extraerBolillas(this.datos[i]);
            const anterior = this._extraerBolillas(this.datos[i-1]);
            let repetidos = 0;
            actual.forEach(n => { if (anterior.includes(n)) repetidos++; });
            conteoRepeticiones[repetidos]++;
        }

        console.log("¿Cuántos números se repiten del sorteo inmediatamente anterior?");
        Object.entries(conteoRepeticiones).forEach(([num, cant]) => {
            const porc = (cant / (this.datos.length - 1) * 100).toFixed(2);
            console.log(`   ${num} números repetidos: ${cant} veces (${porc}%)`);
        });
    }

    analizarTripletasFrecuentes() {
        console.log("\n--- BUSCANDO TRIPLETAS FRECUENTES (COINCIDENCIAS DE 3 NÚMEROS) ---");
        const tripletas = {};

        this.datos.forEach(row => {
            const b = this._extraerBolillas(row);
            for (let i = 0; i < b.length; i++) {
                for (let j = i + 1; j < b.length; j++) {
                    for (let k = j + 1; k < b.length; k++) {
                        const key = `${b[i]}-${b[j]}-${b[k]}`;
                        tripletas[key] = (tripletas[key] || 0) + 1;
                    }
                }
            }
        });

        const topTripletas = Object.entries(tripletas)
            .filter(x => x[1] > 1)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        if (topTripletas.length === 0) {
            console.log("No se encontraron tripletas que se repitan más de una vez.");
        } else {
            console.log("Top 10 tripletas más ganadoras de la historia:");
            topTripletas.forEach(([t, cant]) => {
                console.log(`   [${t.split('-').join(', ')}]: ha salido ${cant} veces`);
            });
        }
    }

    analizarCorrelacionPlus() {
        console.log("\n--- CORRELACIÓN NÚMERO PLUS vs BOLILLAS ---");
        const plusMap = {}; // { plusValue: { ballValue: frequency } }

        this.datos.forEach(row => {
            const p = parseInt(row['Plus']);
            const b = this._extraerBolillas(row);
            if (!plusMap[p]) plusMap[p] = {};
            b.forEach(n => {
                plusMap[p][n] = (plusMap[p][n] || 0) + 1;
            });
        });

        console.log("Si el PLUS es un número específico, ¿qué bolillas suelen acompañarlo?");
        Object.keys(plusMap).sort((a,b)=>a-b).forEach(p => {
            const topForPlus = Object.entries(plusMap[p])
                .sort((a,b) => b[1] - a[1])
                .slice(0, 3)
                .map(x => x[0]);
            console.log(`   PLUS ${p} -> Suele venir con: ${topForPlus.join(', ')}`);
        });
    }

    ejecutarTodo() {
        console.log("==================================================");
        console.log("🕵️ MINERÍA DE PATRONES PROFUNDA - LOTO PLUS");
        console.log("==================================================");
        console.log(`Analizando ${this.datos.length} sorteos históricos...\n`);
        
        this.analizarRepeticiones();
        this.analizarTripletasFrecuentes();
        this.analizarCorrelacionPlus();
        
        console.log("\n==================================================");
    }
}

const miner = new PatternMiner();
miner.ejecutarTodo();
