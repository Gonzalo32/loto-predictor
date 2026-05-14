const fs = require('fs');

class QuiniMiner {
    constructor(filename = 'historico_quini_limpio.csv') {
        this.datos = this._loadData(filename).reverse(); // Reverse to have it chronological
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

    analizarRepeticionesYRangos() {
        console.log("\n--- ANALIZANDO REPETICIONES Y RANGOS (QUINI 6) ---");
        const conteoRepeticiones = { 0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0 };
        const rangosRepetidos = { 'Inferior (0-22)': 0, 'Superior (23-45)': 0 };
        
        for (let i = 1; i < this.datos.length; i++) {
            const actual = this._extraerBolillas(this.datos[i]);
            const anterior = this._extraerBolillas(this.datos[i-1]);
            let repetidos = 0;
            actual.forEach(n => { 
                if (anterior.includes(n)) {
                    repetidos++; 
                    if (n <= 22) rangosRepetidos['Inferior (0-22)']++;
                    else rangosRepetidos['Superior (23-45)']++;
                }
            });
            conteoRepeticiones[repetidos]++;
        }

        console.log("¿Se repite algún número del sorteo anterior?");
        Object.entries(conteoRepeticiones).forEach(([num, cant]) => {
            const porc = (cant / (this.datos.length - 1) * 100).toFixed(2);
            console.log(`   ${num} números repetidos: ${cant} veces (${porc}%)`);
        });

        console.log("\n¿De qué rango son los números que suelen repetirse?");
        const totalRep = rangosRepetidos['Inferior (0-22)'] + rangosRepetidos['Superior (23-45)'];
        Object.entries(rangosRepetidos).forEach(([rango, cant]) => {
            const porc = (cant / totalRep * 100).toFixed(2);
            console.log(`   ${rango}: ${cant} veces (${porc}% de las repeticiones)`);
        });
    }

    analizarFrecuenciaGlobal() {
        console.log("\n--- NÚMEROS QUE MÁS SALEN (FRECUENCIA GLOBAL) ---");
        const freq = {};
        this.datos.forEach(row => {
            this._extraerBolillas(row).forEach(n => freq[n] = (freq[n] || 0) + 1);
        });

        const sorted = Object.entries(freq).sort((a,b) => b[1] - a[1]);
        console.log("Top 10 números más frecuentes en el Quini 6:");
        sorted.slice(0, 10).forEach(([n, c]) => {
            console.log(`   Número ${n}: ${c} veces`);
        });
    }

    ejecutarTodo() {
        console.log("==================================================");
        console.log("🕵️ MINERÍA DE PATRONES - QUINI 6");
        console.log("==================================================");
        console.log(`Analizando ${this.datos.length} sorteos...\n`);
        
        this.analizarRepeticionesYRangos();
        this.analizarFrecuenciaGlobal();
        
        console.log("\n==================================================");
    }
}

const miner = new QuiniMiner();
miner.ejecutarTodo();
