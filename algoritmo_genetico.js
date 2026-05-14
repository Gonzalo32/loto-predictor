const fs = require('fs');

class OptimizadorGenetico {
    constructor(archivo, maxBolillas) {
        this.datos = this._cargarDatos(archivo);
        this.maxBolillas = maxBolillas; // Loto: 45, Quini: 45
        this.POBLACION_SIZE = 500;
        this.GENERACIONES = 100;
        this.MUTATION_RATE = 0.1;
    }

    _cargarDatos(filename) {
        if (!fs.existsSync(filename)) return [];
        const content = fs.readFileSync(filename, 'utf-8').trim().split('\n');
        const headers = content[0].split(',');
        return content.slice(1).map(line => {
            const values = line.split(',');
            const b = [
                parseInt(values[headers.indexOf('B1')]),
                parseInt(values[headers.indexOf('B2')]),
                parseInt(values[headers.indexOf('B3')]),
                parseInt(values[headers.indexOf('B4')]),
                parseInt(values[headers.indexOf('B5')]),
                parseInt(values[headers.indexOf('B6')])
            ];
            return b;
        });
    }

    _generarIndividuo() {
        const s = new Set();
        while(s.size < 6) s.add(Math.floor(Math.random() * (this.maxBolillas + 1)));
        return Array.from(s).sort((a,b) => a-b);
    }

    // El fitness busca maximizar aciertos históricos (peso masivo a 5 y 6 aciertos)
    _calcularFitness(individuo) {
        let puntaje = 0;
        // Solo evaluamos los últimos 200 sorteos para buscar tendencias "recientes" fuertes
        const limite = Math.min(200, this.datos.length);
        const ventana = this.datos.slice(0, limite); // están invertidos o no? Asumimos que [0] es el mas reciente si invertimos
        
        for (const sorteo of ventana) {
            let aciertos = 0;
            for (const num of individuo) {
                if (sorteo.includes(num)) aciertos++;
            }
            if (aciertos === 3) puntaje += 1;
            if (aciertos === 4) puntaje += 10;
            if (aciertos === 5) puntaje += 1000;
            if (aciertos === 6) puntaje += 100000;
        }
        return puntaje;
    }

    _cruzar(padre1, padre2) {
        const s = new Set();
        const pool = [...padre1, ...padre2];
        while(s.size < 6) {
            s.add(pool[Math.floor(Math.random() * pool.length)]);
        }
        return Array.from(s).sort((a,b)=>a-b);
    }

    _mutar(individuo) {
        if (Math.random() < this.MUTATION_RATE) {
            const idx = Math.floor(Math.random() * 6);
            let nuevo;
            do {
                nuevo = Math.floor(Math.random() * (this.maxBolillas + 1));
            } while(individuo.includes(nuevo));
            individuo[idx] = nuevo;
            individuo.sort((a,b)=>a-b);
        }
        return individuo;
    }

    evolucionar() {
        let poblacion = [];
        for(let i=0; i<this.POBLACION_SIZE; i++) poblacion.push(this._generarIndividuo());

        let mejorHistorico = { ind: [], fitness: -1 };

        for(let gen=0; gen<this.GENERACIONES; gen++) {
            const evaluados = poblacion.map(ind => ({ ind, fitness: this._calcularFitness(ind) }));
            evaluados.sort((a,b) => b.fitness - a.fitness);

            if (evaluados[0].fitness > mejorHistorico.fitness) {
                mejorHistorico = evaluados[0];
            }

            const nuevaPoblacion = [];
            // Elitismo
            for(let i=0; i<50; i++) nuevaPoblacion.push(evaluados[i].ind);

            while(nuevaPoblacion.length < this.POBLACION_SIZE) {
                const p1 = evaluados[Math.floor(Math.random() * 50)].ind;
                const p2 = evaluados[Math.floor(Math.random() * 50)].ind;
                let hijo = this._cruzar(p1, p2);
                hijo = this._mutar(hijo);
                nuevaPoblacion.push(hijo);
            }
            poblacion = nuevaPoblacion;
        }
        return mejorHistorico.ind;
    }
}

console.log("==================================================");
console.log("🧬 ALGORITMO GENÉTICO: BUSCADOR DE COMBINACIONES PERFECTAS");
console.log("==================================================");

console.log("⏳ Entrenando Red Evolutiva para Loto Plus...");
const lotoDatos = fs.readFileSync('historico_loto.csv', 'utf8').split('\n').slice(1).reverse(); // Recientes primero
fs.writeFileSync('temp_loto.csv', 'B1,B2,B3,B4,B5,B6\n' + lotoDatos.join('\n'));
const gaLoto = new OptimizadorGenetico('temp_loto.csv', 45);
const lotoTicket = gaLoto.evolucionar();

console.log("\n⏳ Entrenando Red Evolutiva para Quini 6...");
const quiniDatos = fs.readFileSync('historico_quini_limpio.csv', 'utf8').split('\n').slice(1); // Quini ya tiene recientes primero
fs.writeFileSync('temp_quini.csv', 'B1,B2,B3,B4,B5,B6\n' + quiniDatos.join('\n'));
const gaQuini = new OptimizadorGenetico('temp_quini.csv', 45);
const quiniTicket = gaQuini.evolucionar();

console.log("\n==================================================");
console.log("🌟 RESULTADOS DEL ALGORITMO GENÉTICO (BUSQUEDA PROFUNDA)");
console.log("==================================================");
console.log(`🎰 TICKET ÓPTIMO LOTO PLUS : [ ${lotoTicket.map(n => String(n).padStart(2,'0')).join(' - ')} ]`);
console.log(`🎰 TICKET ÓPTIMO QUINI 6   : [ ${quiniTicket.map(n => String(n).padStart(2,'0')).join(' - ')} ]`);
console.log("==================================================");

fs.unlinkSync('temp_loto.csv');
fs.unlinkSync('temp_quini.csv');
