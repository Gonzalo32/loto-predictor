const fs = require('fs');
const LotoPredictor = require('./predictor');

function cargarDatos(filename) {
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

function extraerBolillas(fila) {
    return [
        parseInt(fila['B1']), parseInt(fila['B2']), parseInt(fila['B3']),
        parseInt(fila['B4']), parseInt(fila['B5']), parseInt(fila['B6'])
    ].sort((a,b)=>a-b);
}

const estrategias = [
    'sugerirPorMarkov', 'sugerirPorDelta', 'sugerirPorBalancePares', 
    'sugerirPorSumaCampanaGauss', 'sugerirPorDecenas', 'sugerirPorSimilitud', 
    'sugerirPorAtraso', 'sugerirPorConsecutivos', 'sugerirPorTendenciaLineal', 
    'sugerirPorNumerosPrimos', 'sugerirPorTerminaciones', 'sugerirPorMitades', 
    'sugerirPorRangoTotal', 'sugerirPorSumaDigitos'
];

function generarTicketsPonderados(archivo, esQuini) {
    // Para Loto, el CSV normal ya tiene el más reciente al final? No, en Loto historico_loto.csv:
    // La línea 513 es la última fecha.
    let datos = cargarDatos(archivo);
    
    // Si es Quini, el limpio tiene el más reciente al PRINCIPIO (índice 0).
    // Si es Loto, el historico_loto.csv tiene el más reciente al FINAL.
    // LotoPredictor asume que el orden de los datos es [más_antiguo, ..., más_reciente]
    if (esQuini) {
        datos = datos.reverse(); // Lo ponemos de antiguo a reciente
    }

    const predictor = new LotoPredictor(datos);
    const ultimasBolillas = extraerBolillas(datos[datos.length - 1]);
    
    const freqNumeros = {};
    const freqPlus = {};

    // Hacemos 1000 iteraciones del motor para estabilizar la aleatoriedad de los fallbacks
    for(let i = 0; i < 1000; i++) {
        for (const est of estrategias) {
            let pred;
            if (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') {
                pred = predictor[est](ultimasBolillas);
            } else {
                pred = predictor[est]();
            }

            let peso = 1;
            if (esQuini) {
                if (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') peso = 2;
            } else {
                if (est === 'sugerirPorBalancePares' || est === 'sugerirPorSimilitud' || est === 'sugerirPorMarkov') peso = 2;
            }

            pred.numeros.forEach(n => freqNumeros[n] = (freqNumeros[n] || 0) + peso);
            if (!esQuini) {
                freqPlus[pred.plus] = (freqPlus[pred.plus] || 0) + peso;
            }
        }
    }

    const topNumeros = Object.entries(freqNumeros)
        .sort((a,b) => b[1] - a[1])
        .map(x => parseInt(x[0], 10));

    const t1 = [topNumeros[0], topNumeros[3], topNumeros[6], topNumeros[9], topNumeros[12], topNumeros[15]].sort((a,b)=>a-b);
    const t2 = [topNumeros[1], topNumeros[4], topNumeros[7], topNumeros[10], topNumeros[13], topNumeros[16]].sort((a,b)=>a-b);
    const t3 = [topNumeros[2], topNumeros[5], topNumeros[8], topNumeros[11], topNumeros[14], topNumeros[17]].sort((a,b)=>a-b);

    let plus = [0,0,0];
    if (!esQuini) {
        const topP = Object.entries(freqPlus).sort((a,b)=>b[1]-a[1]).map(x=>parseInt(x[0], 10));
        plus = [topP[0], topP[1], topP[2]];
    }

    return { t1, t2, t3, plus };
}

console.log("==================================================");
console.log("🎟️ TICKETS DE VOTACIÓN PONDERADA (ESTE FIN DE SEMANA)");
console.log("==================================================");

const resQuini = generarTicketsPonderados('historico_quini_limpio.csv', true);
console.log("\n🎰 QUINI 6 (Sorteo Domingo):");
console.log(`   🔸 Ticket A (Alta Probabilidad) : [ ${resQuini.t1.map(n=>String(n).padStart(2,'0')).join(' ')} ]`);
console.log(`   🔸 Ticket B (Soporte Estadístico): [ ${resQuini.t2.map(n=>String(n).padStart(2,'0')).join(' ')} ]`);
console.log(`   🔸 Ticket C (Cobertura Total)    : [ ${resQuini.t3.map(n=>String(n).padStart(2,'0')).join(' ')} ]`);

const resLoto = generarTicketsPonderados('historico_loto.csv', false);
console.log("\n🎰 LOTO PLUS (Sorteo Sábado):");
console.log(`   🔸 Ticket A (Alta Probabilidad) : [ ${resLoto.t1.map(n=>String(n).padStart(2,'0')).join(' ')} ] | Plus: ${resLoto.plus[0]}`);
console.log(`   🔸 Ticket B (Soporte Estadístico): [ ${resLoto.t2.map(n=>String(n).padStart(2,'0')).join(' ')} ] | Plus: ${resLoto.plus[1]}`);
console.log(`   🔸 Ticket C (Cobertura Total)    : [ ${resLoto.t3.map(n=>String(n).padStart(2,'0')).join(' ')} ] | Plus: ${resLoto.plus[2]}`);
console.log("\n==================================================");
