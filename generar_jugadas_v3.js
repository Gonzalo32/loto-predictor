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

function predecirV3(archivo, esQuini) {
    let datos = cargarDatos(archivo);
    if (esQuini) datos = datos.reverse(); // Ordenamos de antiguo a reciente

    const VENTANA_EVALUACION = 10;
    const pesosDinamicos = {};
    estrategias.forEach(e => pesosDinamicos[e] = 0);

    // Calculamos qué algoritmos vienen en racha en los últimos 10 sorteos
    const indexActual = datos.length;
    const inicioVentana = Math.max(2, indexActual - VENTANA_EVALUACION);

    for (let j = inicioVentana; j < indexActual; j++) {
        const historyTemp = datos.slice(0, j);
        const predTemp = new LotoPredictor(historyTemp);
        const ulTemp = extraerBolillas(historyTemp[historyTemp.length - 1]);
        const realSort = extraerBolillas(datos[j]);
        
        for (const est of estrategias) {
            let p;
            if (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') p = predTemp[est](ulTemp).numeros;
            else p = predTemp[est]().numeros;
            
            let matches = 0;
            p.forEach(n => { if (realSort.includes(n)) matches++; });
            pesosDinamicos[est] += matches; 
        }
    }

    // Usamos los pesos aprendidos para predecir el próximo sorteo
    const predictor = new LotoPredictor(datos);
    const ultimasBolillas = extraerBolillas(datos[datos.length - 1]);
    
    const freqNumeros = {};
    const freqPlus = {};

    // Iteramos para mitigar la aleatoriedad de empates/fallbacks
    for(let i = 0; i < 1000; i++) {
        for (const est of estrategias) {
            let pred;
            if (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') {
                pred = predictor[est](ultimasBolillas);
            } else {
                pred = predictor[est]();
            }

            const peso = Math.max(1, pesosDinamicos[est]);

            pred.numeros.forEach(n => freqNumeros[n] = (freqNumeros[n] || 0) + peso);
            if (!esQuini) {
                freqPlus[pred.plus] = (freqPlus[pred.plus] || 0) + peso;
            }
        }
    }

    const topNumeros = Object.entries(freqNumeros)
        .sort((a,b) => b[1] - a[1])
        .map(x => parseInt(x[0], 10));

    // Formamos los 3 Súper Tickets (A=Riesgo/Probabilidad Alta, B=Consenso Medio, C=Cobertura)
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
console.log("🚀 TICKETS GENERADOS CON MOTOR V3 (AUTO-ADAPTATIVO)");
console.log("==================================================");

const resQuini = predecirV3('historico_quini_limpio.csv', true);
console.log("\n🎰 QUINI 6 (Sorteo Domingo):");
console.log(`   🔸 Ticket A (Alta Probabilidad) : [ ${resQuini.t1.map(n=>String(n).padStart(2,'0')).join(' ')} ]`);
console.log(`   🔸 Ticket B (Soporte Estadístico): [ ${resQuini.t2.map(n=>String(n).padStart(2,'0')).join(' ')} ]`);
console.log(`   🔸 Ticket C (Cobertura Total)    : [ ${resQuini.t3.map(n=>String(n).padStart(2,'0')).join(' ')} ]`);

const resLoto = predecirV3('historico_loto.csv', false);
console.log("\n🎰 LOTO PLUS (Sorteo Sábado):");
console.log(`   🔸 Ticket A (Alta Probabilidad) : [ ${resLoto.t1.map(n=>String(n).padStart(2,'0')).join(' ')} ] | Plus: ${resLoto.plus[0]}`);
console.log(`   🔸 Ticket B (Soporte Estadístico): [ ${resLoto.t2.map(n=>String(n).padStart(2,'0')).join(' ')} ] | Plus: ${resLoto.plus[1]}`);
console.log(`   🔸 Ticket C (Cobertura Total)    : [ ${resLoto.t3.map(n=>String(n).padStart(2,'0')).join(' ')} ] | Plus: ${resLoto.plus[2]}`);
console.log("\n==================================================");
