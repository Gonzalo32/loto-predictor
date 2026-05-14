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

console.log("==================================================");
console.log("🧠 OPTIMIZADOR V4 (Consenso Dinámico + Factor de Momento)");
console.log("==================================================");
console.log("Calculando pesos dinámicos + multiplicadores de racha de números...");

const datosQuini = cargarDatos('historico_quini_limpio.csv').reverse();
const sorteosIniciales = 100;
const VENTANA_EVALUACION = 10;

let sorteosJugados = 0;
let aciertosV4 = 0;
const historialTiempos = [];
let sorteosDesdeUltimoAcierto = 0;

const estrategias = [
    'sugerirPorMarkov', 'sugerirPorDelta', 'sugerirPorBalancePares', 
    'sugerirPorSumaCampanaGauss', 'sugerirPorDecenas', 'sugerirPorSimilitud', 
    'sugerirPorAtraso', 'sugerirPorConsecutivos', 'sugerirPorTendenciaLineal', 
    'sugerirPorNumerosPrimos', 'sugerirPorTerminaciones', 'sugerirPorMitades', 
    'sugerirPorRangoTotal', 'sugerirPorSumaDigitos'
];

for (let i = sorteosIniciales; i < datosQuini.length; i++) {
    // 1. Pesos Dinámicos (Como en la V3)
    const pesosDinamicos = {};
    estrategias.forEach(e => pesosDinamicos[e] = 0);
    const inicioVentana = Math.max(2, i - VENTANA_EVALUACION);
    
    for (let j = inicioVentana; j < i; j++) {
        const historyTemp = datosQuini.slice(0, j);
        const predTemp = new LotoPredictor(historyTemp);
        const ulTemp = extraerBolillas(historyTemp[historyTemp.length - 1]);
        const realSort = extraerBolillas(datosQuini[j]);
        for (const est of estrategias) {
            let p = (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') 
                ? predTemp[est](ulTemp).numeros 
                : predTemp[est]().numeros;
            p.forEach(n => { if (realSort.includes(n)) pesosDinamicos[est]++; });
        }
    }

    // 2. Calcular Momentum de cada número (Novedad de la V4)
    const momentum = {};
    for (let n = 0; n <= 45; n++) momentum[n] = 1.0; // Base neutral

    // Revisar últimos 3 sorteos (Racha Caliente)
    const ultimos3 = datosQuini.slice(Math.max(0, i-3), i);
    const setCalientes = new Set();
    ultimos3.forEach(fila => extraerBolillas(fila).forEach(b => setCalientes.add(b)));
    setCalientes.forEach(b => momentum[b] *= 1.5); // Bonus del 50% a los que vienen saliendo

    // Revisar últimos 20 sorteos (Racha Fría / Atrasos extremos)
    const ultimos20 = datosQuini.slice(Math.max(0, i-20), i);
    const setVistos = new Set();
    ultimos20.forEach(fila => extraerBolillas(fila).forEach(b => setVistos.add(b)));
    for (let n = 0; n <= 45; n++) {
        if (!setVistos.has(n)) momentum[n] *= 0.6; // Penalización del 40% a los congelados
    }

    // 3. Predicción con Consenso + Momentum
    const historialEntrenamiento = datosQuini.slice(0, i);
    const predictor = new LotoPredictor(historialEntrenamiento);
    const ultimasBolillas = extraerBolillas(historialEntrenamiento[historialEntrenamiento.length - 1]);
    
    const votos = {};
    for (const est of estrategias) {
        let pred = (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') 
            ? predictor[est](ultimasBolillas).numeros 
            : predictor[est]().numeros;
        const pesoAsignado = Math.max(1, pesosDinamicos[est]); 
        pred.forEach(n => votos[n] = (votos[n] || 0) + pesoAsignado);
    }

    // Aplicar Factor de Momento
    const votosFinales = {};
    for (const [num, count] of Object.entries(votos)) {
        votosFinales[num] = count * momentum[num];
    }

    const superTicketV4 = Object.entries(votosFinales)
        .sort((a,b) => b[1] - a[1])
        .slice(0, 6)
        .map(x => parseInt(x[0]))
        .sort((a,b) => a-b);

    const sorteoReal = extraerBolillas(datosQuini[i]);
    let aciertos = 0;
    for (const num of superTicketV4) if (sorteoReal.includes(num)) aciertos++;

    sorteosJugados++;
    sorteosDesdeUltimoAcierto++;

    if (aciertos >= 4) {
        aciertosV4++;
        historialTiempos.push(sorteosDesdeUltimoAcierto);
        sorteosDesdeUltimoAcierto = 0;
    }
}

console.log(`\nSimulación completada en ${sorteosJugados} sorteos.`);
console.log(`\n📊 COMPARATIVA ÉPICA (4+ ACIERTOS)`);
console.log(`   🔸 V1 (Votos Iguales)     : 11 veces.`);
console.log(`   🔸 V2 (Pesos Fijos)       : 86 veces.`);
console.log(`   🔸 V3 (Dinámico)          : 126 veces.`);
console.log(`   🚀 V4 (Dinámico+Momento)  : ${aciertosV4} veces.`);

if (aciertosV4 > 126) {
    const mejora = (((aciertosV4 - 126) / 126) * 100).toFixed(1);
    console.log(`\n👑 ¡LA V4 DESTRUYÓ EL RÉCORD! Mejoró la predicción en un +${mejora}% respecto a la V3.`);
    const promedio = historialTiempos.reduce((a,c) => a+c, 0) / historialTiempos.length;
    console.log(`   Tiempo de espera promedio: ${promedio.toFixed(1)} sorteos.`);
} else {
    console.log(`\n⚠️ La V4 no superó a la V3. El factor de momento introdujo demasiado ruido.`);
    console.log(`La versión 3 (126 veces) sigue siendo nuestro techo matemático.`);
}
console.log("==================================================");
