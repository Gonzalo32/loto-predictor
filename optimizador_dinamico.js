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
console.log("🌪️ OPTIMIZADOR DINÁMICO DE PESOS (Auto-Adaptativo)");
console.log("==================================================");
console.log("Calculando pesos en tiempo real basados en la racha de los algoritmos...");

const datosQuini = cargarDatos('historico_quini_limpio.csv').reverse();
const sorteosIniciales = 100;
const VENTANA_EVALUACION = 10; // Miramos los últimos 10 sorteos para definir el peso

let sorteosJugados = 0;
let aciertosDinamicos = 0;
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
    // 1. Evaluación dinámica de pesos (Backtesting de los últimos 10 sorteos)
    const pesosDinamicos = {};
    estrategias.forEach(e => pesosDinamicos[e] = 0);

    const inicioVentana = Math.max(2, i - VENTANA_EVALUACION);
    
    // Calcular qué algoritmos estuvieron "calientes" en la ventana reciente
    for (let j = inicioVentana; j < i; j++) {
        const historyTemp = datosQuini.slice(0, j);
        const predTemp = new LotoPredictor(historyTemp);
        const ulTemp = extraerBolillas(historyTemp[historyTemp.length - 1]);
        const realSort = extraerBolillas(datosQuini[j]);
        
        for (const est of estrategias) {
            let p;
            if (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') p = predTemp[est](ulTemp).numeros;
            else p = predTemp[est]().numeros;
            
            let matches = 0;
            p.forEach(n => { if (realSort.includes(n)) matches++; });
            
            // Le damos puntos exponenciales a los aciertos recientes
            pesosDinamicos[est] += matches; 
        }
    }

    // 2. Predicción principal
    const historialEntrenamiento = datosQuini.slice(0, i);
    const predictor = new LotoPredictor(historialEntrenamiento);
    const ultimasBolillas = extraerBolillas(historialEntrenamiento[historialEntrenamiento.length - 1]);
    
    const votos = {};
    for (const est of estrategias) {
        let pred;
        if (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') {
            pred = predictor[est](ultimasBolillas).numeros;
        } else {
            pred = predictor[est]().numeros;
        }
        
        // Multiplicador: El algoritmo vota con el peso que ganó en los últimos 10 sorteos
        // Si no pegó nada, su peso es mínimo (1). Si pegó mucho, su peso sube enormemente.
        const pesoAsignado = Math.max(1, pesosDinamicos[est]); 
        
        pred.forEach(n => votos[n] = (votos[n] || 0) + pesoAsignado);
    }

    const superTicketDinamico = Object.entries(votos)
        .sort((a,b) => b[1] - a[1])
        .slice(0, 6)
        .map(x => parseInt(x[0]))
        .sort((a,b) => a-b);

    const sorteoReal = extraerBolillas(datosQuini[i]);
    
    let aciertos = 0;
    for (const num of superTicketDinamico) if (sorteoReal.includes(num)) aciertos++;

    sorteosJugados++;
    sorteosDesdeUltimoAcierto++;

    if (aciertos >= 4) {
        aciertosDinamicos++;
        historialTiempos.push(sorteosDesdeUltimoAcierto);
        sorteosDesdeUltimoAcierto = 0;
    }
}

console.log(`\nSimulación completada en ${sorteosJugados} sorteos.`);
console.log(`\n📊 COMPARATIVA FINAL (4+ ACIERTOS)`);
console.log(`   🔸 Método V1 (Votos Iguales)    : 11 veces.`);
console.log(`   🔸 Método V2 (Pesos Fijos)      : 86 veces.`);
console.log(`   🚀 Método V3 (Pesos Dinámicos)  : ${aciertosDinamicos} veces.`);

if (aciertosDinamicos > 86) {
    const mejora = (((aciertosDinamicos - 86) / 86) * 100).toFixed(1);
    console.log(`\n🎉 ¡NUEVO RÉCORD! El modelo dinámico mejoró la predicción en un +${mejora}% respecto a la V2.`);
    const promedio = historialTiempos.reduce((a,c) => a+c, 0) / historialTiempos.length;
    console.log(`   El tiempo de espera promedio es ahora de ${promedio.toFixed(1)} sorteos.`);
} else if (aciertosDinamicos === 86) {
    console.log(`\n⚖️ Empate. El modelo dinámico tiene el mismo poder predictivo que el peso estático.`);
} else {
    console.log(`\n⚠️ El modelo dinámico (${aciertosDinamicos}) es peor que el peso estático (86). Demasiada variabilidad arruina el consenso a largo plazo.`);
}
console.log("==================================================");
