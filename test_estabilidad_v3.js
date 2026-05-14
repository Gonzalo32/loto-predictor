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
console.log("🌪️ PRUEBA DE ESTABILIDAD V3 (5 EJECUCIONES COMPLETAS)");
console.log("==================================================");

const datosQuini = cargarDatos('historico_quini_limpio.csv').reverse();
const sorteosIniciales = 100;
const VENTANA_EVALUACION = 10;

const estrategias = [
    'sugerirPorMarkov', 'sugerirPorDelta', 'sugerirPorBalancePares', 
    'sugerirPorSumaCampanaGauss', 'sugerirPorDecenas', 'sugerirPorSimilitud', 
    'sugerirPorAtraso', 'sugerirPorConsecutivos', 'sugerirPorTendenciaLineal', 
    'sugerirPorNumerosPrimos', 'sugerirPorTerminaciones', 'sugerirPorMitades', 
    'sugerirPorRangoTotal', 'sugerirPorSumaDigitos'
];

const resultados = [];
const NUM_PASADAS = 5;

for (let pasada = 1; pasada <= NUM_PASADAS; pasada++) {
    console.log(`⏳ Iniciando Pasada ${pasada} de ${NUM_PASADAS}...`);
    let aciertosDinamicos = 0;

    for (let i = sorteosIniciales; i < datosQuini.length; i++) {
        const pesosDinamicos = {};
        estrategias.forEach(e => pesosDinamicos[e] = 0);
        const inicioVentana = Math.max(2, i - VENTANA_EVALUACION);
        
        for (let j = inicioVentana; j < i; j++) {
            const historyTemp = datosQuini.slice(0, j);
            const predTemp = new LotoPredictor(historyTemp);
            const ulTemp = extraerBolillas(historyTemp[historyTemp.length - 1]);
            const realSort = extraerBolillas(datosQuini[j]);
            
            for (const est of estrategias) {
                let p = (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') ? predTemp[est](ulTemp).numeros : predTemp[est]().numeros;
                p.forEach(n => { if (realSort.includes(n)) pesosDinamicos[est]++; });
            }
        }

        const historialEntrenamiento = datosQuini.slice(0, i);
        const predictor = new LotoPredictor(historialEntrenamiento);
        const ultimasBolillas = extraerBolillas(historialEntrenamiento[historialEntrenamiento.length - 1]);
        
        const votos = {};
        for (const est of estrategias) {
            let pred = (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') ? predictor[est](ultimasBolillas).numeros : predictor[est]().numeros;
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

        if (aciertos >= 4) {
            aciertosDinamicos++;
        }
    }
    
    console.log(`   ✅ Pasada ${pasada} finalizada: ${aciertosDinamicos} veces se logró 4+ aciertos.`);
    resultados.push(aciertosDinamicos);
}

console.log("\n==================================================");
console.log("📊 RESULTADOS FINALES DE ESTABILIDAD (V3)");
console.log("==================================================");
resultados.forEach((r, idx) => console.log(`   🔸 Pasada ${idx + 1}: ${r} aciertos de 4+`));

const promedio = resultados.reduce((a, b) => a + b, 0) / resultados.length;
const min = Math.min(...resultados);
const max = Math.max(...resultados);

console.log(`\n📈 Promedio de aciertos V3 : ${promedio.toFixed(1)} veces por simulación.`);
console.log(`📉 Rango de fluctuación    : de ${min} a ${max} veces.`);
console.log("==================================================");
