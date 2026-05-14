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
console.log("⏱️ SIMULADOR DE ESPERANZA MATEMÁTICA (Quini 6)");
console.log("==================================================");
console.log("Objetivo: ¿Cuántos sorteos hay que jugar el Súper Ticket para acertar 4+ números?");

const datosQuini = cargarDatos('historico_quini_limpio.csv').reverse(); // Orden cronológico (antiguo a nuevo)
const sorteosIniciales = 100; // Empezamos a jugar a partir del sorteo 100 para tener historial

let sorteosJugados = 0;
let sorteosDesdeUltimoAcierto = 0;
let aciertosDe4oMas = 0;
const historialTiempos = [];

console.log(`\nSimulando juego continuo a lo largo de ${datosQuini.length - sorteosIniciales} sorteos históricos...`);

// Para que sea rápido, hacemos un Consenso Directo (sin las 10.000 iteraciones de Montecarlo)
// Simplemente tomamos lo que dicen los algoritmos y elegimos los 6 más votados.
const estrategias = [
    'sugerirPorMarkov', 'sugerirPorDelta', 'sugerirPorBalancePares', 
    'sugerirPorSumaCampanaGauss', 'sugerirPorDecenas', 'sugerirPorSimilitud', 
    'sugerirPorAtraso', 'sugerirPorConsecutivos', 'sugerirPorTendenciaLineal', 
    'sugerirPorNumerosPrimos', 'sugerirPorTerminaciones', 'sugerirPorMitades', 
    'sugerirPorRangoTotal', 'sugerirPorSumaDigitos'
];

for (let i = sorteosIniciales; i < datosQuini.length; i++) {
    const historialEntrenamiento = datosQuini.slice(0, i);
    const predictor = new LotoPredictor(historialEntrenamiento);
    const ultimasBolillas = extraerBolillas(historialEntrenamiento[historialEntrenamiento.length - 1]);
    
    // Votación rápida
    const votos = {};
    for (const est of estrategias) {
        let pred;
        if (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') {
            pred = predictor[est](ultimasBolillas).numeros;
        } else {
            pred = predictor[est]().numeros;
        }
        pred.forEach(n => votos[n] = (votos[n] || 0) + 1);
    }

    // Armar el "Súper Ticket Rápido" con los 6 más votados
    const superTicket = Object.entries(votos)
        .sort((a,b) => b[1] - a[1])
        .slice(0, 6)
        .map(x => parseInt(x[0]))
        .sort((a,b) => a-b);

    // Comparar contra el sorteo real que ocurrió ese día
    const sorteoReal = extraerBolillas(datosQuini[i]);
    let aciertos = 0;
    for (const num of superTicket) {
        if (sorteoReal.includes(num)) aciertos++;
    }

    sorteosJugados++;
    sorteosDesdeUltimoAcierto++;

    if (aciertos >= 4) {
        aciertosDe4oMas++;
        historialTiempos.push(sorteosDesdeUltimoAcierto);
        console.log(`🎯 ¡BINGO! ${aciertos} aciertos en el sorteo de fecha ${datosQuini[i].Fecha}. Tomó ${sorteosDesdeUltimoAcierto} jugadas lograrlo.`);
        sorteosDesdeUltimoAcierto = 0; // Resetear contador
    }
}

console.log("\n==================================================");
console.log("📊 RESULTADOS FINALES DE LA SIMULACIÓN");
console.log("==================================================");
console.log(`Total de sorteos jugados (Súper Tickets comprados): ${sorteosJugados}`);
console.log(`Cantidad de veces que se logró 4 o más aciertos: ${aciertosDe4oMas}`);

if (aciertosDe4oMas > 0) {
    const promedio = historialTiempos.reduce((a,c) => a+c, 0) / historialTiempos.length;
    console.log(`\n⏳ ESPERANZA MATEMÁTICA:`);
    console.log(`En promedio, debes jugar el Súper Ticket durante ${promedio.toFixed(1)} sorteos seguidos para pegarle a 4 números.`);
    console.log(`Tiempos de espera registrados: [ ${historialTiempos.join(', ')} ] sorteos.`);
} else {
    console.log(`\n📉 Lamentablemente, en toda la simulación histórica no se logró ningún acierto de 4+.`);
    console.log(`Esto demuestra la extrema dificultad de vencer el azar puro en este nivel.`);
}
console.log("==================================================");
