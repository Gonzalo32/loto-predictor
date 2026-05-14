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

// Filtro Exótico para Quini 6 basado en la minería previa
function esCombinacionValidaQuini(ticket) {
    // 1. Evitar números pegados (Distancia mínima 2)
    for(let i=1; i<ticket.length; i++) {
        if (ticket[i] - ticket[i-1] < 2) return false; 
    }
    
    // 2. Control de Zonas Muertas (limitar números altos)
    let altos = ticket.filter(n => n >= 30).length;
    if (altos > 2) return false; // En Quini los altos suelen faltar, máximo 2 altos por ticket.

    return true;
}

console.log("==================================================");
console.log("🚀 OPTIMIZADOR PARA 4+ ACIERTOS (Quini 6)");
console.log("==================================================");
console.log("Aplicando filtros exóticos (Sin consecutivos, Máximo 2 números altos)...");

const datosQuini = cargarDatos('historico_quini_limpio.csv').reverse();
const sorteosIniciales = 100;

let sorteosJugados = 0;
let aciertosDe4oMas = 0;
let aciertosAntiguos = 0;
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
    const historialEntrenamiento = datosQuini.slice(0, i);
    const predictor = new LotoPredictor(historialEntrenamiento);
    const ultimasBolillas = extraerBolillas(historialEntrenamiento[historialEntrenamiento.length - 1]);
    
    // Votación Ponderada (le damos x2 a Markov y Similitud que son los mejores en Quini)
    const votos = {};
    for (const est of estrategias) {
        let pred;
        if (est === 'sugerirPorMarkov' || est === 'sugerirPorSimilitud') {
            pred = predictor[est](ultimasBolillas).numeros;
            pred.forEach(n => votos[n] = (votos[n] || 0) + 2); // Peso x2
        } else {
            pred = predictor[est]().numeros;
            pred.forEach(n => votos[n] = (votos[n] || 0) + 1); // Peso x1
        }
    }

    const ordenados = Object.entries(votos).sort((a,b) => b[1] - a[1]).map(x => parseInt(x[0]));
    
    // Antiguo método (solo agarrar los 6 primeros sin filtrar)
    const superTicketAntiguo = ordenados.slice(0, 6).sort((a,b)=>a-b);
    
    // Nuevo método: Filtrado Exótico
    let superTicketFiltrado = [];
    // Intentar armar un ticket válido iterando sobre los más votados
    // Necesitamos combinaciones de 6. Para no hacer fuerza bruta completa, usamos una heurística greedy:
    for (const num of ordenados) {
        let temp = [...superTicketFiltrado, num].sort((a,b)=>a-b);
        // Validamos la inserción (relajada para tickets parciales)
        let valido = true;
        for(let j=1; j<temp.length; j++) if(temp[j] - temp[j-1] < 2) valido = false;
        let altos = temp.filter(n => n >= 30).length;
        if(altos > 2) valido = false;

        if (valido) {
            superTicketFiltrado.push(num);
            superTicketFiltrado.sort((a,b)=>a-b);
        }
        if (superTicketFiltrado.length === 6) break;
    }
    
    // Fallback si el filtro es muy estricto
    if (superTicketFiltrado.length < 6) {
        superTicketFiltrado = superTicketAntiguo; 
    }

    const sorteoReal = extraerBolillas(datosQuini[i]);
    
    let aciertosAnt = 0;
    for (const num of superTicketAntiguo) if (sorteoReal.includes(num)) aciertosAnt++;
    if (aciertosAnt >= 4) aciertosAntiguos++;

    let aciertosNuevos = 0;
    for (const num of superTicketFiltrado) if (sorteoReal.includes(num)) aciertosNuevos++;

    sorteosJugados++;
    sorteosDesdeUltimoAcierto++;

    if (aciertosNuevos >= 4) {
        aciertosDe4oMas++;
        historialTiempos.push(sorteosDesdeUltimoAcierto);
        sorteosDesdeUltimoAcierto = 0;
    }
}

console.log(`\nSimulación completada en ${sorteosJugados} sorteos.`);
console.log(`\n📊 COMPARATIVA DE RENDIMIENTO (4+ ACIERTOS)`);
console.log(`   🔸 Método Anterior (Consenso Puro): ${aciertosAntiguos} veces.`);
console.log(`   🟢 Método Nuevo (Consenso + Filtro Exótico): ${aciertosDe4oMas} veces.`);

const mejora = (((aciertosDe4oMas - aciertosAntiguos) / aciertosAntiguos) * 100).toFixed(1);

if (aciertosDe4oMas > aciertosAntiguos) {
    console.log(`\n🎉 ¡LOGRADO! El nuevo cálculo aumentó los premios en un +${mejora}%.`);
    const promedio = historialTiempos.reduce((a,c) => a+c, 0) / historialTiempos.length;
    console.log(`   Ahora el tiempo de espera promedio se redujo a ${promedio.toFixed(1)} sorteos.`);
} else {
    console.log(`\n⚠️ El filtro no logró superar la varianza natural del modelo anterior.`);
}
console.log("==================================================");
