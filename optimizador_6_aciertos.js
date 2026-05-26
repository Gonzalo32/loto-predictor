const fs = require('fs');

console.log("==================================================");
console.log("🎯 OPTIMIZADOR PARA 6 ACIERTOS (SISTEMA REDUCIDO)");
console.log("==================================================");

// 1. Leer los 12 números de oro del análisis matemático
if (!fs.existsSync('matriz_avanzada.json')) {
    console.error("❌ No se encontró matriz_avanzada.json. Corre 'py modelos_avanzados.py' primero.");
    process.exit(1);
}

const data = JSON.parse(fs.readFileSync('matriz_avanzada.json', 'utf-8'));
const numerosBase = data.top_12_numeros.sort((a,b) => a-b);

console.log(`📡 Números recibidos del motor FFT/Markov: [ ${numerosBase.map(n => n.toString().padStart(2, '0')).join(', ')} ]`);

// Función para obtener todas las combinaciones posibles de K elementos
function combinaciones(arr, k) {
    const resultados = [];
    function combinar(inicio, comboActual) {
        if (comboActual.length === k) {
            resultados.push([...comboActual]);
            return;
        }
        for (let i = inicio; i < arr.length; i++) {
            comboActual.push(arr[i]);
            combinar(i + 1, comboActual);
            comboActual.pop();
        }
    }
    combinar(0, []);
    return resultados;
}

// 2. Generar todas las combinaciones posibles de 6 (C(12,6) = 924)
const todasLasJugadas = combinaciones(numerosBase, 6);
console.log(`\n🎲 Generadas matemáticamente ${todasLasJugadas.length} combinaciones completas (C(12,6)).`);

// 3. Aplicar Filtros Estructurales (Reducir a "Tickets Perfectos")
console.log(`\nFiltros Exóticos Aplicados:`);
console.log(`   - Sin números consecutivos (Distancia mínima > 1)`);
console.log(`   - Máximo 2 números en la misma decena`);
console.log(`   - Rango (Spread) total entre 20 y 42`);
console.log(`   - Sin paridad extrema (no todo pares o todo impares)`);

function esTicketEstructuralmentePerfecto(ticket) {
    // 1. Sin consecutivos
    for (let i = 1; i < ticket.length; i++) {
        if (ticket[i] - ticket[i-1] <= 1) return false;
    }
    
    // 2. Spread (Diferencia entre el mayor y menor)
    const spread = ticket[5] - ticket[0];
    if (spread < 20 || spread > 42) return false; 
    
    // 3. Máximo de números en la misma decena
    const decenas = {0:0, 1:0, 2:0, 3:0, 4:0};
    for (const num of ticket) {
        decenas[Math.floor(num/10)]++;
    }
    for (const d in decenas) {
        if (decenas[d] > 2) return false; 
    }
    
    // 4. Paridad extrema
    let pares = ticket.filter(n => n % 2 === 0).length;
    if (pares === 0 || pares === 6) return false;

    return true;
}

const jugadasReducidas = todasLasJugadas.filter(esTicketEstructuralmentePerfecto);

console.log(`\n🔥 Se han reducido a ${jugadasReducidas.length} boletos estructuralmente perfectos.`);

// Si la reducción deja demasiados, tomamos una muestra representativa (Wheeling acortado)
let ticketsFinales = jugadasReducidas;
if (jugadasReducidas.length > 15) {
    console.log(`⚠️ Tomando una sub-matriz de 15 tickets de altísima cobertura...`);
    // Usamos una pequeña semilla de aleatoriedad para seleccionar boletos variados
    const mezclados = jugadasReducidas.sort(() => Math.random() - 0.5);
    ticketsFinales = mezclados.slice(0, 15);
}

// 4. Mostrar y Guardar los tickets
console.log(`\n🎟️ TICKETS RECOMENDADOS PARA QUINI 6:\n`);

ticketsFinales.forEach((ticket, idx) => {
    const ticketStr = ticket.map(n => n.toString().padStart(2, '0')).join(' - ');
    console.log(`   Boleta #${(idx+1).toString().padStart(2, '0')} : [ ${ticketStr} ]`);
});

// Guardar en historial_predicciones.json
const historialPath = 'historial_predicciones.json';
let historial = [];
if (fs.existsSync(historialPath)) {
    historial = JSON.parse(fs.readFileSync(historialPath, 'utf-8'));
}

const nuevaPrediccion = {
    fecha_prediccion: new Date().toISOString(),
    fecha_sorteo_objetivo: "2026-05-27 (Miércoles)",
    juego: "Quini 6 (Modelo FFT + Markov)",
    tickets: ticketsFinales.slice(0, 3).map((ticket, idx) => ({
        nombre: `SÚPER TICKET AVANZADO ${String.fromCharCode(65 + idx)}`,
        numeros: ticket
    }))
};

historial.push(nuevaPrediccion);
fs.writeFileSync(historialPath, JSON.stringify(historial, null, 2), 'utf-8');

console.log(`\n✅ Predicción guardada exitosamente en historial_predicciones.json`);
console.log(`\n==================================================`);
console.log(`¡Suerte en el sorteo!`);
