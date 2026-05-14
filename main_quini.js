const LotoScraper = require('./scraper');
const LotoPredictor = require('./predictor');
const fs = require('fs');

async function main() {
    console.log("==================================================");
    console.log("🎰 PREDICTOR AUTOMÁTICO - QUINI 6 ARGENTINA 🇦🇷");
    console.log("==================================================");

    // 1. Cargar Datos del Quini 6
    const filename = 'historico_quini_limpio.csv';
    const scraper = new LotoScraper(filename);
    const datos = scraper.collectData();

    if (datos.length === 0) {
        console.error("No se pudieron cargar los datos del Quini 6.");
        return;
    }

    const predictor = new LotoPredictor(datos);

    // 2. Backtesting Progresivo
    console.log("\n🧠 Procesando Inteligencia de Datos y calculando patrones...");
    console.log("🧪 Realizando Backtesting Progresivo (Simulación Paso a Paso)...");

    const estrategias = [
        { nombre: 'Cadenas de Markov', metodo: 'sugerirPorMarkov' },
        { nombre: 'Sistema Delta', metodo: 'sugerirPorDelta' },
        { nombre: 'Balance Par/Impar', metodo: 'sugerirPorBalancePares' },
        { nombre: 'Suma de Gauss', metodo: 'sugerirPorSumaCampanaGauss' },
        { nombre: 'Patrón de Decenas', metodo: 'sugerirPorDecenas' },
        { nombre: 'KNN (Similitud)', metodo: 'sugerirPorSimilitud' },
        { nombre: 'Ciclo de Atrasos', metodo: 'sugerirPorAtraso' },
        { nombre: 'Patrón Consecutivos', metodo: 'sugerirPorConsecutivos' },
        { nombre: 'Regresión Lineal', metodo: 'sugerirPorTendenciaLineal' },
        { nombre: 'Números Primos', metodo: 'sugerirPorNumerosPrimos' },
        { nombre: 'Terminaciones', metodo: 'sugerirPorTerminaciones' },
        { nombre: 'Mitades (Alta/Baja)', metodo: 'sugerirPorMitades' },
        { nombre: 'Rango Total', metodo: 'sugerirPorRangoTotal' },
        { nombre: 'Suma de Dígitos', metodo: 'sugerirPorSumaDigitos' }
    ];

    const resultados = [];
    for (const est of estrategias) {
        const stats = predictor.evaluarEstrategia(est.metodo);
        resultados.push({ ...est, puntaje: stats.promedio, stats });
        console.log(`\n📊 ${est.nombre} (sobre ${stats.total} sorteos simulados):`);
        console.log(`   - Aciertos Promedio: ${stats.promedio.toFixed(4)} por ticket`);
        console.log(`   - Distribución: [0: ${stats.distribucion[0]}] [1: ${stats.distribucion[1]}] [2: ${stats.distribucion[2]}] [3+: ${stats.distribucion[3]}]`);
    }

    resultados.sort((a, b) => b.puntaje - a.puntaje);
    const mejorEstrategia = resultados[0];

    console.log(`\n🏆 Mejor estrategia histórica para Quini 6: ${mejorEstrategia.nombre} (${mejorEstrategia.puntaje.toFixed(2)} aciertos prom.)\n`);

    // 3. Generar Predicciones
    console.log("==================================================");
    console.log("🔮 PREDICCIONES PARA EL PRÓXIMO SORTEO QUINI 6");
    console.log("==================================================\n");

    const ultimasBolillas = [
        parseInt(datos[datos.length-1]['B1']), parseInt(datos[datos.length-1]['B2']),
        parseInt(datos[datos.length-1]['B3']), parseInt(datos[datos.length-1]['B4']),
        parseInt(datos[datos.length-1]['B5']), parseInt(datos[datos.length-1]['B6'])
    ];

    const imprimirTicket = (nombre, num, plus) => {
        const nStr = num.map(n => String(n).padStart(2, '0')).join('  ');
        console.log(`--------------------------------------------------`);
        console.log(`🌟 ${nombre}`);
        console.log(`--------------------------------------------------`);
        console.log(`🎱 BOLIILAS: [ ${nStr} ]`);
        console.log(`--------------------------------------------------\n`);
    };

    // Motor Montecarlo para Quini 6 (Sin Plus)
    console.log("\n⏳ Iniciando motor de Montecarlo (10.000 iteraciones)...");
    const freqNumeros = {};
    const ITERACIONES_MONTECARLO = 10000;

    for (let i = 0; i < ITERACIONES_MONTECARLO; i++) {
        const batch = [
            predictor.sugerirCalientes(),
            predictor.sugerirFrios(),
            predictor.sugerirMixtoBalanceado(),
            predictor.sugerirPorMarkov(ultimasBolillas),
            predictor.sugerirPorDelta(),
            predictor.sugerirPorBalancePares(),
            predictor.sugerirPorSumaCampanaGauss(),
            predictor.sugerirPorDecenas(),
            predictor.sugerirPorSimilitud(ultimasBolillas),
            predictor.sugerirPorAtraso(),
            predictor.sugerirPorConsecutivos(),
            predictor.sugerirPorTendenciaLineal(),
            predictor.sugerirPorNumerosPrimos(),
            predictor.sugerirPorTerminaciones(),
            predictor.sugerirPorMitades(),
            predictor.sugerirPorRangoTotal(),
            predictor.sugerirPorSumaDigitos()
        ];
        
        batch.forEach(obj => {
            obj.numeros.forEach(n => freqNumeros[n] = (freqNumeros[n] || 0) + 1);
        });
    }

    const topNumeros = Object.entries(freqNumeros)
        .sort((a,b) => b[1] - a[1])
        .map(x => parseInt(x[0], 10));

    const t1 = [topNumeros[0], topNumeros[3], topNumeros[6], topNumeros[9], topNumeros[12], topNumeros[15]].sort((a,b)=>a-b);
    const t2 = [topNumeros[1], topNumeros[4], topNumeros[7], topNumeros[10], topNumeros[13], topNumeros[16]].sort((a,b)=>a-b);
    const t3 = [topNumeros[2], topNumeros[5], topNumeros[8], topNumeros[11], topNumeros[14], topNumeros[17]].sort((a,b)=>a-b);

    console.log("\n==================================================");
    console.log("🤖 CONSENSO MONTECARLO QUINI 6 (SÚPER TICKETS)");
    console.log("==================================================");
    
    imprimirTicket("SÚPER TICKET A (Arquetipo Quini)", t1);
    imprimirTicket("SÚPER TICKET B (Alta Probabilidad)", t2);
    imprimirTicket("SÚPER TICKET C (Cobertura Estratégica)", t3);

    console.log("⚠️ Recuerda: El Quini 6 es un juego de azar matemáticamente independiente.");
}

main().catch(console.error);
