const LotoScraper = require('./scraper');
const LotoPredictor = require('./predictor');

function imprimirTicket(titulo, numeros, plus) {
    console.log("--------------------------------------------------");
    console.log(`🌟 ${titulo}`);
    console.log("--------------------------------------------------");
    const pad = (n) => String(n).padStart(2, '0');
    console.log(`🎱 BOLIILAS: [ ${numeros.map(pad).join("  ")} ]`);
    console.log(`⭐ PLUS:     [ ${pad(plus)} ]`);
    console.log("--------------------------------------------------\n");
}

async function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
    console.log("==================================================");
    console.log("🎰 PREDICTOR AUTOMÁTICO - LOTO PLUS ARGENTINA 🇦🇷");
    console.log("==================================================");
    
    // 1. Recolectar datos
    const scraper = new LotoScraper();
    const datos = scraper.collectData();
    
    if (!datos || datos.length === 0) {
        console.log("❌ Error: No hay suficientes datos para iniciar las predicciones.");
        return;
    }

    console.log("🔄 Simulando actualización web de la última jugada...");
    await sleep(1500);
    const ultimoSorteo = parseInt(datos[datos.length - 1]['Sorteo'], 10);
    const nuevo = scraper.mockFetchNewDraw(ultimoSorteo);
    datos.push(nuevo);
    console.log(`✅ Nuevo sorteo detectado e incorporado! Sorteo: #${nuevo['Sorteo']}\n`);

    // 2. Inicializar Predictor
    console.log("🧠 Procesando Inteligencia de Datos y calculando patrones...");
    await sleep(1000);
    
    const predictor = new LotoPredictor(datos);

    // 3. Backtesting de Estrategias Avanzadas
    console.log("🧪 Realizando Backtesting Progresivo (Simulación Paso a Paso)...");
    console.log("   (El algoritmo toma los N sorteos anteriores para predecir el sorteo N+1, hasta el final)");
    
    const statsMarkov = predictor.evaluarEstrategia('sugerirPorMarkov');
    const statsDelta = predictor.evaluarEstrategia('sugerirPorDelta');
    const statsPares = predictor.evaluarEstrategia('sugerirPorBalancePares');
    const statsSuma = predictor.evaluarEstrategia('sugerirPorSumaCampanaGauss');
    const statsDecenas = predictor.evaluarEstrategia('sugerirPorDecenas');
    const statsSimilitud = predictor.evaluarEstrategia('sugerirPorSimilitud');
    const statsAtraso = predictor.evaluarEstrategia('sugerirPorAtraso');
    const statsConsecutivos = predictor.evaluarEstrategia('sugerirPorConsecutivos');
    const statsTendencia = predictor.evaluarEstrategia('sugerirPorTendenciaLineal');
    const statsPrimos = predictor.evaluarEstrategia('sugerirPorNumerosPrimos');
    const statsTerminaciones = predictor.evaluarEstrategia('sugerirPorTerminaciones');
    const statsMitades = predictor.evaluarEstrategia('sugerirPorMitades');
    const statsRango = predictor.evaluarEstrategia('sugerirPorRangoTotal');
    const statsSumaDigitos = predictor.evaluarEstrategia('sugerirPorSumaDigitos');

    const printStats = (nombre, stats) => {
        console.log(`\n📊 ${nombre} (sobre ${stats.evaluados} sorteos simulados):`);
        console.log(`   - Aciertos Promedio: ${stats.promedio.toFixed(4)} por ticket`);
        console.log(`   - Distribución: [0: ${stats.distribucion[0]}] [1: ${stats.distribucion[1]}] [2: ${stats.distribucion[2]}] [3+: ${stats.distribucion[3] + stats.distribucion[4] + stats.distribucion[5] + stats.distribucion[6]}]`);
    };

    printStats('Cadenas de Markov', statsMarkov);
    printStats('Sistema Delta', statsDelta);
    printStats('Balance Par/Impar', statsPares);
    printStats('Suma de Gauss', statsSuma);
    printStats('Patrón de Decenas', statsDecenas);
    printStats('KNN (Similitud)', statsSimilitud);
    printStats('Ciclo de Atrasos', statsAtraso);
    printStats('Patrón Consecutivos', statsConsecutivos);
    printStats('Regresión Lineal', statsTendencia);
    printStats('Números Primos', statsPrimos);
    printStats('Terminaciones', statsTerminaciones);
    printStats('Mitades (Alta/Baja)', statsMitades);
    printStats('Rango Total', statsRango);
    printStats('Suma de Dígitos', statsSumaDigitos);

    const resultados = [
        { nombre: 'Cadenas de Markov', puntaje: statsMarkov.promedio, metodo: 'sugerirPorMarkov' },
        { nombre: 'Sistema Delta', puntaje: statsDelta.promedio, metodo: 'sugerirPorDelta' },
        { nombre: 'Balance Par/Impar', puntaje: statsPares.promedio, metodo: 'sugerirPorBalancePares' },
        { nombre: 'Suma de Gauss', puntaje: statsSuma.promedio, metodo: 'sugerirPorSumaCampanaGauss' },
        { nombre: 'Patrón de Decenas', puntaje: statsDecenas.promedio, metodo: 'sugerirPorDecenas' },
        { nombre: 'KNN (Similitud)', puntaje: statsSimilitud.promedio, metodo: 'sugerirPorSimilitud' },
        { nombre: 'Ciclo de Atrasos', puntaje: statsAtraso.promedio, metodo: 'sugerirPorAtraso' },
        { nombre: 'Patrón Consecutivos', puntaje: statsConsecutivos.promedio, metodo: 'sugerirPorConsecutivos' },
        { nombre: 'Regresión Lineal', puntaje: statsTendencia.promedio, metodo: 'sugerirPorTendenciaLineal' },
        { nombre: 'Números Primos', puntaje: statsPrimos.promedio, metodo: 'sugerirPorNumerosPrimos' },
        { nombre: 'Terminaciones', puntaje: statsTerminaciones.promedio, metodo: 'sugerirPorTerminaciones' },
        { nombre: 'Mitades (Alta/Baja)', puntaje: statsMitades.promedio, metodo: 'sugerirPorMitades' },
        { nombre: 'Rango Total', puntaje: statsRango.promedio, metodo: 'sugerirPorRangoTotal' },
        { nombre: 'Suma de Dígitos', puntaje: statsSumaDigitos.promedio, metodo: 'sugerirPorSumaDigitos' }
    ];
    resultados.sort((a, b) => b.puntaje - a.puntaje);
    const mejorEstrategia = resultados[0];

    console.log(`\n🏆 Mejor estrategia histórica: ${mejorEstrategia.nombre} (${mejorEstrategia.puntaje.toFixed(2)} aciertos prom.)\n`);

    // 4. Mostrar Sugerencias
    console.log("==================================================");
    console.log("🔮 PREDICCIONES PARA EL PRÓXIMO SORTEO");
    console.log("==================================================\n");

    const ultimasBolillas = [
        parseInt(datos[datos.length-1]['B1'], 10), parseInt(datos[datos.length-1]['B2'], 10),
        parseInt(datos[datos.length-1]['B3'], 10), parseInt(datos[datos.length-1]['B4'], 10),
        parseInt(datos[datos.length-1]['B5'], 10), parseInt(datos[datos.length-1]['B6'], 10)
    ];

    const calientesObj = predictor.sugerirCalientes();
    const friosObj = predictor.sugerirFrios();
    const mixtoObj = predictor.sugerirMixtoBalanceado();
    
    const markovObj = predictor.sugerirPorMarkov(ultimasBolillas);
    const deltaObj = predictor.sugerirPorDelta();
    const paresObj = predictor.sugerirPorBalancePares();
    const sumaObj = predictor.sugerirPorSumaCampanaGauss();
    const decenasObj = predictor.sugerirPorDecenas();
    const similitudObj = predictor.sugerirPorSimilitud(ultimasBolillas);
    const atrasoObj = predictor.sugerirPorAtraso();
    const consObj = predictor.sugerirPorConsecutivos();
    const tendObj = predictor.sugerirPorTendenciaLineal();
    const primosObj = predictor.sugerirPorNumerosPrimos();
    const terminacionesObj = predictor.sugerirPorTerminaciones();
    const mitadesObj = predictor.sugerirPorMitades();
    const rangoObj = predictor.sugerirPorRangoTotal();
    const sumadigObj = predictor.sugerirPorSumaDigitos();

    imprimirTicket("BÁSICO: MÁS CALIENTES", calientesObj.numeros, calientesObj.plus);
    imprimirTicket("BÁSICO: MÁS FRÍOS", friosObj.numeros, friosObj.plus);
    imprimirTicket("BÁSICO: MIXTO BALANCEADO", mixtoObj.numeros, mixtoObj.plus);
    
    imprimirTicket(`AVANZADO 1: MARKOV${mejorEstrategia.metodo === 'sugerirPorMarkov' ? ' 🏆' : ''}`, markovObj.numeros, markovObj.plus);
    imprimirTicket(`AVANZADO 2: DELTA${mejorEstrategia.metodo === 'sugerirPorDelta' ? ' 🏆' : ''}`, deltaObj.numeros, deltaObj.plus);
    imprimirTicket(`AVANZADO 3: PARES/IMPARES${mejorEstrategia.metodo === 'sugerirPorBalancePares' ? ' 🏆' : ''}`, paresObj.numeros, paresObj.plus);
    imprimirTicket(`AVANZADO 4: SUMA GAUSS${mejorEstrategia.metodo === 'sugerirPorSumaCampanaGauss' ? ' 🏆' : ''}`, sumaObj.numeros, sumaObj.plus);
    imprimirTicket(`AVANZADO 5: PATRÓN DECENAS${mejorEstrategia.metodo === 'sugerirPorDecenas' ? ' 🏆' : ''}`, decenasObj.numeros, decenasObj.plus);
    imprimirTicket(`AVANZADO 6: KNN SIMILITUD${mejorEstrategia.metodo === 'sugerirPorSimilitud' ? ' 🏆' : ''}`, similitudObj.numeros, similitudObj.plus);
    imprimirTicket(`AVANZADO 7: CICLO DE ATRASOS${mejorEstrategia.metodo === 'sugerirPorAtraso' ? ' 🏆' : ''}`, atrasoObj.numeros, atrasoObj.plus);
    imprimirTicket(`AVANZADO 8: CONSECUTIVOS${mejorEstrategia.metodo === 'sugerirPorConsecutivos' ? ' 🏆' : ''}`, consObj.numeros, consObj.plus);
    imprimirTicket(`AVANZADO 9: REGRESIÓN LINEAL${mejorEstrategia.metodo === 'sugerirPorTendenciaLineal' ? ' 🏆' : ''}`, tendObj.numeros, tendObj.plus);
    imprimirTicket(`AVANZADO 10: NÚMEROS PRIMOS${mejorEstrategia.metodo === 'sugerirPorNumerosPrimos' ? ' 🏆' : ''}`, primosObj.numeros, primosObj.plus);
    imprimirTicket(`AVANZADO 11: TERMINACIONES${mejorEstrategia.metodo === 'sugerirPorTerminaciones' ? ' 🏆' : ''}`, terminacionesObj.numeros, terminacionesObj.plus);
    imprimirTicket(`AVANZADO 12: MITADES (ALTA/BAJA)${mejorEstrategia.metodo === 'sugerirPorMitades' ? ' 🏆' : ''}`, mitadesObj.numeros, mitadesObj.plus);
    imprimirTicket(`AVANZADO 13: RANGO TOTAL${mejorEstrategia.metodo === 'sugerirPorRangoTotal' ? ' 🏆' : ''}`, rangoObj.numeros, rangoObj.plus);
    imprimirTicket(`AVANZADO 14: SUMA DE DÍGITOS${mejorEstrategia.metodo === 'sugerirPorSumaDigitos' ? ' 🏆' : ''}`, sumadigObj.numeros, sumadigObj.plus);
    console.log("\n⏳ Iniciando motor de Montecarlo (10.000 iteraciones)... Esto puede demorar unos segundos.");
    
    const freqNumeros = {};
    const freqPlus = {};
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
            freqPlus[obj.plus] = (freqPlus[obj.plus] || 0) + 1;
        });
    }

    const topNumeros = Object.entries(freqNumeros)
        .sort((a,b) => b[1] - a[1])
        .map(x => parseInt(x[0], 10));

    const topPlus = Object.entries(freqPlus)
        .sort((a,b) => b[1] - a[1])
        .map(x => parseInt(x[0], 10));

    // Tomamos los 18 números más votados y los cruzamos en 3 tickets
    const t1 = [topNumeros[0], topNumeros[3], topNumeros[6], topNumeros[9], topNumeros[12], topNumeros[15]].sort((a,b)=>a-b);
    const t2 = [topNumeros[1], topNumeros[4], topNumeros[7], topNumeros[10], topNumeros[13], topNumeros[16]].sort((a,b)=>a-b);
    const t3 = [topNumeros[2], topNumeros[5], topNumeros[8], topNumeros[11], topNumeros[14], topNumeros[17]].sort((a,b)=>a-b);

    console.log("\n==================================================");
    console.log("🤖 CONSENSO MONTECARLO (SÚPER TICKETS)");
    console.log("==================================================");
    console.log(`Combinando 17 algoritmos x ${ITERACIONES_MONTECARLO} iteraciones cruzadas:\n`);

    imprimirTicket("SÚPER TICKET A (Arquetipo Matemático Puro)", t1, topPlus[0] || 0);
    imprimirTicket("SÚPER TICKET B (Alta Probabilidad)", t2, topPlus[1] !== undefined ? topPlus[1] : (topPlus[0] || 1));
    imprimirTicket("SÚPER TICKET C (Cobertura Estratégica)", t3, topPlus[2] !== undefined ? topPlus[2] : (topPlus[0] || 2));

    console.log("⚠️ Recuerda: El Loto Plus es un juego de azar matemáticamente independiente.");
    console.log("El uso de este software es estadístico y lúdico. ¡No juegues más de lo que puedas!");
}

main().catch(console.error);
