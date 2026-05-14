const LotoScraper = require('./scraper');
const LotoPredictor = require('./predictor');

function evaluate(filename, name) {
    const scraper = new LotoScraper(filename);
    const datos = scraper.collectData();
    if (datos.length === 0) return;
    const predictor = new LotoPredictor(datos);

    const estrategias = [
        'sugerirPorMarkov', 'sugerirPorDelta', 'sugerirPorBalancePares', 
        'sugerirPorSumaCampanaGauss', 'sugerirPorDecenas', 'sugerirPorSimilitud', 
        'sugerirPorAtraso', 'sugerirPorConsecutivos', 'sugerirPorTendenciaLineal', 
        'sugerirPorNumerosPrimos', 'sugerirPorTerminaciones', 'sugerirPorMitades', 
        'sugerirPorRangoTotal', 'sugerirPorSumaDigitos'
    ];

    let best4Plus = 0;
    let bestStrategy = '';

    console.log(`--- Análisis ${name} ---`);
    for (const est of estrategias) {
        const stats = predictor.evaluarEstrategia(est);
        const hits4 = stats.distribucion[4] || 0;
        const hits5 = stats.distribucion[5] || 0;
        const hits6 = stats.distribucion[6] || 0;
        const total4Plus = hits4 + hits5 + hits6;
        
        if (total4Plus > best4Plus) {
            best4Plus = total4Plus;
            bestStrategy = est;
        }
    }
    console.log(`Mejor estrategia para 4+ aciertos: ${bestStrategy}`);
    console.log(`Total 4+ aciertos: ${best4Plus}`);
}

evaluate('historico_loto.csv', 'Loto Plus');
evaluate('historico_quini_limpio.csv', 'Quini 6');
