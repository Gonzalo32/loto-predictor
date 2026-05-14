const LotoScraper = require('./scraper');
const LotoPredictor = require('./predictor');

function evaluate(filename, name) {
    const scraper = new LotoScraper(filename);
    const datos = scraper.collectData();
    if (datos.length === 0) return;
    const predictor = new LotoPredictor(datos);

    const est = 'sugerirPorSecuenciaCondicional';
    const stats = predictor.evaluarEstrategia(est);
    const hits4 = stats.distribucion[4] || 0;
    const hits5 = stats.distribucion[5] || 0;
    const hits6 = stats.distribucion[6] || 0;
    const total4Plus = hits4 + hits5 + hits6;
    
    console.log(`--- Análisis ${name} (Nuevo Algoritmo) ---`);
    console.log(`Estrategia: ${est}`);
    console.log(`Promedio aciertos: ${stats.promedio.toFixed(4)}`);
    console.log(`Distribución:`, stats.distribucion);
    console.log(`Total 4+ aciertos: ${total4Plus}`);
}

evaluate('historico_loto.csv', 'Loto Plus');
evaluate('historico_quini_limpio.csv', 'Quini 6');
