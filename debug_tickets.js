const LotoScraper = require('./scraper');
const LotoPredictor = require('./predictor');

function debug(filename, name) {
    const scraper = new LotoScraper(filename);
    const datos = scraper.collectData();
    const predictor = new LotoPredictor(datos);
    const ul = predictor._extraerBolillas(datos[datos.length - 1]);

    console.log(`\n--- Debug ${name} ---`);
    console.log("Running Ensemble...");
    const t1 = predictor.sugerirPorEnsembleVotado();
    console.log("Ensemble OK:", t1.numeros);

    console.log("Running Secuencial...");
    const t2 = predictor.sugerirPorSecuenciaCondicional();
    console.log("Secuencial OK:", t2.numeros);

    console.log("Running Patrón...");
    let t3;
    if (name === 'Quini 6') {
        t3 = predictor.sugerirPorSimilitud(ul);
    } else {
        t3 = predictor.sugerirPorMarkov(ul);
    }
    console.log("Patrón OK:", t3.numeros);
}

debug('historico_loto.csv', 'Loto Plus');
debug('historico_quini_limpio.csv', 'Quini 6');
