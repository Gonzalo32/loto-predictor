const LotoScraper = require('./scraper');
const LotoPredictor = require('./predictor');

function generateTop3(filename, name) {
    const scraper = new LotoScraper(filename);
    const datos = scraper.collectData();
    const predictor = new LotoPredictor(datos);
    
    const ul = predictor._extraerBolillas(datos[datos.length - 1]);

    const t1 = predictor.sugerirPorEnsembleVotado();
    const t2 = predictor.sugerirPorSecuenciaCondicional();
    
    // For the 3rd one, let's use Markov or Similitud depending on the game
    let t3;
    if (name === 'Quini 6') {
        t3 = predictor.sugerirPorSimilitud(ul);
    } else {
        t3 = predictor.sugerirPorMarkov(ul);
    }

    console.log(`\n--- ${name} ---`);
    console.log(`Ticket 1 (Ensemble): [ ${t1.numeros.map(n => String(n).padStart(2, '0')).join(' ')} ]` + (t1.plus !== undefined ? ` | Plus: ${t1.plus}` : ''));
    console.log(`Ticket 2 (Secuencial): [ ${t2.numeros.map(n => String(n).padStart(2, '0')).join(' ')} ]` + (t2.plus !== undefined ? ` | Plus: ${t2.plus}` : ''));
    console.log(`Ticket 3 (Patrón): [ ${t3.numeros.map(n => String(n).padStart(2, '0')).join(' ')} ]` + (t3.plus !== undefined ? ` | Plus: ${t3.plus}` : ''));
}

generateTop3('historico_loto.csv', 'Loto Plus');
generateTop3('historico_quini_limpio.csv', 'Quini 6');
