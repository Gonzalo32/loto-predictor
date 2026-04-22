class LotoPredictor {
    constructor(datosHistoricos) {
        this.datos = datosHistoricos;
    }

    _extraerBolillas(fila) {
        return [
            parseInt(fila['B1'], 10), parseInt(fila['B2'], 10), parseInt(fila['B3'], 10),
            parseInt(fila['B4'], 10), parseInt(fila['B5'], 10), parseInt(fila['B6'], 10)
        ];
    }

    analizarFrecuencias() {
        const contadorBolillas = {};
        const contadorPlus = {};

        for (const fila of this.datos) {
            const bolillas = this._extraerBolillas(fila);

            for (const b of bolillas) {
                contadorBolillas[b] = (contadorBolillas[b] || 0) + 1;
            }

            if (fila['Plus'] !== undefined) {
                const p = parseInt(fila['Plus'], 10);
                contadorPlus[p] = (contadorPlus[p] || 0) + 1;
            }
        }
        return { contadorBolillas, contadorPlus };
    }

    sugerirCalientes() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const { contadorBolillas, contadorPlus } = this.analizarFrecuencias();

        const ordenadosBolillas = Object.entries(contadorBolillas)
            .sort((a, b) => b[1] - a[1]);
        
        const masComunes = ordenadosBolillas.slice(0, 6)
            .map(x => parseInt(x[0], 10))
            .sort((a, b) => a - b);

        const ordenadosPlus = Object.entries(contadorPlus).sort((a, b) => b[1] - a[1]);
        const plusComun = ordenadosPlus.length > 0 ? parseInt(ordenadosPlus[0][0], 10) : Math.floor(Math.random() * 10);

        return { numeros: masComunes, plus: plusComun };
    }

    sugerirFrios() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const { contadorBolillas, contadorPlus } = this.analizarFrecuencias();

        const bolillasTodas = {};
        for (let i = 0; i <= 45; i++) {
            bolillasTodas[i] = contadorBolillas[i] || 0;
        }

        const ordenadasFrias = Object.entries(bolillasTodas)
            .sort((a, b) => a[1] - b[1]);

        const masFrias = ordenadasFrias.slice(0, 6)
            .map(x => parseInt(x[0], 10))
            .sort((a, b) => a - b);

        const plusTodas = {};
        for(let i = 0; i <= 9; i++) {
            plusTodas[i] = contadorPlus[i] || 0;
        }
        const ordenadasPlusFrias = Object.entries(plusTodas).sort((a, b) => a[1] - b[1]);
        const plusFrio = parseInt(ordenadasPlusFrias[0][0], 10);

        return { numeros: masFrias, plus: plusFrio };
    }

    sugerirMixtoBalanceado() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };

        const calientes = this.sugerirCalientes().numeros;
        const frios = this.sugerirFrios().numeros;

        const seleccion = new Set();
        
        // Random 3 hot
        const calientesDisponibles = [...calientes];
        for(let i = 0; i < 3 && calientesDisponibles.length > 0; i++) {
            const idx = Math.floor(Math.random() * calientesDisponibles.length);
            seleccion.add(calientesDisponibles.splice(idx, 1)[0]);
        }

        // Random 2 cold
        const friosDisponibles = frios.filter(x => !seleccion.has(x));
        for(let i = 0; i < 2 && friosDisponibles.length > 0; i++) {
            const idx = Math.floor(Math.random() * friosDisponibles.length);
            seleccion.add(friosDisponibles.splice(idx, 1)[0]);
        }

        // Fill with random until 6
        while (seleccion.size < 6) {
            seleccion.add(Math.floor(Math.random() * 46));
        }

        const mixto = Array.from(seleccion).sort((a, b) => a - b);
        const plusAleatorio = Math.floor(Math.random() * 10);

        return { numeros: mixto, plus: plusAleatorio };
    }

    sugerirPorMarkov(ultimasBolillas) {
        if (!this.datos || this.datos.length < 2) return { numeros: [], plus: 0 };
        const transiciones = {};

        for (let i = 0; i < this.datos.length - 1; i++) {
            const sorteoActual = this._extraerBolillas(this.datos[i]);
            const sorteoSiguiente = this._extraerBolillas(this.datos[i+1]);

            for (const bActual of sorteoActual) {
                if (!transiciones[bActual]) transiciones[bActual] = {};
                for (const bSiguiente of sorteoSiguiente) {
                    transiciones[bActual][bSiguiente] = (transiciones[bActual][bSiguiente] || 0) + 1;
                }
            }
        }

        const puntajes = {};
        for (const b of ultimasBolillas) {
            if (transiciones[b]) {
                for (const [sig, count] of Object.entries(transiciones[b])) {
                    puntajes[sig] = (puntajes[sig] || 0) + count;
                }
            }
        }

        const ordenados = Object.entries(puntajes)
            .sort((a, b) => b[1] - a[1])
            .map(x => parseInt(x[0], 10));

        let seleccion = [];
        for (const num of ordenados) {
            if (seleccion.length < 6 && !seleccion.includes(num)) {
                seleccion.push(num);
            }
        }

        while (seleccion.length < 6) {
            let r = Math.floor(Math.random() * 46);
            if (!seleccion.includes(r)) seleccion.push(r);
        }

        return { numeros: seleccion.sort((a, b) => a - b), plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorDelta() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const deltasFreq = {};
        const primerBolillaFreq = {};

        for (const fila of this.datos) {
            const b = this._extraerBolillas(fila).sort((a,b) => a-b);
            primerBolillaFreq[b[0]] = (primerBolillaFreq[b[0]] || 0) + 1;
            for (let i = 1; i < b.length; i++) {
                const delta = b[i] - b[i-1];
                deltasFreq[delta] = (deltasFreq[delta] || 0) + 1;
            }
        }
        
        const deltasComunes = Object.entries(deltasFreq)
            .sort((a, b) => b[1] - a[1])
            .map(x => parseInt(x[0], 10));

        const primerBolilla = parseInt(Object.entries(primerBolillaFreq)
            .sort((a, b) => b[1] - a[1])[0][0], 10);

        let seleccion = [primerBolilla];
        for (let i = 0; i < 5; i++) {
            let added = false;
            for (const d of deltasComunes) {
                let candidato = seleccion[seleccion.length - 1] + d;
                if (candidato <= 45 && !seleccion.includes(candidato)) {
                    seleccion.push(candidato);
                    added = true;
                    break;
                }
            }
            if (!added) {
                 let r;
                 do { r = Math.floor(Math.random() * 46); } while(seleccion.includes(r));
                 seleccion.push(r);
                 seleccion.sort((a,b) => a-b);
            }
        }
        return { numeros: seleccion.sort((a, b) => a - b), plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorBalancePares() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const balances = {}; 
        for (const fila of this.datos) {
            const b = this._extraerBolillas(fila);
            const pares = b.filter(x => x % 2 === 0).length;
            const impares = 6 - pares;
            const key = `${pares}_${impares}`;
            balances[key] = (balances[key] || 0) + 1;
        }
        
        const balanceMasComun = Object.entries(balances).sort((a, b) => b[1] - a[1])[0][0];
        const [objPares, objImpares] = balanceMasComun.split('_').map(Number);

        const { contadorBolillas } = this.analizarFrecuencias();
        const ordenados = Object.entries(contadorBolillas).sort((a, b) => b[1] - a[1]).map(x => parseInt(x[0], 10));
        
        let seleccion = [];
        let pCount = 0;
        let iCount = 0;

        for (const num of ordenados) {
            if (num % 2 === 0 && pCount < objPares) {
                seleccion.push(num);
                pCount++;
            } else if (num % 2 !== 0 && iCount < objImpares) {
                seleccion.push(num);
                iCount++;
            }
            if (seleccion.length === 6) break;
        }

        while (seleccion.length < 6) {
            let r = Math.floor(Math.random() * 46);
            if (!seleccion.includes(r)) {
                if (r % 2 === 0 && pCount < objPares) { seleccion.push(r); pCount++; }
                else if (r % 2 !== 0 && iCount < objImpares) { seleccion.push(r); iCount++; }
            }
        }
        return { numeros: seleccion.sort((a, b) => a - b), plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorSumaCampanaGauss() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const sumas = [];
        for (const fila of this.datos) {
            const b = this._extraerBolillas(fila);
            sumas.push(b.reduce((a, c) => a + c, 0));
        }
        sumas.sort((a,b) => a-b);
        const median = sumas[Math.floor(sumas.length / 2)];
        const minTarget = median - 15;
        const maxTarget = median + 15;

        let seleccion = [];
        let intentos = 0;
        while (intentos < 1000) {
            const setNum = new Set();
            while(setNum.size < 6) setNum.add(Math.floor(Math.random() * 46));
            const arr = Array.from(setNum);
            const s = arr.reduce((a,c) => a+c, 0);
            if (s >= minTarget && s <= maxTarget) {
                seleccion = arr;
                break;
            }
            intentos++;
        }
        if (seleccion.length === 0) seleccion = [1,2,3,4,5,6]; // fallback
        return { numeros: seleccion.sort((a, b) => a - b), plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorDecenas() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const patrones = {};
        for (const fila of this.datos) {
            const b = this._extraerBolillas(fila);
            const counts = [0,0,0,0,0]; // 0-9, 10-19, 20-29, 30-39, 40-45
            for (const n of b) {
                if(n < 10) counts[0]++;
                else if(n < 20) counts[1]++;
                else if(n < 30) counts[2]++;
                else if(n < 40) counts[3]++;
                else counts[4]++;
            }
            const key = counts.join('-');
            patrones[key] = (patrones[key] || 0) + 1;
        }
        const mejorPatron = Object.entries(patrones).sort((a,b) => b[1] - a[1])[0][0].split('-').map(Number);
        
        let seleccion = [];
        for (let i = 0; i < 5; i++) {
            let cantidad = mejorPatron[i];
            let agregados = 0;
            while(agregados < cantidad) {
                let min = i * 10;
                let max = i === 4 ? 45 : (i * 10) + 9;
                let r = Math.floor(Math.random() * (max - min + 1)) + min;
                if (!seleccion.includes(r)) {
                    seleccion.push(r);
                    agregados++;
                }
            }
        }
        return { numeros: seleccion.sort((a, b) => a - b), plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorSimilitud(ultimasBolillas) {
        if (!this.datos || this.datos.length < 2) return { numeros: [], plus: 0 };
        
        let mejorDistancia = Infinity;
        let indiceSiguiente = -1;

        const calcularDistancia = (arr1, arr2) => {
            let d = 0;
            for(let i=0; i<6; i++) d += Math.pow(arr1[i] - arr2[i], 2);
            return Math.sqrt(d);
        };

        for (let i = 0; i < this.datos.length - 1; i++) {
            const bHist = this._extraerBolillas(this.datos[i]);
            const dist = calcularDistancia(ultimasBolillas, bHist);
            if (dist < mejorDistancia) {
                mejorDistancia = dist;
                indiceSiguiente = i + 1;
            }
        }

        if (indiceSiguiente !== -1) {
            const nums = this._extraerBolillas(this.datos[indiceSiguiente]);
            return { numeros: nums, plus: Math.floor(Math.random() * 10) };
        }
        return this.sugerirMixtoBalanceado(); 
    }

    sugerirPorAtraso() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const atrasos = {};
        for(let i=0; i<=45; i++) atrasos[i] = this.datos.length;

        for (let i = this.datos.length - 1; i >= 0; i--) {
            const b = this._extraerBolillas(this.datos[i]);
            for (const num of b) {
                const distance = this.datos.length - 1 - i;
                if (distance < atrasos[num]) {
                    atrasos[num] = distance;
                }
            }
        }
        
        const ordenados = Object.entries(atrasos)
            .sort((a,b) => b[1] - a[1])
            .map(x => parseInt(x[0], 10));

        const seleccion = ordenados.slice(0, 6).sort((a,b) => a-b);
        return { numeros: seleccion, plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorConsecutivos() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const patrones = {};
        for (const fila of this.datos) {
            const b = this._extraerBolillas(fila).sort((a,b) => a-b);
            let consCount = 0;
            for(let i=1; i<b.length; i++) {
                if (b[i] === b[i-1] + 1) consCount++;
            }
            patrones[consCount] = (patrones[consCount] || 0) + 1;
        }
        const targetCons = parseInt(Object.entries(patrones).sort((a,b) => b[1] - a[1])[0][0], 10);

        let seleccion = [];
        let intentos = 0;
        while (intentos < 1000) {
            const s = new Set();
            while(s.size < 6) s.add(Math.floor(Math.random() * 46));
            const arr = Array.from(s).sort((a,b) => a-b);
            let cons = 0;
            for(let i=1; i<arr.length; i++) {
                if (arr[i] === arr[i-1] + 1) cons++;
            }
            if (cons === targetCons) {
                seleccion = arr;
                break;
            }
            intentos++;
        }
        if (seleccion.length === 0) seleccion = [1,2,3,4,5,6];
        return { numeros: seleccion, plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorTendenciaLineal() {
        if (!this.datos || this.datos.length < 5) return this.sugerirMixtoBalanceado();
        const windowSize = Math.min(10, this.datos.length);
        const windowData = this.datos.slice(this.datos.length - windowSize);

        const posSeries = [[], [], [], [], [], []];
        for (const fila of windowData) {
            const b = this._extraerBolillas(fila).sort((a,b) => a-b);
            for(let i=0; i<6; i++) posSeries[i].push(b[i]);
        }

        const sumX = (windowSize - 1) * windowSize / 2;
        const meanX = sumX / windowSize;
        const sumX2 = windowSize * (windowSize - 1) * (2 * windowSize - 1) / 6;

        let seleccion = new Set();
        for (let i = 0; i < 6; i++) {
            const series = posSeries[i];
            const meanY = series.reduce((a,c) => a+c, 0) / windowSize;
            let num = 0;
            for(let x=0; x<windowSize; x++) {
                num += (x - meanX) * (series[x] - meanY);
            }
            const den = sumX2 - windowSize * meanX * meanX;
            const m = den === 0 ? 0 : num / den;
            const b = meanY - m * meanX;
            
            let prediccion = Math.round(m * windowSize + b);
            prediccion = Math.max(0, Math.min(45, prediccion));
            
            while (seleccion.has(prediccion)) {
                prediccion = (prediccion + 1) % 46;
            }
            seleccion.add(prediccion);
        }

        return { numeros: Array.from(seleccion).sort((a, b) => a - b), plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorNumerosPrimos() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const isPrime = n => {
            if (n < 2) return false;
            for(let i=2; i<=Math.sqrt(n); i++) if(n % i === 0) return false;
            return true;
        };
        const conteos = {};
        for (const fila of this.datos) {
            const b = this._extraerBolillas(fila);
            const primes = b.filter(isPrime).length;
            conteos[primes] = (conteos[primes] || 0) + 1;
        }
        const targetPrimes = parseInt(Object.entries(conteos).sort((a,b) => b[1] - a[1])[0][0], 10);

        let seleccion = [];
        let intentos = 0;
        while(intentos < 1000) {
            const s = new Set();
            while(s.size < 6) s.add(Math.floor(Math.random() * 46));
            const arr = Array.from(s).sort((a,b) => a-b);
            if (arr.filter(isPrime).length === targetPrimes) {
                seleccion = arr; break;
            }
            intentos++;
        }
        if(seleccion.length===0) seleccion = [1,2,3,4,5,6];
        return { numeros: seleccion, plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorTerminaciones() {
        let seleccion = [];
        let intentos = 0;
        while(intentos < 1000) {
            const s = new Set();
            while(s.size < 6) s.add(Math.floor(Math.random() * 46));
            const arr = Array.from(s).sort((a,b) => a-b);
            const terms = {};
            for(const n of arr) {
                const term = n % 10;
                terms[term] = (terms[term] || 0) + 1;
            }
            const hasClump = Object.values(terms).some(v => v > 2);
            if (!hasClump) {
                seleccion = arr; break;
            }
            intentos++;
        }
        if(seleccion.length===0) seleccion = [1,2,3,4,5,6];
        return { numeros: seleccion, plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorMitades() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const balances = {}; 
        for (const fila of this.datos) {
            const b = this._extraerBolillas(fila);
            const bajas = b.filter(x => x <= 22).length;
            const altas = 6 - bajas;
            const key = `${bajas}_${altas}`;
            balances[key] = (balances[key] || 0) + 1;
        }
        const balanceMasComun = Object.entries(balances).sort((a, b) => b[1] - a[1])[0][0];
        const [objBajas, objAltas] = balanceMasComun.split('_').map(Number);
        
        let seleccion = [];
        let intentos = 0;
        while(intentos < 1000) {
            const s = new Set();
            while(s.size < 6) s.add(Math.floor(Math.random() * 46));
            const arr = Array.from(s).sort((a,b) => a-b);
            const bLen = arr.filter(x => x <= 22).length;
            if (bLen === objBajas) {
                seleccion = arr; break;
            }
            intentos++;
        }
        if(seleccion.length===0) seleccion = [1,2,3,4,5,6];
        return { numeros: seleccion, plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorRangoTotal() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const rangos = [];
        for (const fila of this.datos) {
            const b = this._extraerBolillas(fila).sort((a,b) => a-b);
            rangos.push(b[5] - b[0]);
        }
        rangos.sort((a,b) => a-b);
        const medRango = rangos[Math.floor(rangos.length / 2)];
        
        let seleccion = [];
        let intentos = 0;
        while(intentos < 1000) {
            const s = new Set();
            while(s.size < 6) s.add(Math.floor(Math.random() * 46));
            const arr = Array.from(s).sort((a,b) => a-b);
            const r = arr[5] - arr[0];
            if (Math.abs(r - medRango) <= 2) {
                seleccion = arr; break;
            }
            intentos++;
        }
        if(seleccion.length===0) seleccion = [1,2,3,4,5,6];
        return { numeros: seleccion, plus: Math.floor(Math.random() * 10) };
    }

    sugerirPorSumaDigitos() {
        if (!this.datos || this.datos.length === 0) return { numeros: [], plus: 0 };
        const conteos = {};
        const sumDig = n => {
            let s = 0;
            while(n > 0) { s += n % 10; n = Math.floor(n / 10); }
            return s;
        };
        for (const fila of this.datos) {
            const b = this._extraerBolillas(fila);
            for(const n of b) {
                const s = sumDig(n);
                conteos[s] = (conteos[s] || 0) + 1;
            }
        }
        const favRoots = Object.entries(conteos).sort((a,b) => b[1] - a[1]).slice(0, 4).map(x => parseInt(x[0], 10));

        let seleccion = [];
        let intentos = 0;
        while(intentos < 1000) {
            const s = new Set();
            while(s.size < 6) s.add(Math.floor(Math.random() * 46));
            const arr = Array.from(s).sort((a,b) => a-b);
            let matches = 0;
            for(const n of arr) if(favRoots.includes(sumDig(n))) matches++;
            if (matches >= 4) {
                seleccion = arr; break;
            }
            intentos++;
        }
        if(seleccion.length===0) seleccion = [1,2,3,4,5,6];
        return { numeros: seleccion, plus: Math.floor(Math.random() * 10) };
    }

    evaluarEstrategia(nombreMetodo) {
        if (this.datos.length < 5) return { promedio: 0, detalles: [] };
        let totalAciertos = 0;
        let sorteosEvaluados = 0;
        const distribucion = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };

        for (let i = 2; i < this.datos.length; i++) {
            const trainData = this.datos.slice(0, i);
            const testBolillas = this._extraerBolillas(this.datos[i]);

            const predictorTemp = new LotoPredictor(trainData);
            let prediccion;

            if (nombreMetodo === 'sugerirPorMarkov' || nombreMetodo === 'sugerirPorSimilitud') {
                const ul = this._extraerBolillas(trainData[trainData.length - 1]);
                prediccion = predictorTemp[nombreMetodo](ul).numeros;
            } else {
                prediccion = predictorTemp[nombreMetodo]().numeros;
            }

            let aciertos = 0;
            for (const p of prediccion) {
                if (testBolillas.includes(p)) aciertos++;
            }
            totalAciertos += aciertos;
            distribucion[aciertos]++;
            sorteosEvaluados++;
        }

        return { 
            promedio: totalAciertos / sorteosEvaluados,
            evaluados: sorteosEvaluados,
            distribucion
        };
    }
}

module.exports = LotoPredictor;

