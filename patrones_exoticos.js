const fs = require('fs');

class ExploradorExotico {
    constructor(archivo, tipo) {
        this.tipo = tipo;
        this.datos = this._cargarDatos(archivo);
    }

    _cargarDatos(filename) {
        if (!fs.existsSync(filename)) return [];
        const content = fs.readFileSync(filename, 'utf-8').trim().split('\n');
        const headers = content[0].split(',');
        return content.slice(1).map(line => {
            const values = line.split(',');
            return {
                fecha: values[headers.indexOf('Fecha')],
                bolillas: [
                    parseInt(values[headers.indexOf('B1')]),
                    parseInt(values[headers.indexOf('B2')]),
                    parseInt(values[headers.indexOf('B3')]),
                    parseInt(values[headers.indexOf('B4')]),
                    parseInt(values[headers.indexOf('B5')]),
                    parseInt(values[headers.indexOf('B6')])
                ].sort((a,b)=>a-b)
            };
        });
    }

    analizarParesEspejo() {
        // Pares de números invertidos (ej. 12 y 21, 04 y 40)
        const paresEspejo = [
            [1, 10], [2, 20], [3, 30], [4, 40], 
            [12, 21], [13, 31], [14, 41], 
            [23, 32], [24, 42], [34, 43]
        ];
        
        let conteo = 0;
        let apariciones = {};

        for (const d of this.datos) {
            const b = d.bolillas;
            for (const par of paresEspejo) {
                if (b.includes(par[0]) && b.includes(par[1])) {
                    conteo++;
                    const key = `[${par[0].toString().padStart(2,'0')} - ${par[1]}]`;
                    apariciones[key] = (apariciones[key] || 0) + 1;
                }
            }
        }

        console.log(`\n🔍 PARES ESPEJO (ej. 12 y 21)`);
        console.log(`En ${this.datos.length} sorteos, salieron ambos números de un "Par Espejo" en el mismo sorteo ${conteo} veces.`);
        if (conteo > 0) {
            const top = Object.entries(apariciones).sort((a,b)=>b[1]-a[1]).slice(0,3);
            console.log(`Los pares espejo más frecuentes:`);
            top.forEach(x => console.log(`   ${x[0]} : ${x[1]} veces`));
        }
    }

    analizarEfectoCalendario() {
        let coincidencias = 0;
        for (const d of this.datos) {
            // Fecha formato YYYY-MM-DD o DD/MM/YYYY
            let dia = 0;
            if (d.fecha.includes('-')) {
                dia = parseInt(d.fecha.split('-')[2]);
            } else if (d.fecha.includes('/')) {
                dia = parseInt(d.fecha.split('/')[0]);
            }
            if (dia > 0 && dia <= 45 && d.bolillas.includes(dia)) {
                coincidencias++;
            }
        }
        const porc = ((coincidencias / this.datos.length) * 100).toFixed(2);
        console.log(`\n📅 EFECTO CALENDARIO (Día del sorteo)`);
        console.log(`¿Sale como ganadora la bolilla que coincide con el DÍA del mes del sorteo?`);
        console.log(`   Ocurrió en ${coincidencias} de ${this.datos.length} sorteos (${porc}% de las veces).`);
        if (porc > 13) {
            console.log(`   💡 DATO: Estadísticamente, la chance aleatoria es ~13%. Si este valor es mayor, ¡el calendario influye!`);
        }
    }

    analizarSaltosFrecuentes() {
        const saltos = {};
        for (const d of this.datos) {
            const b = d.bolillas;
            for (let i = 1; i < b.length; i++) {
                const salto = b[i] - b[i-1];
                saltos[salto] = (saltos[salto] || 0) + 1;
            }
        }
        
        const ordenados = Object.entries(saltos).sort((a,b)=>b[1]-a[1]);
        console.log(`\n📏 PATRÓN DE DISTANCIA (Gaps entre bolillas)`);
        console.log(`¿Qué distancia hay típicamente entre un número ganador y el siguiente?`);
        ordenados.slice(0, 3).forEach(([s, c], i) => {
            console.log(`   Top ${i+1}: Distancia de +${s} (se repitió ${c} veces en la historia)`);
        });
    }

    analizarZonasMuertas() {
        // Divide el tablero en zonas de 10 números
        const conteoZonasVacias = { '00-09': 0, '10-19': 0, '20-29': 0, '30-39': 0, '40-45': 0 };
        for (const d of this.datos) {
            const z = { '00-09': 0, '10-19': 0, '20-29': 0, '30-39': 0, '40-45': 0 };
            for(const n of d.bolillas) {
                if(n<10) z['00-09']++;
                else if(n<20) z['10-19']++;
                else if(n<30) z['20-29']++;
                else if(n<40) z['30-39']++;
                else z['40-45']++;
            }
            for (const [zona, count] of Object.entries(z)) {
                if (count === 0) conteoZonasVacias[zona]++;
            }
        }
        
        console.log(`\n🏴‍☠️ ZONAS MUERTAS (Decenas que no salen)`);
        console.log(`¿Qué tan probable es que un sorteo NO tenga ningún número de una decena específica?`);
        for (const [zona, count] of Object.entries(conteoZonasVacias)) {
            const porc = ((count / this.datos.length) * 100).toFixed(2);
            console.log(`   Zona ${zona}: Quedó vacía en el ${porc}% de los sorteos.`);
        }
    }

    ejecutar() {
        console.log(`\n==================================================`);
        console.log(`👽 PATRONES EXÓTICOS - ${this.tipo}`);
        console.log(`==================================================`);
        this.analizarParesEspejo();
        this.analizarEfectoCalendario();
        this.analizarSaltosFrecuentes();
        this.analizarZonasMuertas();
    }
}

const exLoto = new ExploradorExotico('historico_loto.csv', 'LOTO PLUS');
exLoto.ejecutar();

const exQuini = new ExploradorExotico('historico_quini_limpio.csv', 'QUINI 6');
exQuini.ejecutar();
