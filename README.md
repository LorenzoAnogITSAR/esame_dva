# Esame Finale: Strumenti di Prototipazione per AI & Data Vizualization 🎮

## 1. Contesto di business

Siete il team **Game Analytics** di un grande editore di videogiochi.

## 2. Dataset

Avete a disposizione uno **storico di videogiochi** (dal 2000 in poi) con:

- `Name` – Nome del videogioco.
- `Platform` – Piattaforma di uscita (PS, Xbox, Nintendo, PC, …).
- `Year_of_Release` – Anno di uscita del gioco.
- `Genre` – Genere principale (Action, Sports, Shooter, …).
- `Publisher` – Casa editrice / publisher del gioco.
- `NA_Sales` – Vendite in Nord America (milioni di copie).
- `EU_Sales` – Vendite in Europa (milioni di copie).
- `JP_Sales` – Vendite in Giappone (milioni di copie).
- `Other_Sales` – Vendite nel resto del mondo (milioni di copie).
- `Global_Sales` – Vendite globali totali (milioni di copie).
- `Critic_Score` – Valutazione media della critica (metascore).
- `Critic_Count` – Numero di recensioni della critica considerate.
- `User_Score` – Valutazione media degli utenti.
- `User_Count` – Numero di valutazioni utenti considerate.
- `Developer` – Studio di sviluppo del gioco.
- `Rating` – Classificazione ESRB (E, T, M, …).

Il **dataset pulito** è già fornito in:

```text
data/vgsales_clean.csv
```

## 3. Need
Il **management** vuole usare questi dati per:

1. Capire **su quali piattaforme e generi** ha funzionato meglio il catalogo nel tempo.  
2. Valutare quanto contano **recensioni e rating** sul successo commerciale di un gioco.  
3. Avere un **prototipo di modello ML** che aiuti a stimare la probabilità che un nuovo gioco diventi un **“HIT”** (es. ≥ 1M copie).  
4. (Nice to have) Un modo rapido per **interrogare i dati in linguaggio naturale**.

## 4. Obiettivo
Il vostro compito è costruire, in poco tempo, un **prototipo di un'app interna** in Streamlit che risponda a queste esigenze. 

ATTENZIONE!: L’obiettivo e costruire un **prototipo che un executive potrebbe consultare e usare in autonomia** per prendere decisioni informate. Non conta fare il plot più complicato ma generare insight che **guidino decisioni di business in modo accurato**. Lo scopo è generare valore per l'azienda.

---

## 5. Regole d’esame

### 5.1 Organizzazione del lavoro

- Lavorate in **gruppi da 2–3 persone**.
- Ogni gruppo utilizza **una sola repository GitHub** (clone del template o download del ZIP).

### 5.2 Tempo e freeze del codice

- Avete **4 ore** effettive di lavoro (estendibili a 5 includendo la pausa pranzo per chi lo desidera).
- **Alla scadenza (14:00 max)**:
  - effettuate l’**ultimo commit + push** su `main` (o branch principale concordato).
  - da quel momento **non potete più modificare il codice**.
  - **verificherò l'orario dell'ultimo commit** su Github e **dovreste fare un pull** dalla repo Github prima di fare girare la demo in locale.

### 5.3 Uso di strumenti esterni

- È consentito usare:
  - documentazione ufficiale pacchetti python (Streamlit, Pandas, scikit-learn, ecc.), StackOverflow, ecc.
  - tool di AI generativa **come supporto**, ma:
    - vi chiederò di illustrare **come e quando avete usato l'AI generativa**.
    - dovrete comunque **capire e saper spiegare** ogni parte del codice!
    - **farò domande su snippet di codice** e su scelte prese.

### 5.4 Presentazione

- Presenterete usando **l’app** su cloud se avete fatto deploy, in locale se avete usato Ollama.
- Ogni gruppo ha **10-15 minuti** per:
  - partire **dai need e arrivare subito agli insight** principali ottenuti.
  - giustificare come siete arrivati a risposte e decisioni tramite la **navigazione** dell’app.
  - farmi capire come usarla in autonomia se fossi un **executive**.
- **Potete utilizzare slides** a supporto della demo.
---

## 6. Criteri di valutazione (focus su business & executive experience)

| Area                                            | Peso | Cosa viene valutato                                                                                       |
|-------------------------------------------------|:---:|-----------------------------------------------------------------------------------------------------------|
| 1. Comprensione business & domande ai dati      | 30% | Chiarezza del need, domande ben poste, coerenza tra grafici, metriche e problemi reali del business      |
| 2. Esperienza per l’executive (UX & storytelling)| 25% | Facilità di navigazione, etichette chiare, testi di supporto, layout pulito, insight leggibili e “belli” |
| 3. Modello ML & interazione business-friendly   | 20% | Target sensato, pipeline corretta, metriche comprensibili, form di input usabile da un non tecnico       |
| 4. Qualità tecnica & struttura del codice       | 15% | Organizzazione della repo, pulizia del codice, uso corretto di Streamlit, gestione base degli errori     |
| 5. Extra & GenAI                                | 10% | Pagina LLM/GenAI, idee originali, cura del “prodotto finale” oltre il minimo indispensabile              |

## 7. Hints: “pensare come un executive”

Obiettivo: un executive ha poco tempo, non guarda il codice e vuole capire subito:
- cosa succede se prende una certa decisione,
- se le conclusioni sono supportate dai dati.

### Dashboard

- Poche KPI **parlanti** (es. “Vendite globali ultimi 5 anni (M copie)”).
- 2–3 grafici ben fatti, non troppi plot caotici.
- Titoli, colori e legende **autoesplicativi**.
- Sotto ogni grafico 1–2 frasi che spiegano cosa mostra e come leggerlo.

### Pagina ML

- Spiega in alto, in italiano:
  - il **target** (es. “HIT = vendite globali ≥ 1M copie”),
  - cosa significa in termini di business.
- Form di input come una **scheda prodotto** realistica.
- Output intuitivo (es. probabilità di HIT con semaforo verde/giallo/rosso).
- Mostra almeno un paio di esempi reali (un gioco HIT e uno no).
- Modello **robusto ma spiegabile**.

### Dati e confronti

- Controllate missing, outlier e scala delle vendite; se serve aggiungete una breve sezione “Note sui dati”.
- Evitate confronti fuorvianti (es. sommare vendite totali di console con anni di vita diversi).
- Quando confrontate generi/piattaforme o recensioni/vendite, specificate sempre il **contesto** (anno, piattaforma, area geografica).