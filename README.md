# Bioscience-PLM

Studiamo gli effetti delle sostituzioni amminoacidiche di TEM-1 con ESM-2 ed ESM-C. Il progetto ha **tre passaggi**: predizione zero-shot, preparazione degli embedding e regressione supervisionata. I risultati salvati permettono di esplorare il confronto senza ripetere l’inferenza dei modelli proteici.

**Indice:** [Obiettivo biologico](#1-obiettivo-biologico-e-fitness) · [Dati e struttura](#2-dataset-e-struttura) · [Zero-shot](#3-zero-shot) · [Embedding](#4-preparazione-degli-embedding) · [Supervised](#5-supervised-con-ridge) · [Installazione e riproducibilità](#6-installazione-riproducibilità-e-limiti)

## 1. Obiettivo biologico e fitness

Le proteine sono catene di amminoacidi, detti *residui*. **TEM-1 è una beta-lattamasi**, un enzima che può inattivare antibiotici beta-lattamici come l’ampicillina. Una TEM-1 sufficientemente funzionante può proteggere il batterio **Escherichia coli** che la produce.

Il **wild type (WT)** è la sequenza di riferimento. Una sostituzione come `H24V` cambia l’istidina (`H`) alla posizione 24 in valina (`V`), producendo una variante.

Qui la **fitness** misura la prestazione del batterio sotto antibiotico: maggiore fitness significa che il batterio se la cava meglio nelle condizioni del saggio. Compromettere la protezione può aumentare la sensibilità all’ampicillina e ridurre la crescita. La crescita è però una misura complessiva: non identifica automaticamente cambiamenti nella catalisi, stabilità, quantità della proteina o altri fattori.

Fitness, patogenicità e capacità dell’antibiotico di uccidere non sono misure intercambiabili. L’interesse applicativo è comprendere fenomeni utili a contrastare la resistenza. **Il nostro obiettivo è migliorare la previsione degli effetti in entrambe le direzioni**, non massimizzare la fitness batterica.

## 2. Dataset e struttura

Il **deep mutational scanning (DMS)** misura molte varianti in parallelo: la selezione modifica la loro rappresentazione nella popolazione e il sequenziamento ne fornisce i conteggi. Noi utilizziamo dati già pubblicati, raccolti e standardizzati da [ProteinGym](https://github.com/OATML-Markslab/ProteinGym#overview).

Il saggio `BLAT_ECOLX_Stiffler_2015` contiene **4.996 sostituzioni singole**, nelle **263 posizioni 24–286** di un riferimento di **286 residui**, alla condizione selezionata di **2.500 µg/mL di ampicillina**. Manca `H24A` rispetto alle 4.997 sostituzioni possibili nella regione: non imputiamo un valore. Le coordinate partono da 1 nella sequenza ProteinGym, senza offset aggiuntivi.

| Colonna | Contenuto |
| --- | --- |
| `mutant` | Identità della sostituzione, per esempio `H24V` |
| `mutated_sequence` | Sequenza completa della variante |
| `DMS_score` | Target sperimentale continuo |
| `DMS_score_bin` | Classe derivata applicando una soglia al target |

[Stiffler et al., equazione 1](https://ranganathanlab.org/wp-content/uploads/2018/03/Stiffler_Ranganathan.pdf) definiscono il punteggio tramite l’arricchimento sotto selezione rispetto alla condizione senza antibiotico, normalizzato al WT:

```text
DMS_score = log10(arricchimento della variante / arricchimento del WT)
```

**Vicino a zero significa prestazione simile al WT; negativo inferiore; positivo superiore.** Zero non significa assenza di crescita e lo score non è una percentuale di sopravvivenza. Un piccolo valore positivo non dimostra da solo un vantaggio significativo. ProteinGym orienta i punteggi verso fitness crescente, ma lo zero non ha lo stesso significato in tutti i saggi.

Qui `DMS_score_bin` usa la **mediana −1,159498916**, non zero né una soglia clinica di resistenza. Il benchmark usa il target continuo.

```text
configs/                 checkpoint, revisioni e mappa degli artefatti storici
src/tem1/                zero-shot, embeddings.py, supervised.py e utilità
notebooks/               tem1_exploration.ipynb, tem1_supervised_exploration.ipynb
data/raw/                FASTA, ZIP ProteinGym; tem1/ per la cache estratta
data/processed/          stiffler_single_folds.csv
features/esm2_150m/      WT e feature delle varianti
results/zero_shot/       esm2/ e esmc/
results/supervised/      baseline/ e global_extension/
tests/                   verifiche automatiche
```

I cinque fold fissati nel [CSV](data/processed/stiffler_single_folds.csv) contengono **53, 53, 53, 52, 52 posizioni**: ogni posizione appartiene a un solo fold. Il codice controlla identità delle varianti, target, sequenze e assegnazioni; dati mancanti o incoerenti producono un errore, senza esclusioni silenziose.

## 3. Zero-shot

Un **protein language model (pLM)** apprende regolarità delle sequenze proteiche. L’ipotesi è: **le sostituzioni meno compatibili con quelle regolarità tendono anche a compromettere maggiormente la prestazione misurata?**

Con *masked marginals* nascondiamo un solo sito del WT e confrontiamo nello stesso contesto le probabilità del residuo alternativo e dell’originale:

```text
score = ln P(alternativa | WT mascherato) − ln P(originale | WT mascherato)
```

Positivo significa che il modello preferisce l’alternativa; negativo che preferisce il WT. **Uno score negativo non dimostra automaticamente perdita di funzione**: questo score probabilistico e il `DMS_score` sperimentale hanno scale e zeri diversi. Il modello non riceve la concentrazione dell’antibiotico. *Zero-shot* significa nessun training sulle etichette DMS del saggio; proteina o omologhi potrebbero essere stati presenti nel pretraining.

La **correlazione di Spearman (ρ)** misura l’accordo fra graduatorie: +1 indica ordine concorde, −1 opposto, circa zero scarso accordo monotono. **0,73 non significa 73% di predizioni corrette** e non misura la calibrazione dei valori assoluti.

Run completi su RTX 3090, stesse 4.996 varianti, FP32 e TF32 disabilitato:

| Modello | Spearman | Intervallo bootstrap 95% |
| --- | ---: | --- |
| ESM-2 35M, baseline | 0,556712 | [0,502; 0,609] |
| ESM-2 150M | 0,660518 | [0,613; 0,703] |
| ESM-2 650M | 0,731491 | [0,683; 0,771] |
| ESM-C 600M | 0,736425 | [0,695; 0,770] |

M indica milioni nominali di parametri. La GPU accelera il calcolo; le differenze di correlazione riguardano i modelli, e una maggiore dimensione non garantisce risultati migliori.

Il **bootstrap appaiato per posizione** usa 2.000 ripetizioni, seed 0: ogni volta estrae 263 posizioni con ripetizione, includendo tutte le sostituzioni di ciascun sito. Tutti i modelli ricevono la stessa estrazione. Non si rifanno training o predizioni: cambiano composizione e peso dei siti. Ripetere la posizione 102 ripete l’intero blocco di varianti. I percentili 2,5 e 97,5 delimitano l’intervallo; questa variabilità interna al saggio non rappresenta l’errore di ogni variante o le repliche di laboratorio.

**ESM-C − ESM-2 650M = +0,004934**, intervallo **[−0,013804; +0,026102]**: non emerge un vantaggio statisticamente chiaro. ESM-C usa il checkpoint ufficiale `esmc_600m_2024_12_v0.pth`, revisione e SHA-256 verificati nella [configurazione](configs/esmc-600m.json). L’indirizzo EvolutionaryScale reindirizza a Biohub; i pesi coincidono con l’upload originale richiamato dal loader ProteinGym. Il nostro forward diretto FP32 evita l’autocast BF16 usato dal percorso ProteinGym.

**Riferimenti esterni ProteinGym:** il [CSV ufficiale versionato](https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv) riporta 0,735 per ESM-C 600M, 0,731 per ESM-2 650M, 0,589 per 3B e 0,430 per 15B. Gli ultimi due sono riferimenti esterni; non fanno parte dei nostri run completati. Lo 0,735 esterno non è una replica numerica del nostro protocollo FP32.

Dopo l’[installazione](#6-installazione-riproducibilità-e-limiti), dalla radice:

```bash
# Riusa la cache estratta; se assente scarica i dati ufficiali.
tem1 download
tem1 inspect
python -m tem1.benchmark --output results/zero_shot/esm2_repeat
python -m tem1.compare results/zero_shot/esm2_repeat/35m \
  results/zero_shot/esm2_repeat/150m results/zero_shot/esm2_repeat/650m \
  --replicates 2000 --seed 0 --output results/zero_shot/esm2_repeat/comparison.json
```

Per uno smoke test aggiungere `--sizes 35m --limit-positions 4` e scegliere un altro output. ESM-C richiede il suo ambiente separato:

```bash
.venv-esmc/bin/python -m tem1.benchmark --config configs/esmc-scale.json --sizes 600m \
  --output results/zero_shot/esmc_repeat
.venv-esmc/bin/python -m tem1.compare results/zero_shot/esm2/650m \
  results/zero_shot/esmc_repeat/600m --replicates 2000 --seed 0 \
  --output results/zero_shot/esmc_repeat/comparison.json
```

[Predizioni e grafici ESM-2](results/zero_shot/esm2/) e [ESM-C](results/zero_shot/esmc/) sono disponibili. Nei [grafici 650M](results/zero_shot/esm2/650m/correlation.png), lo scatter degli score mostra la tendenza crescente su scale differenti; il pannello dei ranghi confronta direttamente gli ordinamenti. Il [profilo per posizione](results/zero_shot/esm2/650m/position_profile.png) mostra medie delle sostituzioni misurate, utili per individuare sensibilità locali, senza spiegarne il meccanismo biologico.

## 4. Preparazione degli embedding

Gli **embedding** sono rappresentazioni numeriche delle sequenze. `tem1.embeddings` applica ESM-2 150M congelato a ogni sequenza mutata; questa fase può usare la GPU. Il WT e le feature salvate in [features/esm2_150m](features/esm2_150m/) preparano il successivo training sulla CPU.

Ogni vettore base ha **640 componenti**:

- `emb_mut_pos`: residuo mutato;
- `mean_local_5`, `mean_local_10`, `mean_local_20`: medie entro ±5/10/20 residui, limitate ai bordi;
- `mean_pooled`: media di tutti i residui della sequenza mutata.

Le medie escludono i token speciali. L’inferenza usa FP32; gli artefatti sono memorizzati FP16. La colonna **`zeroshot` usa il WT senza masking**, con un solo forward: è il metodo **WT-marginals**, distinto dalla baseline masked-marginals. Per associare feature e target si usa sempre `mutant`, non l’ordine implicito delle righe.

```bash
# Verifica gli artefatti disponibili, senza ricalcolare gli embedding.
python -m tem1.embeddings --validate-only --local-files-only
# Nuova estrazione; togliere --local-files-only se il checkpoint manca in cache.
python -m tem1.embeddings --device cuda --batch-size 32 --local-files-only \
  --wt-out features/esm2_150m/repeat/esm_wt.npz \
  --variants-out features/esm2_150m/repeat/esm_variant_features.parquet
```

Il checkpoint è fissato alla revisione `a695f6045e2e32885fa60af20c13cb35398ce30c`. Il manifest `.parquet.run.json` consente la ripresa solo con provenienza compatibile. Gli artefatti originali non registrano la revisione: restano validabili, ma non vengono mescolati automaticamente a nuove estrazioni. Un solo vettore globale WT identico per tutte le varianti non basterebbe a distinguerle.

[tem1_exploration.ipynb](notebooks/tem1_exploration.ipynb) conserva l’esplorazione iniziale di dati, WT e fold; non sostituisce l’estrazione modulare.

## 5. Supervised con Ridge

`tem1.supervised` usa **StandardScaler + Ridge** sugli embedding congelati, convertiti in float64. I cinque fold esterni per posizione sono fissati: una posizione di test non compare nel relativo training. All’interno del solo training esterno, **GroupKFold(4)** sceglie alpha minimizzando l’**MSE medio** sulla griglia `10**arange(-2, 6, 0.5)`. Lo scaler viene stimato in ogni training interno; il modello scelto viene rifittato sul training esterno. Solver Cholesky, CPU, 2 thread.

Ogni variante riceve esattamente una predizione **fuori training (OOF)**. I due zero-shot ESM-2 150M vengono riutilizzati sulle medesime varianti di test. Per confrontare tutti i metodi usiamo Spearman; RMSE e MAE riguardano solo i regressori, perché gli score zero-shot non sono calibrati alla scala della fitness.

| Metodo | Feature | Spearman: media ± DS fold | Spearman OOF | RMSE OOF | MAE OOF |
| --- | ---: | ---: | ---: | ---: | ---: |
| WT-marginals | — | 0,6425 ± 0,0410 | 0,6410 | — | — |
| Masked-marginals | — | 0,6634 ± 0,0289 | 0,6605 | — | — |
| emb_mut_pos | 640 | 0,7161 ± 0,0254 | 0,7125 | 0,8182 | 0,6590 |
| mean_local_5 | 640 | 0,6431 ± 0,0501 | 0,6363 | 0,8943 | 0,7261 |
| mean_local_10 | 640 | 0,6383 ± 0,0708 | 0,6356 | 0,9020 | 0,7280 |
| mean_local_20 | 640 | 0,6152 ± 0,0859 | 0,6109 | 0,9218 | 0,7508 |
| mean_pooled | 640 | 0,6730 ± 0,0484 | 0,6753 | 0,8768 | 0,6988 |
| emb_mut_pos + delta | 1280 | 0,7682 ± 0,0219 | 0,7605 | 0,7670 | 0,6129 |
| emb_mut_pos + delta + mean_pooled | 1920 | 0,7647 ± 0,0256 | 0,7600 | 0,7682 | 0,6093 |

`delta` è la differenza fra embedding mutato e WT allo stesso sito. Aggiungere la media globale ai 1.280 valori produce cambiamenti piccoli e misti: Spearman OOF **−0,000517**, RMSE **+0,001238**, MAE **−0,003614**. Non emerge un miglioramento complessivo con questo protocollo.

La **media ± DS** pesa ugualmente i cinque fold, con deviazione standard campionaria (`ddof=1`); **OOF** ricalcola la metrica sulle 4.996 predizioni aggregate, quindi può differire. La DS è descrittiva, non un intervallo di confidenza; i training esterni si sovrappongono. Il JSON contiene entrambi i riepiloghi anche per RMSE e MAE.

```bash
# Tutte le rappresentazioni, in una directory nuova.
python -m tem1.supervised --threads 2 --output results/supervised/repeat
# Solo il confronto 1.280 vs 1.920 feature.
python -m tem1.supervised --threads 2 \
  --representations emb_mut_pos_delta emb_mut_pos_delta_mean_pooled \
  --output results/supervised/global_repeat
```

Ogni run conserva **un CSV OOF** identificato da `mutant` e **un JSON** con metriche per fold, alpha, MSE interni, configurazione e hash. Non duplica gli embedding. I risultati sono in [baseline](results/supervised/baseline/) e [global_extension](results/supervised/global_extension/). [tem1_supervised_exploration.ipynb](notebooks/tem1_supervised_exploration.ipynb) richiama il codice comune per esplorarli senza ripetere il training.

## 6. Installazione, riproducibilità e limiti

Tutti i comandi si eseguono dalla radice `Bioscience-PLM`. Python supportato: 3.10–3.13; ambiente consigliato 3.12. Creare il venv solo se assente:

```bash
uv --no-config venv --python 3.12
uv --no-config pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu126 'torch==2.7.1'
uv --no-config pip install --python .venv/bin/python -e . -c requirements-baseline.txt
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
python -m unittest discover -s tests -v
```

`--no-config` evita l’incompatibilità locale fra uv 0.9.28 e la configurazione di sistema. Su macchine senza GPU sostituire `cu126` con `cpu`; Ridge non richiede CUDA. Su Colab partire dai modelli piccoli e salvare gli output prima della scadenza della sessione. I tempi fra hardware diversi non sono confronti controllati.

Per i notebook, scegliere il kernel della `.venv`; entrambi riconoscono la radice anche se avviati da `notebooks/`:

```bash
uv --no-config pip install --python .venv/bin/python -e '.[notebooks]' -c requirements-baseline.txt biopython requests jupyter
jupyter notebook notebooks/tem1_exploration.ipynb
jupyter notebook notebooks/tem1_supervised_exploration.ipynb
```

ESM-C usa un ambiente separato per conservare le versioni compatibili del SDK e di Transformers:

```bash
uv --no-config venv --python 3.12 .venv-esmc
uv --no-config pip install --python .venv-esmc/bin/python --index-url https://download.pytorch.org/whl/cu126 'torch==2.7.1' 'torchvision==0.22.1'
uv --no-config pip install --python .venv-esmc/bin/python -e '.[esmc]' -c requirements-esmc.txt
```

Le configurazioni fissano checkpoint e revisioni; i vincoli diretti non sono un lockfile completo. Nei risultati sono conservati versioni effettive, configurazioni, checksum e hash delle fonti. **I JSON storici mantengono percorsi e contenuti originali**: [artifact-paths.json](configs/artifact-paths.json) ne mappa la posizione attuale, anche nei notebook. Il commit `835fc26` conserva la struttura precedente alla riorganizzazione. I vecchi run locali sono conservati; cache, ambienti e nuove repliche restano esclusi da Git. I risultati ufficiali, i fold e le feature di riferimento sono versionati. Usare sempre output nuovi per le repliche; uno smoke test non è il benchmark completo.

Le conclusioni riguardano **una proteina e una condizione**. Selezionare la rappresentazione migliore sui risultati dei fold rende il confronto esplorativo, non un nuovo test indipendente. La separazione per posizione riguarda le etichette supervisionate: il pLM usa l’intera sequenza come contesto. Generalizzare ad altre proteine richiede altri saggi e controllo dell’omologia.
