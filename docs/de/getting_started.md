# Installationsanleitung

🇺🇸 [English](../getting_started.md) | 🇰🇷 [한국어](../ko/getting_started.md) | 🇨🇳 [中文](../zh/getting_started.md) | 🇯🇵 [日本語](../ja/getting_started.md) | 🇩🇪 [Deutsch](getting_started.md)

Zielgruppe: Nutzer, die dieses Repository zum ersten Mal ausfuehren
Fuer detaillierte CLI-Optionen und Ergebnisinterpretation: `docs/de/usage.md`

## 1. Allgemeine Installation

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent

pip install -e .
cp .env.example .env
```

### Submodul (AutoDataCollector)

Dieses Projekt enthaelt `external/AutoDataCollector` als Git-Submodul. Es wird waehrend der Datensammlung (Stufe 3) fuer Bewertungsprompts und IK-Hilfsfunktionen referenziert.

Beim Klonen mit `--recurse-submodules` wird es automatisch heruntergeladen. Falls es versaeumt wurde, manuell initialisieren:

```bash
git submodule update --init --recursive
ls external/AutoDataCollector/  # Dateien sollten vorhanden sein
```

Erforderliche `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1/
```

Optionale Einstellungen:

```bash
# VLM-Backends (fuer Umgebungsverifikation + Episodenerfolgs-Bewertung)
# ANTHROPIC_API_KEY=...
# GOOGLE_API_KEY=...

# Pfade
# ISAACLAB_PATH=~/workspace/IsaacLab

# Token-Verbrauchsverfolgung
# TOKEN_USAGE_FILE=outputs/token_usage.jsonl
```

## 2. IsaacLab-Installation (Primaerer Pfad)

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

So verweisen Sie das Projekt auf Ihren IsaacLab-Installationsort:

```bash
export ISAACLAB_PATH=~/workspace/IsaacLab
```

Oder setzen Sie `isaaclab.path` direkt in `configs/isaaclab_agent_config.yaml`.

Verbindung ueberpruefen:

```bash
# Lokal (Conda-Umgebung)
conda run -n env_isaaclab --no-capture-output \
  python -c "import isaaclab; print('IsaacLab import OK')"

# In Docker wird venv statt Conda verwendet, daher wie folgt ueberpruefen
# python -c "import isaaclab; print('IsaacLab import OK')"

python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run
```

## 3. Isaac Sim + MCP-Installation (Optional)

Isaac Sim wird nur fuer den visuellen Verifikationspfad benoetigt.

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/isaac-sim-mcp.git
```

```bash
cd ~/workspace/isaac-sim
./isaac-sim.streaming.sh \
  --ext-folder /home/$USER/workspace/isaac-sim-mcp \
  --enable isaac.sim.mcp_extension
```

Bei erfolgreichem Start lauscht ein TCP-Socket auf `localhost:8766`.

Verbindung ueberpruefen:

```bash
python3 tests/test_components.py connection
```

## 4. Datensammlungs-Installation (Optional)

Die Datensammlung laeuft auf IsaacLab auf.

```bash
pip install -e ".[data-collection]"
pip install pin
```

Ueberpruefen:

```bash
python3 -c "from src.agent.data_collection.adc_imports import is_adc_available; print(is_adc_available())"
python3 -c "import pinocchio; print('Pinocchio OK')"
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --help
```

## 5. Task-Spec-Agent-Installation (NL -> YAML)

Der Task-Spec-Agent ist eine Stufe-1-Pipeline, die natuerlichsprachliche Eingaben in YAML-Aufgabenspezifikationen umwandelt.

```bash
pip install langchain langchain-openai langchain-community faiss-cpu sentence-transformers
```

Der Vektorspeicher wird beim ersten Lauf automatisch unter `data/vector_store/` erstellt (Indizierung vorhandener YAMLs aus dem tasks/-Verzeichnis).

Ueberpruefen:

```bash
cd scripts/task_spec_agent
python3 -c "from rag_match_yaml_generator import RAGYAMLGenerator; print('RAG OK')"
cd ../..
```

Eigenstaendige Ausfuehrung:

```bash
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube" --robot franka --output outputs/test_task.yaml
```

## 6. Empfohlene Reihenfolge fuer den ersten erfolgreichen Lauf

### Einfachste Ausfuehrung -- Eine einzige Zeile natuerlicher Sprache

```bash
./run_agent.sh "Stack the blocks inside the tray on the table"
```

Geben Sie einfach eine natuerlichsprachliche Aufgabenbeschreibung ein, und die gesamte NL->YAML->IsaacLab->Datensammlungs-Pipeline wird automatisch ausgefuehrt.

Optionen:
```bash
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

### Vollstaendige Pipeline (Batch mit 13 Aufgaben)

```bash
bash scripts/run_full_test.sh
```

Fuehrt alle 3 Stufen sequenziell fuer 13 Franka-Aufgaben aus:
1. **NL -> YAML**: `task_spec_agent` wandelt natuerliche Sprache in YAML um
2. **YAML -> IsaacLab**: LLM generiert Umgebungs-Python-Code und verifiziert die Ausfuehrung
3. **CaP -> Daten**: LLM generiert Skill-Code, fuehrt ihn ueber IK aus, bewertet den Erfolg und sammelt Daten

### Nur IsaacLab (nur Stufe 2)

```bash
./run_agent.sh --mode isaac-lab --task tasks/franka/stack/franka_stack.yaml
```

Erfolgskriterien:
- `outputs/isaaclab/<run_dir>/env_cfg.py`
- `outputs/isaaclab/<run_dir>/result.json`

### Nur Datensammlung (nur Stufe 3)

```bash
./run_agent.sh --mode data-collection --task tasks/franka/stack/franka_stack.yaml
```

Erfolgskriterien:
- `outputs/data_collection/<run_dir>/collection_results.json`
- `outputs/data_collection/<run_dir>/raw_dataset/`

### Trainingsvorbereitungspfad

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets

python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

## 7. Mit Docker ausfuehren

Sie koennen sofort mit einem Docker-Container starten, ohne lokale Umgebungseinrichtung.

Voraussetzungen: [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
# Submodul initialisieren + Image erstellen
git submodule update --init --recursive
docker build -t simgen-agent .

# Vollstaendige Pipeline mit natuerlichsprachlicher Eingabe ausfuehren
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5

# JSON-Eingabe
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  data/input_sample.json results/output.json

# Hilfe anzeigen
docker run --rm simgen-agent --help
```

> **Unterschied im Ausgabepfad**: In Docker werden Ergebnisse unter `/workspace/artifacts` gespeichert (auf dem Host nach `./artifacts/` gemountet). Lokal werden Ergebnisse unter `outputs/` gespeichert. Um in Docker denselben Pfad wie lokal zu verwenden, fuegen Sie `-e SIMGEN_ARTIFACT_ROOT=/workspace/Simulation-Generation-Agent/outputs` hinzu.

Detaillierte Docker-Anleitung: [README.docker.md](../../README.docker.md)

## 8. Was Sie als Naechstes lesen sollten

- Details zur Pipeline-Architektur: [docs/de/architecture.md](architecture.md)
- CLI-Optionen / Ausgabestruktur / Ergebnisinterpretation: [docs/de/usage.md](usage.md)
