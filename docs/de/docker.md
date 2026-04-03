# Docker-Anleitung

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../../README.docker.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](../ko/docker.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](../zh/docker.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](../ja/docker.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](docker.md)

Diese Anleitung fuehrt Sie Schritt fuer Schritt durch das Erstellen und Ausfuehren des Simulation-Generation-Agent in einem Docker-Container. Eine lokale IsaacLab-Installation ist nicht erforderlich.

---

## Voraussetzungen

Stellen Sie vor dem Start sicher, dass Sie Folgendes haben:

- [ ] **NVIDIA GPU** mit aktuellen Treibern
- [ ] **Docker Engine** (v26.0+)
- [ ] **NVIDIA Container Toolkit**
- [ ] **OpenAI API-Schluessel** (oder Azure OpenAI-Anmeldedaten)

### Systemanforderungen

| Komponente | Minimum | Empfohlen |
|------------|---------|-----------|
| GPU-Speicher | 8 GB | 16+ GB |
| Festplattenspeicher | 50 GB frei | 100+ GB frei |
| RAM | 16 GB | 32+ GB |
| Betriebssystem | Ubuntu 22.04 | Ubuntu 22.04 |

### Docker Engine installieren

Folgen Sie der offiziellen Anleitung: [Docker Engine auf Ubuntu installieren](https://docs.docker.com/engine/install/ubuntu/)

Nach der Installation ueberpruefen:
```bash
docker --version
docker info
```

### NVIDIA Container Toolkit installieren

Folgen Sie der offiziellen Anleitung: [NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Nach der Installation den GPU-Zugriff aus Docker ueberpruefen:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi
```

Ihre GPU sollte aufgelistet sein. Bei einem Berechtigungsfehler versuchen Sie `newgrp docker` oder melden Sie sich ab und wieder an.

---

## Schritt 1: Quellcode herunterladen

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent
```

> Falls Sie bereits ohne `--recurse-submodules` geklont haben, fuehren Sie aus:
> ```bash
> git submodule update --init --recursive
> ```

---

## Schritt 2: API-Schluessel konfigurieren

```bash
cp .env.example .env
```

Oeffnen Sie `.env` und setzen Sie Ihren API-Schluessel (fuer lokale Ausfuehrung):
```
OPENAI_API_KEY=sk-your-key-here
```

> **Docker-Benutzer**: Fuer Docker-Ausfuehrungen muessen Sie keine `.env`-Datei erstellen. API-Schluessel werden zur Laufzeit direkt ueber `-e`-Flags uebergeben. Die `.env`-Datei wird nur fuer lokale (Nicht-Docker-)Ausfuehrung benoetigt.

---

## Schritt 3: Docker-Image erstellen

Dieses Repository enthaelt ein `Dockerfile`, das die vollstaendige Umgebung automatisch einrichtet.

```bash
docker build -t simgen-agent .
```

> **Festplattenspeicher**: ca. 50GB erforderlich. **Erster Build**: ca. 30-60 Minuten. Dank Docker-Layer-Caching sind nachfolgende Builds deutlich schneller.

### Was das Dockerfile ausfuehrt

Das `Dockerfile` im Repository-Stammverzeichnis erstellt ein eigenstaendiges Image mit:

| Layer | What's installed |
|-------|-----------------|
| Base | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| Python | 3.11 (venv at `/opt/isaaclab-env`) |
| Isaac Sim | 5.1.0 (pip from `pypi.nvidia.com`) |
| IsaacLab | v2.3.2 (source build from GitHub) |
| PyTorch | 2.7.0 (CUDA 12.8) |
| Project | `requirements.txt` + all source code |
| Entry point | `ENTRYPOINT ["./run_agent.sh"]` |

Der Einstiegspunkt des Containers ist `run_agent.sh`. Wenn Sie also ausfuehren:
```bash
docker run simgen-agent "Stack the blocks..."
```
wird automatisch `run_agent.sh "Stack the blocks..."` im Container ausgefuehrt.

Ueberpruefen Sie, ob das Image erstellt wurde:
```bash
docker images | grep simgen-agent
```

---

## Schritt 4: Pipeline ausfuehren

### Option A: Natuerlichsprachliche Eingabe (einfachste Variante)

Beschreiben Sie einfach, was der Roboter tun soll:

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key-here" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

Mit Optionen:
```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key-here" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

### Option B: JSON-Eingabe

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  data/input_sample.json results/output.json
```

Die Standard-`data/input_sample.json` enthaelt eine Beispiel-FrankaStackTray-Aufgabe. Sie koennen auch eigene erstellen:
```json
{
  "tasks": [
    {"task_description": "Pick up the cube and place it on the target", "robot": "franka"}
  ],
  "config": {"target_success": 1, "max_attempts": 3}
}
```

### Option C: Einzelne Stufen (Fortgeschritten)

```bash
# Nur Stage 2: IsaacLab-Umgebungscode generieren
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode isaac-lab --task tasks/franka/stack/franka_stack.yaml

# Nur Stage 3: Datensammlung
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode data-collection --task tasks/franka/stack/franka_stack.yaml

# Batch-Ausfuehrung (mehrere Aufgaben)
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode e2e-batch --config configs/docker/e2e_batch_smoke.yaml
```

### Azure OpenAI verwenden

Um Azure OpenAI anstelle der OpenAI-Plattform zu verwenden, übergeben Sie die Azure-Anmeldedaten über `-e` Flags:

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e AZURE_OPENAI_API_KEY="your-azure-key" \
  -e AZURE_OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/v1/" \
  -e AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
  -e AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

| Variable | Erforderlich | Beschreibung |
|----------|-------------|--------------|
| `AZURE_OPENAI_API_KEY` | Ja | Azure OpenAI API-Schlüssel |
| `AZURE_OPENAI_BASE_URL` | Ja | Azure-Endpunkt (mit `/openai/v1/` Suffix) |
| `AZURE_OPENAI_ENDPOINT` | Ja | Azure-Ressourcenendpunkt (wird in Stage 1 verwendet) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Ja | In Azure bereitgestellter Modellname (wird in Stage 1 verwendet) |

> Sie benötigen entweder `OPENAI_API_KEY` oder den obigen Azure-Variablensatz. Nicht beides.

### Modellauswahl

Das Standardmodell ist `gpt-5`. Sie können es mit der Umgebungsvariable `OPENAI_MODEL` überschreiben:

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="your-key" \
  -e OPENAI_MODEL="gpt-4o" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

### Hilfe anzeigen

```bash
docker run --rm simgen-agent --help
```

### Ausgabe waehrend der Ausfuehrung

Waehrend die Pipeline laeuft, werden uebersichtliche Fortschrittsmeldungen im Terminal angezeigt:

```
🚀 RAPIDS Pipeline — "Stack the blocks inside the tray on the table"
   Robot: franka | Target: 5 episodes

  ✅ Stage 1: NL → YAML                              1m 12s
  ✅ Stage 2: YAML → IsaacLab                         9m 44s
  ✅ Stage 3: Data Collection (5/5 episodes)           7m 30s

──────────────────────────────────────────────────────
  📊 Result: ✅ completed
  ⏱️  Total: 18m 26s
  🔤 Tokens: 123,008 (10 API calls)
  💰 Cost: ~$0.15
  📄 Output: results/output.json
  📁 Logs: outputs/challenge_run_20260402_151823/
──────────────────────────────────────────────────────
```

Waehrend jede Stufe laeuft, wird ein Drehindikator angezeigt. Detaillierte Protokolle werden in Dateien gespeichert — im Terminal erscheinen nur uebersichtliche Statuszeilen.

---

## Schritt 5: Ergebnisse ansehen

Die Ergebnisse werden unter `./artifacts/` auf Ihrem Host-Rechner gespeichert (gemappt von `/workspace/artifacts` im Container).

```bash
ls artifacts/
```

Typische Ausgabestruktur:
```
artifacts/
├── isaaclab/              # Stage 2: generierter Umgebungscode
│   └── 20260402_*/        # Ausfuehrungsverzeichnis mit Zeitstempel
│       ├── env_cfg.py     # Umgebungskonfiguration
│       ├── run_env.py     # Umgebungs-Runner
│       ├── result.json    # Erfolgs-/Fehlerstatus
│       └── debug/         # Screenshots (Vorderseite, Oberseite, Handgelenk)
└── data_collection/       # Stage 3: gesammelte Episoden
    └── TaskName_*/
        ├── collection_results.json
        ├── raw_dataset/   # Episodendaten
        └── videos/        # aufgezeichnete Videos
```

Im JSON-Eingabemodus werden die Ergebnisse auch in `results/output.json` geschrieben.

### Benutzerdefiniertes Ausgabeverzeichnis

Standardmaessig werden Pipeline-Ausgaben in `outputs/` gespeichert. Um den Pfad zu aendern, verwenden Sie `--output-root`:

```bash
# Lokal
./run_agent.sh "Stack the blocks" --output-root /path/to/my/outputs

# Docker
docker run --rm --gpus all \
  -v /path/to/my/outputs:/workspace/custom_outputs \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks" --output-root /workspace/custom_outputs
```

### Datensatz-Speicherort

Nach einer erfolgreichen Ausfuehrung befindet sich der Rohdatensatz unter:
```
outputs/data_collection/<TaskName>_<timestamp>/raw_dataset/
├── episodes/
│   └── episode_000000/
│       ├── actions.npy          # Gelenkaktionen
│       ├── states.npy           # Gelenkzustaende
│       ├── gripper_state.npy    # Greifer oeffnen/schliessen
│       ├── tcp_world_xyzrpy.npy # TCP-Pose (Weltkoordinaten)
│       ├── tcp_robot_xyzrpy.npy # TCP-Pose (Roboterkoordinaten)
│       ├── skills.json          # Skill-Sequenz-Protokoll
│       └── images/
│           └── front_cam/*.png  # Kameraframes
└── metadata.json                # Roboterinfo, DOF, Gelenknamen
```

---

## Fehlerbehebung

### GPU wird nicht erkannt

```
Error: could not select device driver "nvidia"
```

**Loesung**: Installieren oder reinstallieren Sie das NVIDIA Container Toolkit:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Build schlaegt bei der IsaacSim-Installation fehl

Das IsaacSim-pip-Paket ist ca. 15GB gross. Bei Netzwerkproblemen:
```bash
# Ohne Cache erneut versuchen
docker build --no-cache -t simgen-agent .
```

### Speicher nicht ausreichend (OOM)

Die IsaacLab-Simulation benoetigt mindestens **8GB GPU-Speicher** (nur Stage 2) oder **16GB+** (vollstaendige Pipeline mit Stage 3). Bei OOM-Fehlern:
- Schliessen Sie andere GPU-intensive Anwendungen
- Reduzieren Sie `--episodes` auf 1
- Verwenden Sie `--mode isaac-lab`, um zuerst nur Stage 2 zu testen

### API-Schluessel funktioniert nicht

```
Either OPENAI_API_KEY or AZURE_OPENAI_API_KEY must be set.
```

**Loesung**: Stellen Sie sicher, dass der Schluessel korrekt mit `-e` uebergeben wird:
```bash
# Methode 1: Inline
-e OPENAI_API_KEY="sk-your-key"

# Methode 2: Aus .env-Datei
-e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)"

# Methode 3: Zuerst exportieren
export OPENAI_API_KEY="sk-your-key"
docker run ... -e OPENAI_API_KEY ...
```

---

## Referenz

### Umgebungsvariablen

| Variable | Erforderlich | Beschreibung |
|----------|-------------|--------------|
| `OPENAI_API_KEY` | Ja (oder Azure) | OpenAI-Plattform-API-Schluessel |
| `AZURE_OPENAI_API_KEY` | Ja (oder OpenAI) | Azure OpenAI API-Schluessel |
| `AZURE_OPENAI_BASE_URL` | Bei Azure | Azure-Endpunkt-URL |
| `OPENAI_BASE_URL` | Nein | Benutzerdefinierter OpenAI-kompatibler Endpunkt |
| `HF_TOKEN` | Nein | HuggingFace-Token (fuer Dataset-Upload) |
| `OPENAI_MODEL` | Nein | Standardmodell ueberschreiben (Standard: gpt-5) |

### `run_agent.sh`-Modi

| Modus | Beschreibung |
|-------|--------------|
| *(Positionsargumente)* | Vollstaendige Pipeline: natuerlichsprachliche oder JSON-Eingabe |
| `--mode e2e-batch` | Batch-Ausfuehrung mehrerer Aufgaben |
| `--mode isaac-lab` | Nur Stage 2: Umgebungscode-Generierung |
| `--mode data-collection` | Nur Stage 3: Datensammlung |

### Image-Spezifikationen

| Komponente | Version |
|------------|---------|
| Basis-Image | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| Python | 3.11 |
| Isaac Sim | 5.1.0 (pip) |
| IsaacLab | v2.3.2 (source) |
| PyTorch | 2.7.0 (CUDA 12.8) |

### Validierungsskripte

```bash
# Statische Pruefung (Secrets, Pfade, Struktur)
python3 scripts/audit_release_repo.py

# Vollstaendige Docker-Validierung (Build + Ausfuehrung + Test)
python3 scripts/validate_docker_release.py

# Vollstaendiger Dauertest (alle Aufgaben)
python3 scripts/validate_docker_release.py --run-full-soak
```
