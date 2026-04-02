# Docker-Anleitung

🇺🇸 [English](README.docker.md) | 🇰🇷 [한국어](README.docker.ko.md) | 🇨🇳 [中文](README.docker.zh.md) | 🇯🇵 [日本語](README.docker.ja.md) | 🇩🇪 [Deutsch](README.docker.de.md)

Diese Anleitung fuehrt Sie Schritt fuer Schritt durch das Erstellen und Ausfuehren des Simulation-Generation-Agent in einem Docker-Container. Eine lokale IsaacLab-Installation ist nicht erforderlich.

---

## Voraussetzungen

Stellen Sie vor dem Start sicher, dass Sie Folgendes haben:

- [ ] **NVIDIA GPU** mit aktuellen Treibern
- [ ] **Docker Engine** (v26.0+)
- [ ] **NVIDIA Container Toolkit**
- [ ] **OpenAI API-Schluessel** (oder Azure OpenAI-Anmeldedaten)

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

Oeffnen Sie `.env` und setzen Sie Ihren API-Schluessel:
```
OPENAI_API_KEY=sk-your-key-here
```

> **Hinweis**: Die `.env`-Datei wird von Git ignoriert und niemals in das Docker-Image eingebettet. Schluessel werden zur Laufzeit ueber `-e`-Flags injiziert.

---

## Schritt 3: Docker-Image erstellen

```bash
docker build -t simgen-agent .
```

Dabei werden folgende Schritte ausgefuehrt:
1. CUDA 12.1 Basis-Image herunterladen
2. Python 3.11 und Systembibliotheken installieren
3. Isaac Sim 5.1.0 ueber pip installieren
4. IsaacLab v2.3.2 klonen und installieren
5. Projektabhaengigkeiten installieren

> **Der erste Build dauert je nach Internetgeschwindigkeit ca. 30-60 Minuten**. Dank Docker-Layer-Caching sind nachfolgende Builds deutlich schneller.

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
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

Mit Optionen:
```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

### Option B: JSON-Eingabe

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
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
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode isaac-lab --task tasks/franka/stack/franka_stack.yaml

# Nur Stage 3: Datensammlung
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode data-collection --task tasks/franka/stack/franka_stack.yaml

# Batch-Ausfuehrung (mehrere Aufgaben)
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode e2e-batch --config configs/docker/e2e_batch_smoke.yaml
```

### Hilfe anzeigen

```bash
docker run --rm simgen-agent --help
```

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

Die IsaacLab-Simulation benoetigt mindestens 6GB GPU-Speicher. Bei OOM-Fehlern:
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
| `OPENAI_MODEL` | Nein | Standardmodell ueberschreiben (Standard: gpt-5-mini) |

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
