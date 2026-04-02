# Datensatz-Workflow

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../dataset.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](../ko/dataset.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](../zh/dataset.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](../ja/dataset.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](dataset.md)

Diese Anleitung beschreibt den vollstaendigen Workflow von der Pipeline-Ausfuehrung bis zum Hochladen eines trainingsbereitigen Datensatzes auf HuggingFace.

---

## Uebersicht

```
run_agent.sh "task description"
    |
    v
outputs/data_collection/<run>/raw_dataset/     <-- Stage 3 output
    |
    v  scripts/export_dataset.py
outputs/exported_datasets/<name>/              <-- Normalized training schema
    |
    v  scripts/preprocess_dataset.py
outputs/preprocessed_datasets/<name>/          <-- train.jsonl / val.jsonl
    |
    v  scripts/convert_lerobot_dataset.py
outputs/lerobot_datasets/<repo_id>/            <-- LeRobot v3.0 format (Parquet)
    |
    v  scripts/publish_lerobot_dataset.py
https://huggingface.co/datasets/<org>/<name>   <-- HuggingFace Hub
```

---

## Schritt 1: Pipeline ausfuehren

```bash
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

Wenn Stage 3 abgeschlossen ist, werden die Ergebnisse gespeichert unter:
- **Lokal**: `outputs/data_collection/<TaskName>_<timestamp>/`
- **Docker**: `/workspace/artifacts/data_collection/` (eingebunden unter `./artifacts/`)

### Ausgabe ueberpruefen

```bash
ls outputs/data_collection/
# FrankaStackTray_20260402_151823/

ls outputs/data_collection/FrankaStackTray_20260402_151823/
# collection_results.json   raw_dataset/   videos/

cat outputs/data_collection/FrankaStackTray_20260402_151823/collection_results.json
```

Die Datei `collection_results.json` enthaelt:
- `pipeline_completed`: ob die Datenerhebung abgeschlossen wurde
- `geometry_successful_episodes`: Episoden, die die geometrische Verifizierung bestanden haben
- `total_episodes`: Gesamtzahl der versuchten Episoden

### Struktur des Rohdatensatzes

```
raw_dataset/
├── metadata.json              # Robot config, DOF, FPS, cameras
├── episodes/
│   ├── episode_000000/
│   │   ├── states.npy         # (T, N_dof) joint positions
│   │   ├── actions.npy        # (T, N_dof) commanded positions
│   │   ├── tcp_world_xyzrpy.npy  # (T, 6) TCP in world frame
│   │   ├── tcp_robot_xyzrpy.npy  # (T, 6) TCP in robot base frame
│   │   ├── gripper_state.npy  # (T, 1) gripper state
│   │   ├── goal_robot_xyzrpy.npy # (T, 6) goal pose per frame
│   │   ├── skills.json        # Per-frame skill metadata
│   │   └── images/
│   │       ├── top/           # 000000.png, 000001.png, ...
│   │       └── wrist/
│   └── episode_000001/
│       └── ...
```

---

## Schritt 2: Export (Optional)

Normalisiert Rohdaten in ein trainingsfreundliches Schema mit konsistenter Gelenk-Normalisierung.

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets
```

Optionen:
- `--schema adc_compatible` (Standard): Normalisiert Gelenkwerte auf den Bereich [-100, 100], ADC-kompatible Feldnamen
- `--schema canonical_training`: Sauberes Schema mit expliziten TCP-Ist/Soll-Posen
- `--no-link-images`: Bilder kopieren statt Symlinks zu erstellen

---

## Schritt 3: Vorverarbeitung (Optional)

Erstellt Frame-basierte Trainingsmanifeste mit Train/Val-Aufteilung.

```bash
python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets \
  --success-only \
  --train-ratio 0.9
```

Ausgabe:
- `train.jsonl` / `val.jsonl` — Einzelbild-Samples mit Dateipfaden
- `manifest.json` — Schema und Statistiken

---

## Schritt 4: In LeRobot-Format konvertieren

Konvertiert Rohdaten in das [LeRobot v3.0](https://github.com/huggingface/lerobot)-Format (Apache Parquet).

### Voraussetzungen

```bash
pip install "lerobot>=0.4.0,<0.5.0"
```

### Konvertierung

```bash
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --repo-id local/franka_stack_sim \
  --output-root outputs/lerobot_datasets
```

Ausgabestruktur:
```
outputs/lerobot_datasets/local/franka_stack_sim/
├── meta/
│   └── info.json           # LeRobot metadata, features, frame count
├── data/
│   └── chunk-0000/
│       └── file-00000.parquet
└── videos/                 # (if video recording enabled)
```

### Validierung

```bash
python3 scripts/check_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack_sim \
  --repo-id local/franka_stack_sim
```

Die Validierung prueft:
- Erforderliche Features: `observation.state`, `observation.gripper_state`, `observation.tcp.robot_xyzrpy`, `action`, `skill.goal_position.robot_xyzrpy`
- Parquet-Shard-Integritaet
- Frame-Anzahl > 0
- Vorhandensein von `meta/info.json`

---

## Schritt 5: Auf HuggingFace hochladen

### Voraussetzungen

```bash
pip install huggingface_hub
```

HuggingFace-Token setzen:
```bash
export HF_TOKEN="hf_your_token_here"
# Oder: huggingface-cli login
```

### Veroeffentlichen

```bash
python3 scripts/publish_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack_sim \
  --repo-id your-org/franka-stack-sim \
  --local-repo-id local/franka_stack_sim \
  --private
```

Optionen:
- `--private`: Als privaten Datensatz erstellen (Standard: oeffentlich)
- `--token-env HF_TOKEN`: Umgebungsvariable fuer die Authentifizierung (Standard)
- `--token <token>`: Token direkt uebergeben

Das Skript wird:
1. Den lokalen Datensatz validieren
2. Das HuggingFace-Dataset-Repository erstellen (falls nicht vorhanden)
3. Alle Dateien mit `upload_large_folder()` hochladen
4. Einen JSON-Bericht mit der Repository-URL ausgeben

---

## Kurzreferenz

### Alles-in-einem (von Rohdaten bis HuggingFace)

```bash
# 1. Pipeline ausfuehren
./run_agent.sh "Stack the blocks" --robot franka --episodes 5

# 2. Ausgabe finden
RUN_DIR=$(ls -td outputs/data_collection/*/ | head -1)

# 3. In LeRobot konvertieren
python3 scripts/convert_lerobot_dataset.py \
  ${RUN_DIR}/raw_dataset \
  --repo-id local/franka_stack \
  --output-root outputs/lerobot_datasets

# 4. Validieren
python3 scripts/check_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack \
  --repo-id local/franka_stack

# 5. Hochladen
python3 scripts/publish_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack \
  --repo-id your-org/franka-stack --private
```

### Batch-Pipeline (automatisiert)

Der `e2e-batch`-Modus kann den gesamten Ablauf einschliesslich LeRobot-Konvertierung und HuggingFace-Upload automatisieren:

```bash
./run_agent.sh --mode e2e-batch --config configs/docker/e2e_batch_release.yaml
```

Upload in der Batch-Konfigurations-YAML einstellen:
```yaml
hf:
  upload: true
  namespace: your-org
  dataset_name: franka-sim-dataset
  private: true
  token_env: HF_TOKEN
```

---

## LeRobot-Datensatz-Features

Jeder Frame im konvertierten Datensatz enthaelt:

| Feature | Shape | Beschreibung |
|---------|-------|--------------|
| `observation.state` | (N_dof,) | Gelenkpositionen |
| `observation.gripper_state` | (1,) | Greiferzustand |
| `observation.tcp.world_xyzrpy` | (6,) | TCP-Pose im Weltkoordinatensystem |
| `observation.tcp.robot_xyzrpy` | (6,) | TCP-Pose im Roboter-Basiskoordinatensystem |
| `action` | (N_dof,) | Kommandierte Gelenkpositionen |
| `skill.natural_language` | (1,) | Skill-Beschreibung |
| `skill.type` | (1,) | Skill-Typ (pick, place, usw.) |
| `skill.progress` | (1,) | Skill-Fortschritt [0, 1] |
| `skill.goal_position.joint` | (N_dof,) | Ziel-Gelenkpositionen |
| `skill.goal_position.world_xyzrpy` | (6,) | Zielpose im Weltkoordinatensystem |
| `skill.goal_position.robot_xyzrpy` | (6,) | Zielpose im Roboter-Basiskoordinatensystem |
| `skill.goal_position.gripper` | (1,) | Ziel-Greiferzustand |
| `observation.images.{cam}` | (480, 640, 3) | Kamerabilder (top, wrist) |

**N_dof nach Roboter:** Franka=9, UR10e=12, OpenARM=9, SO-101=6

---

## Verwandte Dokumente

- Pipeline-Architektur: [architecture.md](architecture.md)
- CLI-Nutzung: [usage.md](usage.md)
- Installation: [getting_started.md](getting_started.md)
