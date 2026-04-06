# Nutzung
[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../usage.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](../ko/usage.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](../zh/usage.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](../ja/usage.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](usage.md)

Zielgruppe: Nutzer, die das CLI tatsaechlich verwenden
Was dieses Dokument behandelt: Repraesentative Befehle, wichtige Optionen, Ausgabestruktur, Ergebnisinterpretation
Installation und Umgebungseinrichtung: `docs/de/getting_started.md`

Dieses Dokument ist die **massgebliche Quelle fuer Ausfuehrung und Ergebnisinterpretation**. Interne Implementierungsdetails und Schema-Hintergruende sind in andere Dokumente ausgelagert.
Pipeline-Architektur: [docs/de/architecture.md](architecture.md)

## Interaktiver TUI-Modus

Starten Sie `run_agent.sh` ohne Argumente, um das interaktive Terminal-Dashboard zu oeffnen:

```bash
./run_agent.sh
```

Das TUI bietet:
- **Aufgabeneingabe** — Geben Sie eine natuerlichsprachliche Aufgabenbeschreibung ein und druecken Sie Enter, um die gesamte Pipeline auszufuehren
- **Einstellungen (F1)** — Pipeline-Modus, LLM-Anbieter/Modell, Roboter, Episoden, Pfade und erweiterte Optionen konfigurieren
- **Ausfuehrungsverlauf (F2)** — Vergangene Laeufe mit Pfeiltasten durchsuchen, Enter fuer detaillierte Ergebnisse (Umgebungsbewertungen, Datensatz-Infos)
- **Hilfe (F3)** — Tastenkuerzel und Slash-Befehle

Slash-Befehle: `/config`, `/history`, `/help`, `/quit`

In Docker verwenden Sie `-it` fuer den interaktiven Modus:
```bash
docker run -it --rm --gpus all \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  --env-file .env \
  simgen-agent
```

---

## run_agent.sh — Vollstaendige Pipeline-Ausfuehrung (CLI-Modus)

Geben Sie eine natuerlichsprachliche Aufgabenbeschreibung als Argument fuer die nicht-interaktive Ausfuehrung an:

```bash
# Natuerlichsprachliche Eingabe
./run_agent.sh "Stack the blocks inside the tray on the table"
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5

# JSON-Eingabe
./run_agent.sh data/input_sample.json results/output.json
```

### Optionen

| Option | Beschreibung |
| --- | --- |
| `--robot <type>` | Robotertyp: franka, ur10e, openarm, so101 (Standard: franka) |
| `--episodes <n>` | Zielanzahl erfolgreicher Episoden (Standard: 1) |
| `--max-attempts <n>` | Maximale Anzahl von Versuchen (Standard: 3) |

### In Docker ausfuehren

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --episodes 5
```

> **Ausgabepfad**: In Docker werden Ergebnisse unter `/workspace/artifacts` gespeichert. Mounten Sie diesen Pfad mit der Option `-v` auf den Host. Bei lokaler Ausfuehrung werden Ergebnisse unter `outputs/` gespeichert.

### Ergebnisse

Ergebnisse werden als strukturiertes JSON in `results/output.json` aufgezeichnet:

```json
{
  "status": "completed",
  "tasks": [{
    "name": "StackTheBlocksInsideTheTrayOnThe",
    "steps": {
      "nl_to_yaml": {"success": true},
      "yaml_to_isaaclab": {"success": true},
      "data_collection": {"success": true, "success_episodes": 5}
    }
  }]
}
```

---

## 0. Task-Spec-Agent (NL → YAML)

Wandelt eine natuerlichsprachliche Aufgabenbeschreibung in eine strukturierte YAML-Aufgabenspezifikation um (Pipeline-Stufe 1).

> **Hinweis**: Die Verwendung von `run_agent.sh "natuerlichsprachliche Aufgabe"` fuehrt automatisch die gesamte Stufe-1→2→3-Pipeline aus.
> Nachfolgend wird beschrieben, wie Stufe 1 einzeln ausgefuehrt wird.

### Repraesentative Befehle

```bash
# Grundlegende Nutzung
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube and place it on the target" \
  --robot franka --output task.yaml

# Vorlagenbasierte Generierung ohne RAG
python3 scripts/task_spec_agent/task_spec_agent.py "Stack blocks" --no-rag

# Anderen LLM-Anbieter verwenden
python3 scripts/task_spec_agent/task_spec_agent.py "Sort the colored blocks into matching colored bins" --provider huggingface

# Ausfuehrliche Protokollierung
python3 scripts/task_spec_agent/task_spec_agent.py "Reach the goal" --verbose
```

### Wichtige Optionen

| Option | Beschreibung |
| --- | --- |
| `--robot {franka,openarm,ur10,so101}` | Zielroboter (Standard: franka) |
| `--output <path>` | YAML-Speicherpfad (Ausgabe auf stdout, wenn nicht angegeben) |
| `--provider {azure,huggingface,bedrock}` | LLM-Anbieter ueberschreiben (verwendet Konfigurationsstandard, wenn nicht angegeben, Standard=openai) |
| `--no-rag` | Vorlagenbasierte YAML-Generierung statt RAG |
| `--verbose` | DEBUG-Level-Protokollierung |

### Interne Verarbeitungsschritte

```
Natuerlichsprachliche Eingabe
  → NL-Parser (Aktionen, Objekte, Orte extrahieren)
  → Aufgabenzerleger (in atomare Aktionssequenzen zerlegen)
  → Machbarkeitspruefer (physische Machbarkeit fuer den Roboter ueberpruefen)
  → RAG-YAML-Generator (FAISS-Vektorsuche → naechstliegendes Aufgaben-YAML abgleichen)
  → task.yaml
```

## 1. IsaacLab-Pipeline (Stufe 2)

> **Ausfuehrung ueber run_agent.sh**: `./run_agent.sh --mode isaac-lab --task <yaml>`
> Nachfolgend wird beschrieben, wie das Python-Skript direkt ausgefuehrt wird.

### Repraesentative Befehle

```bash
# Generieren + Ausfuehren
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml

# Nur generieren
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run

# Generieren + Ausfuehren + Evaluieren
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate

# Vorhandene Ausgabe erneut evaluieren
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml \
  --eval-only outputs/isaaclab/frankastack_20260219_160916

# Eigenstaendiger Evaluator
python3 scripts/evaluate.py \
  outputs/isaaclab/frankastack_20260219_160916 \
  tasks/franka/stack/franka_stack.yaml

# Batch
python3 scripts/run_isaac_lab.py --batch tasks/franka/
```

### Wichtige Optionen

| Option | Beschreibung |
| --- | --- |
| `--dry-run` | Nur Code generieren, IsaacLab-Ausfuehrung ueberspringen |
| `--evaluate` | Evaluator nach Erfolg ausfuehren |
| `--eval-only <dir>` | Nur vorhandene generierte Ausgabe evaluieren |
| `--batch <dir>` | Alle YAMLs in einem Verzeichnis batch-verarbeiten |
| `--output-dir <dir>` | Anderen Ausgabewurzel-Pfad statt Standard `outputs/isaaclab` verwenden |
| `--config <path>` | Agent-Konfiguration ueberschreiben |

### Ausgabestruktur

```text
outputs/isaaclab/<task_slug>_<timestamp>/
├── env_cfg.py
├── run_env.py
├── mdp/
├── .success_marker
├── error_attempt_*.txt
├── debug/                    # Umgebungs-Screenshots (front/top/wrist)
├── result.json               # Ausfuehrungsergebnis + enthaelt scene_verification
└── eval_report.json          # Bei Verwendung von --evaluate
```

## 2. Isaac Sim-Pipeline (nur lokale Entwicklungsumgebung)

> **Hinweis**: Die visuelle Verifikation ueber Isaac Sim MCP ist nur verfuegbar, wenn Isaac Sim Desktop lokal laeuft. In Docker-Umgebungen wird sie nicht unterstuetzt.

### Repraesentative Befehle

```bash
# Automatisch erstellen + Screenshot + VLM-Bewertung
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --backend auto

# Einmal erstellen + aufnehmen ohne VLM
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --skip-vlm

# Iterations-/Schwellenwert-Ueberschreibung
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml \
  --backend azure --max-iterations 3 --threshold 85
```

### Wichtige Optionen

| Option | Beschreibung |
| --- | --- |
| `--skip-vlm` | Einmaliges Erstellen/Aufnehmen ohne VLM-Bewertung durchfuehren |
| `--backend {auto,azure,claude,gemini,ollama,mock}` | VLM-Backend-Auswahl |
| `--max-iterations <n>` | Maximale Iterationsanzahl ueberschreiben |
| `--threshold <n>` | Erfolgsschwellenwert ueberschreiben |
| `--output-dir <dir>` | Ausgabewurzel ueberschreiben |
| `--config <path>` | Alternative zu `configs/pipeline_config.yaml` |

### Ausgabestruktur

```text
outputs/isaac_sim/<task_slug>_<timestamp>/
├── iter_01.png
├── iter_02.png
├── rgb_0000.png
├── metadata.txt
└── run_report.json
```

## 3. Datensammlung (Stufe 3)

> **Ausfuehrung ueber run_agent.sh**: `./run_agent.sh --mode data-collection --task <yaml>`
> Nachfolgend wird beschrieben, wie das Python-Skript direkt ausgefuehrt wird.

### Repraesentative Befehle

```bash
# Umgebung automatisch generieren, dann sammeln
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml

# Vorhandene Umgebung verwenden
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/frankastack_20260219_160916

# Zielanzahl erfolgreicher Episoden
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --target-success 5 --max-attempts 25

# Nur geometrische Verifikation, ohne VLM
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --no-vlm-judge

# Batch
python3 scripts/run_data_collection.py --batch tasks/franka/ --episodes 10
```

### Wichtige Optionen

| Option | Beschreibung |
| --- | --- |
| `--env-dir <path>` | Vorhandenes IsaacLab-Ausgabeverzeichnis verwenden |
| `--episodes <n>` | Maximale Episodenanzahl ueberschreiben |
| `--target-success <n>` | Zielanzahl erfolgreicher Episoden |
| `--max-attempts <n>` | Obergrenze fuer Gesamtversuche |
| `--repo-id <id>` | Datensatz-ID |
| `--fps <n>` | Aufnahme-FPS ueberschreiben |
| `--no-vlm-judge` | VLM-Bewertung deaktivieren |
| `--gui` | Mit GUI statt im Headless-Modus ausfuehren |
| `--config <path>` | Datensammlungskonfiguration ueberschreiben |
| `-v`, `--verbose` | Ausfuehrliche Protokollierung |

### Ausgabestruktur

```text
outputs/data_collection/<TaskName>_<timestamp>/
├── collect_data.py
├── pipeline_config.json
├── cap_runs/
├── debug_initial_*.png
├── raw_dataset/
│   ├── episodes/
│   └── metadata.json
├── collection_results.json
├── COLLECTION_COMPLETE_MARKER
└── <repo_id>/                   # Optionale LeRobot-Konvertierungsausgabe
```

### Ergebnisinterpretation

Die ersten zu pruefenden Werte in `collection_results.json` sind die folgenden.

| Schluessel | Bedeutung |
| --- | --- |
| `pipeline_completed` | Ob die Sammlungs-/Bereinigungspipeline bis zum Ende abgeschlossen wurde |
| `target_met` | Ob die Zielanzahl erfolgreicher Episoden erreicht wurde |
| `successful_episodes` | Anzahl erfolgreicher Episoden |
| `total_episodes` | Anzahl tatsaechlich versuchter/gespeicherter Episoden |
| `raw_dataset` | Rohdatensatz-Pfad |

Interpretationsregeln:

- `success` ist derzeit ein Alias fuer `pipeline_completed`.
- `pipeline_completed=true` bedeutet nicht zwingend `target_met=true`.
- Das Standard-Abschlusskriterium ist die Erstellung von `raw_dataset/`. Die LeRobot-Konvertierung kann je nach Umgebung uebersprungen werden.

## 4. Datensatzexport / Vorverarbeitung

### Export

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets
```

Das Standardschema ist `adc_compatible`.

### Vorverarbeitung

```bash
python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

Ausgaben:

- `manifest.json`
- `samples.jsonl`
- `train.jsonl`
- `val.jsonl`

Im Standard-`adc_compatible`-Pfad sind `observation.gripper_state`, `observation.tcp.robot_xyzrpy` und `skill.goal_position.robot_xyzrpy` im Trainings-Eingabe-Manifest enthalten.

Der Standardexport verwendet das `adc_compatible`-Schema, und `canonical_training` ist fuer erweiterte Analyse vorgesehen.

### Optional: Lokale LeRobot-Konvertierung

```bash
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --repo-id local/franka_stack_sim
python3 scripts/check_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/franka_stack_sim \
  --repo-id local/franka_stack_sim
```

Das `lerobot`-Paket muss in der Host-Python-Umgebung installiert sein.

### Optional: Im Hub veroeffentlichen

```bash
python3 scripts/publish_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/franka_stack_sim \
  --repo-id <org>/<dataset_name> \
  --local-repo-id local/franka_stack_sim \
  --private
```

Die Standardauthentifizierung verwendet die Umgebungsvariable `HF_TOKEN` oder eine vorhandene `huggingface-cli login`-Sitzung.

## 5. Empfohlene Arbeitsablauf-Reihenfolge

### NL → Datensatz (Vollstaendige Pipeline)

```bash
# 1. NL → YAML
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube and stack it" \
  --robot franka --output outputs/generated_task.yaml

# 2. YAML → IsaacLab-Umgebungscode
python3 scripts/run_isaac_lab.py outputs/generated_task.yaml --evaluate

# 3. Datensammlung
python3 scripts/run_data_collection.py outputs/generated_task.yaml \
  --env-dir outputs/isaaclab/<run_dir> --target-success 10

# 4. Export / Vorverarbeitung
python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

### Vorhandenes YAML → Datensatz

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --env-dir outputs/isaaclab/<run_dir>
python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> --output-dir outputs/preprocessed_datasets
```

Fuegen Sie `scripts/run_isaac_sim.py` nur hinzu, wenn eine visuelle Verifikation erforderlich ist.

## 6. Vollstaendige Pipeline (NL → Video)

`run_full_test.sh` ist ein Integrationsskript, das sequenziell Stufe 1→2→3 fuer 13 Franka-Aufgaben ausfuehrt.

```bash
bash scripts/run_full_test.sh
```

Intern wird fuer jede Aufgabe ausgefuehrt:
1. `task_spec_agent.py` -- Natuerliche Sprache → YAML-Generierung
2. `run_isaac_lab.py` -- YAML → IsaacLab-Umgebungscode-Generierung/-Verifikation
3. `run_data_collection.py` -- CaP-Codegenerierung → Ausfuehrung → Erfolgsbestimmung → Datensammlung

### Ausgabestruktur

```text
outputs/test_run_<timestamp>/
├── summary.txt                  # Gesamtuebersicht der Aufgabenergebnisse
├── token_usage.jsonl            # API-Token-Verbrauch (wenn TOKEN_USAGE_FILE gesetzt)
├── FrankaLift/
│   ├── step1_nl_to_yaml.log
│   ├── step2_isaaclab.log
│   ├── step3_cap_execution.log
│   └── task.yaml
├── FrankaStack/
│   └── ...
└── ...
```

### Token-Verbrauchsverfolgung

```bash
export TOKEN_USAGE_FILE=outputs/token_usage.jsonl
export TOKEN_USAGE_LOG=1  # Echtzeit-Konsolenprotokollierung
bash scripts/run_full_test.sh
```

## 7. E2E-Batch-Pipeline

Verarbeitet Umgebungsgenerierung + Datensammlung + Export + LeRobot-Konvertierung fuer mehrere Aufgaben basierend auf einer Konfigurationsdatei im Batch.

### Repraesentative Befehle

```bash
# Batch-Ausfuehrung (Docker)
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml

# Nach Unterbrechung fortsetzen
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml --resume

# Reduzierte Protokollierung
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml -q
```

### Docker / run_agent.sh

```bash
# In Docker (Standardmodus: e2e-batch)
run_agent.sh --mode e2e-batch --resume

# Einzelne Aufgabe
run_agent.sh --mode isaac-lab --task tasks/franka/lift/franka_lift.yaml

# Datensammlung
run_agent.sh --mode data-collection --task tasks/franka/lift/franka_lift.yaml -- --episodes 10
```

### Konfigurationsstruktur

Die E2E-Batch-Konfiguration im YAML-Format besteht aus 4 Abschnitten:

| Abschnitt | Rolle |
| --- | --- |
| `run` | output_root, Fortsetzung, Bereinigung, Exporteinstellungen |
| `collection` | LLM/VLM-Modell, max_attempts, Zeitlimit, VLM-Bewertungsnutzung |
| `hf` | HuggingFace Hub Upload-Einstellungen (optional) |
| `tasks` | Aufgabenliste (YAML-Pfad, Ziel-Demo-Anzahl, Aktiviert-Flag) |

### Wichtige Optionen

| Option | Beschreibung |
| --- | --- |
| `--resume` | Zuvor erfolgreiche Aufgaben ueberspringen und nur fehlgeschlagene/unvollstaendige erneut ausfuehren |
| `-q`, `--quiet` | INFO-Level-Protokollierung (Standard: DEBUG) |

### Ausgabestruktur

```text
<output_root>/
├── config_snapshot.yaml         # Kopie der verwendeten Ausfuehrungskonfiguration
├── task_runs/                   # Sammlungsergebnisse pro Aufgabe
├── task_reports/                # JSON-Berichte pro Aufgabe
├── successful_raw/              # Rohdatensatz nur erfolgreicher Episoden
├── exported/                    # Exportergebnisse
├── preprocessed/                # Vorverarbeitungsergebnisse
├── lerobot/                     # LeRobot-Konvertierungsergebnisse
├── batch_report.md              # Markdown-Zusammenfassungsbericht
└── batch_report.json            # JSON-Bericht
```

## Verwandte Dokumente

- Pipeline-Architektur: `docs/de/architecture.md`
- Installation: `docs/de/getting_started.md`
