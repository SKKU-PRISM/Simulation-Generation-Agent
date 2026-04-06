# 使用方法
[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../usage.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](../ko/usage.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](../zh/usage.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](usage.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](../de/usage.md)

対象読者: 実際にCLIを実行するユーザー
本ドキュメントの内容: 代表的なコマンド、主要オプション、出力構造、結果の解釈
インストールと環境設定: `docs/getting_started.md`

本ドキュメントは**実行方法と結果の解釈に関する正式なリファレンス**です。内部実装の詳細やスキーマの背景は他のドキュメントに分離されています。
パイプラインアーキテクチャ: [docs/architecture.md](architecture.md)

## インタラクティブTUIモード

引数なしで `run_agent.sh` を起動すると、インタラクティブターミナルダッシュボードが開きます：

```bash
./run_agent.sh
```

TUIの機能：
- **タスク入力** — 自然言語のタスク記述を入力してEnterでパイプライン全体を実行
- **設定 (F1)** — パイプラインモード、LLMプロバイダー/モデル、ロボット、エピソード数、パス、詳細オプションの設定
- **実行履歴 (F2)** — 矢印キーで過去の実行を閲覧、Enterで詳細結果（環境スコア、データセット情報）を表示
- **ヘルプ (F3)** — キーボードショートカットとスラッシュコマンド

スラッシュコマンド：`/config`、`/history`、`/help`、`/quit`

Dockerではインタラクティブモードに `-it` を使用：
```bash
docker run -it --rm --gpus all \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  --env-file .env \
  simgen-agent
```

---

## run_agent.sh — フルパイプライン実行（CLIモード）

自然言語のタスク記述を引数として指定し、非インタラクティブで実行：

```bash
# 自然言語入力
./run_agent.sh "Stack the blocks inside the tray on the table"
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5

# JSON入力
./run_agent.sh data/input_sample.json results/output.json
```

### オプション

| オプション | 説明 |
| --- | --- |
| `--robot <type>` | ロボットタイプ: franka, ur10e, openarm, so101（デフォルト: franka） |
| `--episodes <n>` | 成功エピソードの目標数（デフォルト: 1） |
| `--max-attempts <n>` | 最大試行回数（デフォルト: 3） |

### Dockerでの実行

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --episodes 5
```

> **出力パス**: Docker内では結果は`/workspace/artifacts`に保存されます。`-v`オプションでホストにマッピングします。ローカル実行の場合は`outputs/`に保存されます。

### 結果

結果は構造化JSONとして`results/output.json`に記録されます:

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

## 0. Task Spec Agent（NL → YAML）

自然言語のタスク記述を構造化YAMLタスクスペックに変換します（パイプラインステージ1）。

> **注意**: `run_agent.sh "自然言語タスク"`を使用すると、ステージ1→2→3の全パイプラインが自動的に実行されます。
> 以下はステージ1を個別に実行する方法です。

### 代表的なコマンド

```bash
# 基本的な使い方
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube and place it on the target" \
  --robot franka --output task.yaml

# RAGなしのテンプレートベース生成
python3 scripts/task_spec_agent/task_spec_agent.py "Stack blocks" --no-rag

# 別のLLMプロバイダーを使用
python3 scripts/task_spec_agent/task_spec_agent.py "Sort the colored blocks into matching colored bins" --provider huggingface

# 詳細ログ
python3 scripts/task_spec_agent/task_spec_agent.py "Reach the goal" --verbose
```

### 主要オプション

| オプション | 説明 |
| --- | --- |
| `--robot {franka,openarm,ur10,so101}` | 対象ロボット（デフォルト: franka） |
| `--output <path>` | YAML保存パス（未指定の場合はstdoutに出力） |
| `--provider {azure,huggingface,bedrock}` | LLMプロバイダーのオーバーライド（未指定の場合は設定デフォルトを使用、デフォルト=openai） |
| `--no-rag` | RAGの代わりにテンプレートベースYAML生成 |
| `--verbose` | DEBUGレベルログ |

### 内部処理ステップ

```
自然言語入力
  → NLパーサー（アクション、オブジェクト、位置を抽出）
  → タスク分解器（アトミックアクション列に分解）
  → 実現可能性バリデータ（ロボットの物理的実現可能性を検証）
  → RAG YAML生成器（FAISSベクトル検索 → 最近傍タスクYAMLマッチング）
  → task.yaml
```

## 1. IsaacLabパイプライン（ステージ2）

> **run_agent.sh経由での実行**: `./run_agent.sh --mode isaac-lab --task <yaml>`
> 以下はPythonスクリプトを直接実行する方法です。

### 代表的なコマンド

```bash
# 生成 + 実行
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml

# 生成のみ
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run

# 生成 + 実行 + 評価
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate

# 既存出力の再評価
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml \
  --eval-only outputs/isaaclab/frankastack_20260219_160916

# 単独エバリュエータ
python3 scripts/evaluate.py \
  outputs/isaaclab/frankastack_20260219_160916 \
  tasks/franka/stack/franka_stack.yaml

# バッチ
python3 scripts/run_isaac_lab.py --batch tasks/franka/
```

### 主要オプション

| オプション | 説明 |
| --- | --- |
| `--dry-run` | コード生成のみ、IsaacLab実行をスキップ |
| `--evaluate` | 成功後にエバリュエータを実行 |
| `--eval-only <dir>` | 既存の生成出力のみを評価 |
| `--batch <dir>` | ディレクトリ内の全YAMLをバッチ処理 |
| `--output-dir <dir>` | デフォルトの`outputs/isaaclab`の代わりに別の出力ルートを使用 |
| `--config <path>` | エージェント設定のオーバーライド |

### 出力構造

```text
outputs/isaaclab/<task_slug>_<timestamp>/
├── env_cfg.py
├── run_env.py
├── mdp/
├── .success_marker
├── error_attempt_*.txt
├── debug/                    # 環境スクリーンショット（front/top/wrist）
├── result.json               # 実行結果 + scene_verificationを含む
└── eval_report.json          # --evaluate使用時
```

## 2. Isaac Simパイプライン（ローカル開発環境のみ）

> **注意**: Isaac Sim MCPビジュアル検証はIsaac Simデスクトップがローカルで実行されている場合のみ利用可能です。Docker環境ではサポートされていません。

### 代表的なコマンド

```bash
# 自動ビルド + スクリーンショット + VLM評価
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --backend auto

# VLMなしでビルド + キャプチャを1回
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --skip-vlm

# イテレーション/閾値のオーバーライド
python3 scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml \
  --backend azure --max-iterations 3 --threshold 85
```

### 主要オプション

| オプション | 説明 |
| --- | --- |
| `--skip-vlm` | VLM評価なしでビルド/キャプチャを1回実行 |
| `--backend {auto,azure,claude,gemini,ollama,mock}` | VLMバックエンド選択 |
| `--max-iterations <n>` | 最大イテレーション回数のオーバーライド |
| `--threshold <n>` | 成功閾値スコアのオーバーライド |
| `--output-dir <dir>` | 出力ルートのオーバーライド |
| `--config <path>` | `configs/pipeline_config.yaml`の代替 |

### 出力構造

```text
outputs/isaac_sim/<task_slug>_<timestamp>/
├── iter_01.png
├── iter_02.png
├── rgb_0000.png
├── metadata.txt
└── run_report.json
```

## 3. データ収集（ステージ3）

> **run_agent.sh経由での実行**: `./run_agent.sh --mode data-collection --task <yaml>`
> 以下はPythonスクリプトを直接実行する方法です。

### 代表的なコマンド

```bash
# 環境を自動生成してから収集
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml

# 既存環境を使用
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --env-dir outputs/isaaclab/frankastack_20260219_160916

# 成功エピソード目標
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
  --target-success 5 --max-attempts 25

# VLMなしの幾何学検証のみ
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --no-vlm-judge

# バッチ
python3 scripts/run_data_collection.py --batch tasks/franka/ --episodes 10
```

### 主要オプション

| オプション | 説明 |
| --- | --- |
| `--env-dir <path>` | 既存のIsaacLab出力ディレクトリを使用 |
| `--episodes <n>` | 最大エピソード数のオーバーライド |
| `--target-success <n>` | 成功エピソードの目標数 |
| `--max-attempts <n>` | 総試行回数の上限 |
| `--repo-id <id>` | データセットID |
| `--fps <n>` | 記録FPSのオーバーライド |
| `--no-vlm-judge` | VLM判定を無効化 |
| `--gui` | ヘッドレスの代わりにGUIで実行 |
| `--config <path>` | データ収集設定のオーバーライド |
| `-v`, `--verbose` | 詳細ログ |

### 出力構造

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
└── <repo_id>/                   # 任意のLeRobot変換出力
```

### 結果の解釈

`collection_results.json`で最初に確認すべき値は以下の通りです。

| キー | 意味 |
| --- | --- |
| `pipeline_completed` | 収集/クリーンアップパイプラインが最後まで完了したかどうか |
| `target_met` | 成功エピソードの目標数が達成されたかどうか |
| `successful_episodes` | 成功エピソード数 |
| `total_episodes` | 実際に試行/保存されたエピソード数 |
| `raw_dataset` | 生データセットパス |

解釈ルール:

- `success`は現在`pipeline_completed`のエイリアスです。
- `pipeline_completed=true`は必ずしも`target_met=true`を意味しません。
- デフォルトの完了基準は`raw_dataset/`の作成です。LeRobot変換は環境によってスキップされる場合があります。

## 4. データセットエクスポート / 前処理

### エクスポート

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets
```

デフォルトのスキーマは`adc_compatible`です。

### 前処理

```bash
python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

出力:

- `manifest.json`
- `samples.jsonl`
- `train.jsonl`
- `val.jsonl`

デフォルトの`adc_compatible`パスでは、`observation.gripper_state`、`observation.tcp.robot_xyzrpy`、`skill.goal_position.robot_xyzrpy`が学習入力マニフェストに含まれます。

デフォルトエクスポートは`adc_compatible`スキーマを使用し、`canonical_training`は拡張分析用です。

### 任意: ローカルLeRobot変換

```bash
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --repo-id local/franka_stack_sim
python3 scripts/check_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/franka_stack_sim \
  --repo-id local/franka_stack_sim
```

ホストのPython環境に`lerobot`パッケージがインストールされている必要があります。

### 任意: Hubへの公開

```bash
python3 scripts/publish_lerobot_dataset.py \
  outputs/data_collection/<run_dir>/local/franka_stack_sim \
  --repo-id <org>/<dataset_name> \
  --local-repo-id local/franka_stack_sim \
  --private
```

デフォルトの認証は`HF_TOKEN`環境変数または既存の`huggingface-cli login`セッションを使用します。

## 5. 推奨ワークフロー順序

### NL → データセット（フルパイプライン）

```bash
# 1. NL → YAML
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube and stack it" \
  --robot franka --output outputs/generated_task.yaml

# 2. YAML → IsaacLab環境コード
python3 scripts/run_isaac_lab.py outputs/generated_task.yaml --evaluate

# 3. データ収集
python3 scripts/run_data_collection.py outputs/generated_task.yaml \
  --env-dir outputs/isaaclab/<run_dir> --target-success 10

# 4. エクスポート / 前処理
python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

### 既存YAML → データセット

```bash
python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --env-dir outputs/isaaclab/<run_dir>
python3 scripts/export_dataset.py outputs/data_collection/<run_dir>/raw_dataset --source-type sim_raw --output-dir outputs/exported_datasets
python3 scripts/preprocess_dataset.py outputs/exported_datasets/<export_dir> --output-dir outputs/preprocessed_datasets
```

ビジュアル検証が必要な場合のみ`scripts/run_isaac_sim.py`を追加します。

## 6. フルパイプライン（NL → ビデオ）

`run_full_test.sh`は13のFrankaタスクに対してステージ1→2→3を順次実行する統合スクリプトです。

```bash
bash scripts/run_full_test.sh
```

内部的には各タスクに対して:
1. `task_spec_agent.py` -- 自然言語 → YAML生成
2. `run_isaac_lab.py` -- YAML → IsaacLab環境コード生成/検証
3. `run_data_collection.py` -- CaPコード生成 → 実行 → 成功判定 → データ収集

### 出力構造

```text
outputs/test_run_<timestamp>/
├── summary.txt                  # 全体タスク結果サマリー
├── token_usage.jsonl            # APIトークン使用量（TOKEN_USAGE_FILE設定時）
├── FrankaLift/
│   ├── step1_nl_to_yaml.log
│   ├── step2_isaaclab.log
│   ├── step3_cap_execution.log
│   └── task.yaml
├── FrankaStack/
│   └── ...
└── ...
```

### トークン使用量追跡

```bash
export TOKEN_USAGE_FILE=outputs/token_usage.jsonl
export TOKEN_USAGE_LOG=1  # リアルタイムコンソールログ
bash scripts/run_full_test.sh
```

## 7. E2Eバッチパイプライン

設定ファイルに基づいて、複数タスクの環境生成 + データ収集 + エクスポート + LeRobot変換をバッチ処理します。

### 代表的なコマンド

```bash
# バッチ実行（Docker）
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml

# 中断後の再開
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml --resume

# 静かなログ
python3 scripts/run_e2e_batch.py configs/docker/e2e_batch_release.yaml -q
```

### Docker / run_agent.sh

```bash
# Docker内（デフォルトモード: e2e-batch）
run_agent.sh --mode e2e-batch --resume

# 単一タスク
run_agent.sh --mode isaac-lab --task tasks/franka/lift/franka_lift.yaml

# データ収集
run_agent.sh --mode data-collection --task tasks/franka/lift/franka_lift.yaml -- --episodes 10
```

### 設定構造

E2Eバッチ設定YAMLは4つのセクションで構成されます:

| セクション | 役割 |
| --- | --- |
| `run` | output_root、resume、cleanup、エクスポート設定 |
| `collection` | LLM/VLMモデル、max_attempts、timeout、VLMジャッジ使用 |
| `hf` | HuggingFace Hubアップロード設定（任意） |
| `tasks` | タスクリスト（YAMLパス、目標デモ数、有効フラグ） |

### 主要オプション

| オプション | 説明 |
| --- | --- |
| `--resume` | 以前に成功したタスクをスキップし、失敗/未完了のタスクのみ再実行 |
| `-q`, `--quiet` | INFOレベルログ（デフォルト: DEBUG） |

### 出力構造

```text
<output_root>/
├── config_snapshot.yaml         # 実行に使用した設定のコピー
├── task_runs/                   # タスクごとの収集結果
├── task_reports/                # タスクごとのJSONレポート
├── successful_raw/              # 成功エピソードのみの生データセット
├── exported/                    # エクスポート結果
├── preprocessed/                # 前処理結果
├── lerobot/                     # LeRobot変換結果
├── batch_report.md              # Markdownサマリーレポート
└── batch_report.json            # JSONレポート
```

## 関連ドキュメント

- パイプラインアーキテクチャ: `docs/architecture.md`
- インストール: `docs/getting_started.md`
