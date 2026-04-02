# データセットワークフロー

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../dataset.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](../ko/dataset.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](../zh/dataset.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](dataset.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](../de/dataset.md)

このガイドでは、パイプラインの実行からトレーニング用データセットのHuggingFaceへのアップロードまでの完全なワークフローについて説明します。

---

## 概要

```
run_agent.sh "task description"
    |
    v
outputs/data_collection/<run>/raw_dataset/     <-- Stage 3 出力
    |
    v  scripts/export_dataset.py
outputs/exported_datasets/<name>/              <-- 正規化されたトレーニングスキーマ
    |
    v  scripts/preprocess_dataset.py
outputs/preprocessed_datasets/<name>/          <-- train.jsonl / val.jsonl
    |
    v  scripts/convert_lerobot_dataset.py
outputs/lerobot_datasets/<repo_id>/            <-- LeRobot v3.0 形式 (Parquet)
    |
    v  scripts/publish_lerobot_dataset.py
https://huggingface.co/datasets/<org>/<name>   <-- HuggingFace Hub
```

---

## ステップ 1: パイプラインの実行

```bash
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

Stage 3 が完了すると、結果は以下に保存されます：
- **ローカル**: `outputs/data_collection/<TaskName>_<timestamp>/`
- **Docker**: `/workspace/artifacts/data_collection/`（`./artifacts/` にマウント）

### 出力の確認

```bash
ls outputs/data_collection/
# FrankaStackTray_20260402_151823/

ls outputs/data_collection/FrankaStackTray_20260402_151823/
# collection_results.json   raw_dataset/   videos/

cat outputs/data_collection/FrankaStackTray_20260402_151823/collection_results.json
```

`collection_results.json` ファイルには以下が含まれます：
- `pipeline_completed`: データ収集が完了したかどうか
- `geometry_successful_episodes`: 幾何学的検証に合格したエピソード数
- `total_episodes`: 試行されたエピソードの合計数

### 生データセットの構造

```
raw_dataset/
├── metadata.json              # ロボット設定、自由度、FPS、カメラ
├── episodes/
│   ├── episode_000000/
│   │   ├── states.npy         # (T, N_dof) 関節位置
│   │   ├── actions.npy        # (T, N_dof) 指令位置
│   │   ├── tcp_world_xyzrpy.npy  # (T, 6) ワールド座標系でのTCP
│   │   ├── tcp_robot_xyzrpy.npy  # (T, 6) ロボットベース座標系でのTCP
│   │   ├── gripper_state.npy  # (T, 1) グリッパー状態
│   │   ├── goal_robot_xyzrpy.npy # (T, 6) フレームごとの目標姿勢
│   │   ├── skills.json        # フレームごとのスキルメタデータ
│   │   └── images/
│   │       ├── top/           # 000000.png, 000001.png, ...
│   │       └── wrist/
│   └── episode_000001/
│       └── ...
```

---

## ステップ 2: エクスポート（オプション）

生データを一貫した関節正規化を持つトレーニング向けスキーマに変換します。

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets
```

オプション：
- `--schema adc_compatible`（デフォルト）: 関節を [-100, 100] 範囲に正規化、ADC互換フィールド名
- `--schema canonical_training`: 明示的なTCP現在位置/目標位置を持つクリーンなスキーマ
- `--no-link-images`: シンボリックリンクの代わりに画像をコピー

---

## ステップ 3: 前処理（オプション）

フレーム単位のトレーニングマニフェストをtrain/val分割で作成します。

```bash
python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets \
  --success-only \
  --train-ratio 0.9
```

出力：
- `train.jsonl` / `val.jsonl` — ファイルパス付きのフレーム単位サンプル
- `manifest.json` — スキーマと統計情報

---

## ステップ 4: LeRobot形式への変換

生データを [LeRobot v3.0](https://github.com/huggingface/lerobot) 形式（Apache Parquet）に変換します。

### 前提条件

```bash
pip install "lerobot>=0.4.0,<0.5.0"
```

### 変換

```bash
python3 scripts/convert_lerobot_dataset.py \
  outputs/data_collection/<run>/raw_dataset \
  --repo-id local/franka_stack_sim \
  --output-root outputs/lerobot_datasets
```

出力構造：
```
outputs/lerobot_datasets/local/franka_stack_sim/
├── meta/
│   └── info.json           # LeRobotメタデータ、特徴量、フレーム数
├── data/
│   └── chunk-0000/
│       └── file-00000.parquet
└── videos/                 # （ビデオ録画が有効な場合）
```

### 検証

```bash
python3 scripts/check_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack_sim \
  --repo-id local/franka_stack_sim
```

検証で確認される項目：
- 必須特徴量: `observation.state`, `observation.gripper_state`, `observation.tcp.robot_xyzrpy`, `action`, `skill.goal_position.robot_xyzrpy`
- Parquetシャードの整合性
- フレーム数 > 0
- `meta/info.json` の存在

---

## ステップ 5: HuggingFaceへのアップロード

### 前提条件

```bash
pip install huggingface_hub
```

HuggingFaceトークンを設定：
```bash
export HF_TOKEN="hf_your_token_here"
# または: huggingface-cli login
```

### 公開

```bash
python3 scripts/publish_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack_sim \
  --repo-id your-org/franka-stack-sim \
  --local-repo-id local/franka_stack_sim \
  --private
```

オプション：
- `--private`: プライベートデータセットとして作成（デフォルト: 公開）
- `--token-env HF_TOKEN`: 認証用の環境変数（デフォルト）
- `--token <token>`: トークンを直接指定

スクリプトは以下を実行します：
1. ローカルデータセットの検証
2. HuggingFaceデータセットリポジトリの作成（存在しない場合）
3. `upload_large_folder()` を使用した全ファイルのアップロード
4. リポジトリURLを含むJSONレポートの出力

---

## クイックリファレンス

### オールインワン（生データからHuggingFaceまで）

```bash
# 1. パイプライン実行
./run_agent.sh "Stack the blocks" --robot franka --episodes 5

# 2. 出力の確認
RUN_DIR=$(ls -td outputs/data_collection/*/ | head -1)

# 3. LeRobot形式に変換
python3 scripts/convert_lerobot_dataset.py \
  ${RUN_DIR}/raw_dataset \
  --repo-id local/franka_stack \
  --output-root outputs/lerobot_datasets

# 4. 検証
python3 scripts/check_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack \
  --repo-id local/franka_stack

# 5. アップロード
python3 scripts/publish_lerobot_dataset.py \
  outputs/lerobot_datasets/local/franka_stack \
  --repo-id your-org/franka-stack --private
```

### バッチパイプライン（自動化）

`e2e-batch` モードは、LeRobot変換とHuggingFaceアップロードを含む全フローを自動化できます：

```bash
./run_agent.sh --mode e2e-batch --config configs/docker/e2e_batch_release.yaml
```

バッチ設定YAMLでアップロードを構成：
```yaml
hf:
  upload: true
  namespace: your-org
  dataset_name: franka-sim-dataset
  private: true
  token_env: HF_TOKEN
```

---

## LeRobotデータセットの特徴量

変換されたデータセットの各フレームには以下が含まれます：

| 特徴量 | 形状 | 説明 |
|--------|------|------|
| `observation.state` | (N_dof,) | 関節位置 |
| `observation.gripper_state` | (1,) | グリッパー状態 |
| `observation.tcp.world_xyzrpy` | (6,) | ワールド座標系でのTCP姿勢 |
| `observation.tcp.robot_xyzrpy` | (6,) | ロボットベース座標系でのTCP姿勢 |
| `action` | (N_dof,) | 指令関節位置 |
| `skill.natural_language` | (1,) | スキルの説明 |
| `skill.type` | (1,) | スキルの種類（pick, place など） |
| `skill.progress` | (1,) | スキルの進捗 [0, 1] |
| `skill.goal_position.joint` | (N_dof,) | 目標関節位置 |
| `skill.goal_position.world_xyzrpy` | (6,) | ワールド座標系での目標姿勢 |
| `skill.goal_position.robot_xyzrpy` | (6,) | ロボットベース座標系での目標姿勢 |
| `skill.goal_position.gripper` | (1,) | 目標グリッパー状態 |
| `observation.images.{cam}` | (480, 640, 3) | カメラ画像（top, wrist） |

**ロボット別N_dof:** Franka=9, UR10e=12, OpenARM=9, SO-101=6

---

## 関連ドキュメント

- パイプラインアーキテクチャ: [architecture.md](architecture.md)
- CLIの使い方: [usage.md](usage.md)
- インストール: [getting_started.md](getting_started.md)
