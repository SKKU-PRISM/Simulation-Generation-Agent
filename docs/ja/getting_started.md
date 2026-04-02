# インストールガイド

🇺🇸 [English](../getting_started.md) | 🇰🇷 [한국어](../ko/getting_started.md) | 🇨🇳 [中文](../zh/getting_started.md) | 🇯🇵 [日本語](getting_started.md) | 🇩🇪 [Deutsch](../de/getting_started.md)

対象読者: このリポジトリを初めて実行するユーザー
CLIオプションの詳細と結果の解釈については: `docs/usage.md`

## 1. 共通インストール

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent

pip install -e .
cp .env.example .env
```

### サブモジュール（AutoDataCollector）

このプロジェクトはgitサブモジュールとして`external/AutoDataCollector`を含んでいます。データ収集（ステージ3）でジャッジプロンプトやIKユーティリティを参照する際に使用されます。

`--recurse-submodules`付きでクローンすると自動的に取得されます。取得できなかった場合は、手動で初期化してください:

```bash
git submodule update --init --recursive
ls external/AutoDataCollector/  # ファイルが存在するはずです
```

必須の`.env`:

```bash
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1/
```

オプション設定:

```bash
# VLMバックエンド（環境検証 + エピソード成功判定用）
# ANTHROPIC_API_KEY=...
# GOOGLE_API_KEY=...

# パス
# ISAACLAB_PATH=~/workspace/IsaacLab

# トークン使用量追跡
# TOKEN_USAGE_FILE=outputs/token_usage.jsonl
```

## 2. IsaacLabインストール（メインパス）

```bash
cd ~/workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

プロジェクトにIsaacLabの場所を指定する方法:

```bash
export ISAACLAB_PATH=~/workspace/IsaacLab
```

または`configs/isaaclab_agent_config.yaml`で`isaaclab.path`を直接設定します。

接続を確認:

```bash
# ローカル（conda環境）
conda run -n env_isaaclab --no-capture-output \
  python -c "import isaaclab; print('IsaacLab import OK')"

# Docker内ではcondaの代わりにvenvを使用するため、以下で確認
# python -c "import isaaclab; print('IsaacLab import OK')"

python3 scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run
```

## 3. Isaac Sim + MCPインストール（任意）

Isaac Simはビジュアル検証パスでのみ必要です。

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

正常に起動すると、`localhost:8766`でTCPソケットがリスニングします。

接続を確認:

```bash
python3 tests/test_components.py connection
```

## 4. データ収集インストール（任意）

データ収集はIsaacLab上で動作します。

```bash
pip install -e ".[data-collection]"
pip install pin
```

確認:

```bash
python3 -c "from src.agent.data_collection.adc_imports import is_adc_available; print(is_adc_available())"
python3 -c "import pinocchio; print('Pinocchio OK')"
python3 scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --help
```

## 5. Task Spec Agentインストール（NL -> YAML）

Task Spec Agentは自然言語入力をYAMLタスクスペックに変換するステージ1パイプラインです。

```bash
pip install langchain langchain-openai langchain-community faiss-cpu sentence-transformers
```

ベクトルストアは初回実行時に`data/vector_store/`に自動作成されます（tasks/ディレクトリの既存YAMLをインデキシング）。

確認:

```bash
cd scripts/task_spec_agent
python3 -c "from rag_match_yaml_generator import RAGYAMLGenerator; print('RAG OK')"
cd ../..
```

単独実行:

```bash
python3 scripts/task_spec_agent/task_spec_agent.py "Pick up the cube" --robot franka --output outputs/test_task.yaml
```

## 6. 初回実行の推奨手順

### 最も簡単な実行 -- 自然言語を1行入力するだけ

```bash
./run_agent.sh "Stack the blocks inside the tray on the table"
```

自然言語のタスク記述を入力するだけで、NL→YAML→IsaacLab→DataCollectionのパイプライン全体が自動的に実行されます。

실행 시 터미널에 깔끔한 진행 상태가 표시됩니다:
```
🚀 RAPIDS Pipeline — "Stack the blocks inside the tray on the table"
   Robot: franka | Target: 1 episodes

  ✅ Stage 1: NL → YAML                              1m 12s
  ✅ Stage 2: YAML → IsaacLab                         9m 44s
  ✅ Stage 3: Data Collection (1/1 episodes)           7m 30s

──────────────────────────────────────────────────────
  📊 Result: ✅ completed
  ⏱️  Total: 18m 26s
  🔤 Tokens: 123,008 (10 API calls)
  💰 Cost: ~$0.15
  📄 Output: results/output.json
──────────────────────────────────────────────────────
```



실행 시 터미널에 진행 상태가 표시됩니다:
オプション:
```bash
./run_agent.sh "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

### フルパイプライン（13タスクのバッチ）

```bash
bash scripts/run_full_test.sh
```

13のFrankaタスクに対して全3ステージを順次実行します:
1. **NL -> YAML**: `task_spec_agent`が自然言語をYAMLに変換
2. **YAML -> IsaacLab**: LLMが環境Pythonコードを生成し実行を検証
3. **CaP -> データ**: LLMがスキルコードを生成、IKで実行、成功を判定、データを収集

### IsaacLabのみ（ステージ2のみ）

```bash
./run_agent.sh --mode isaac-lab --task tasks/franka/stack/franka_stack.yaml
```

成功基準:
- `outputs/isaaclab/<run_dir>/env_cfg.py`
- `outputs/isaaclab/<run_dir>/result.json`

### データ収集のみ（ステージ3のみ）

```bash
./run_agent.sh --mode data-collection --task tasks/franka/stack/franka_stack.yaml
```

成功基準:
- `outputs/data_collection/<run_dir>/collection_results.json`
- `outputs/data_collection/<run_dir>/raw_dataset/`

### 学習準備パス

```bash
python3 scripts/export_dataset.py \
  outputs/data_collection/<run_dir>/raw_dataset \
  --source-type sim_raw \
  --output-dir outputs/exported_datasets

python3 scripts/preprocess_dataset.py \
  outputs/exported_datasets/<export_dir> \
  --output-dir outputs/preprocessed_datasets
```

## 7. Dockerでの実行

ローカル環境の構築なしに、Dockerコンテナを使用してすぐに実行できます。

前提条件: [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
# サブモジュール初期化 + イメージビルド
git submodule update --init --recursive
docker build -t simgen-agent .

# 自然言語入力でフルパイプラインを実行
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5

# JSON入力
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -e OPENAI_API_KEY="your-key" \
  simgen-agent \
  data/input_sample.json results/output.json

# ヘルプの表示
docker run --rm simgen-agent --help
```

> **出力パスの違い**: Docker内では結果は`/workspace/artifacts`に保存されます（ホスト上の`./artifacts/`にマウント）。ローカルでは`outputs/`に保存されます。Docker内でローカルと同じパスを使用するには、`-e SIMGEN_ARTIFACT_ROOT=/workspace/Simulation-Generation-Agent/outputs`を追加してください。

詳細なDockerガイド: [README.docker.md](../README.docker.md)

## 8. 次に読むもの

- パイプラインアーキテクチャの詳細: [docs/architecture.md](architecture.md)
- CLIオプション / 出力構造 / 結果の解釈: [docs/usage.md](usage.md)
