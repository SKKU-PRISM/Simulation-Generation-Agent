# Docker ガイド

[<img src="https://flagcdn.com/24x18/us.png" width="20" alt="English"> English](../../README.docker.md) | [<img src="https://flagcdn.com/24x18/kr.png" width="20" alt="한국어"> 한국어](../ko/docker.md) | [<img src="https://flagcdn.com/24x18/cn.png" width="20" alt="中文"> 中文](../zh/docker.md) | [<img src="https://flagcdn.com/24x18/jp.png" width="20" alt="日本語"> 日本語](docker.md) | [<img src="https://flagcdn.com/24x18/de.png" width="20" alt="Deutsch"> Deutsch](../de/docker.md)

このガイドでは、Simulation-Generation-Agent を Docker コンテナ内でビルドして実行する方法をステップバイステップで説明します。ローカルへの IsaacLab のインストールは不要です。

---

## 前提条件

開始する前に、以下が準備されていることを確認してください：

- [ ] 最新ドライバがインストールされた **NVIDIA GPU**
- [ ] **Docker Engine**（v26.0+）
- [ ] **NVIDIA Container Toolkit**
- [ ] **OpenAI API キー**（または Azure OpenAI の認証情報）

### システム要件

| コンポーネント | 最小 | 推奨 |
|----------------|------|------|
| GPU メモリ | 8 GB | 16+ GB |
| ディスク容量 | 50 GB 空き | 100+ GB 空き |
| RAM | 16 GB | 32+ GB |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |

### Docker Engine のインストール

公式ガイドに従ってください：[Ubuntu に Docker Engine をインストール](https://docs.docker.com/engine/install/ubuntu/)

インストール後に確認：
```bash
docker --version
docker info
```

### NVIDIA Container Toolkit のインストール

公式ガイドに従ってください：[NVIDIA Container Toolkit のインストール](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

インストール後、Docker から GPU にアクセスできることを確認：
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 nvidia-smi
```

GPU の一覧が表示されるはずです。権限エラーが出た場合は、`newgrp docker` を実行するか、ログアウトして再ログインしてください。

---

## ステップ 1：ソースコードの取得

```bash
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent
```

> `--recurse-submodules` なしで既にクローンした場合は、以下を実行してください：
> ```bash
> git submodule update --init --recursive
> ```

---

## ステップ 2：API キーの設定

```bash
cp .env.example .env
```

`.env` を開いて API キーを設定します（ローカル実行用）：
```
OPENAI_API_KEY=sk-your-key-here
```

> **Docker ユーザー**：Docker 実行時に `.env` ファイルを作成する必要はありません。API キーは実行時に `-e` フラグで直接渡されます。`.env` ファイルはローカル（非 Docker）実行時にのみ必要です。

---

## ステップ 3：Docker イメージのビルド

このリポジトリには、完全な環境を自動的にセットアップする `Dockerfile` が含まれています。

```bash
docker build -t simgen-agent .
```

> **ディスク容量**：約 50GB 必要。**初回ビルド**：約 30〜60 分。Docker レイヤーキャッシュにより、以降のビルドはずっと高速です。

### Dockerfile の内容

リポジトリルートにある `Dockerfile` は、以下を含む自己完結型のイメージをビルドします：

| Layer | What's installed |
|-------|-----------------|
| Base | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| Python | 3.11 (venv at `/opt/isaaclab-env`) |
| Isaac Sim | 5.1.0 (pip from `pypi.nvidia.com`) |
| IsaacLab | v2.3.2 (source build from GitHub) |
| PyTorch | 2.7.0 (CUDA 12.8) |
| Project | `requirements.txt` + all source code |
| Entry point | `ENTRYPOINT ["./run_agent.sh"]` |

コンテナのエントリポイントは `run_agent.sh` です。したがって、以下のように実行すると：
```bash
docker run simgen-agent "Stack the blocks..."
```
コンテナ内部で自動的に `run_agent.sh "Stack the blocks..."` が実行されます。

イメージがビルドされたことを確認：
```bash
docker images | grep simgen-agent
```

---

## ステップ 4：パイプラインの実行

### オプション A：自然言語入力（最もシンプル）

ロボットが実行すべき操作を記述するだけです：

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key-here" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

オプション付き：
```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -v $(pwd)/results:/workspace/Simulation-Generation-Agent/results \
  -e OPENAI_API_KEY="your-key-here" \
  simgen-agent \
  "Stack the blocks inside the tray on the table" --robot franka --episodes 5
```

### オプション B：JSON 入力

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  data/input_sample.json results/output.json
```

デフォルトの `data/input_sample.json` にはサンプルの FrankaStackTray タスクが含まれています。独自に作成することもできます：
```json
{
  "tasks": [
    {"task_description": "Pick up the cube and place it on the target", "robot": "franka"}
  ],
  "config": {"target_success": 1, "max_attempts": 3}
}
```

### オプション C：個別ステージ（上級）

```bash
# Stage 2 のみ：IsaacLab 環境コードの生成
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode isaac-lab --task tasks/franka/stack/franka_stack.yaml

# Stage 3 のみ：データ収集
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode data-collection --task tasks/franka/stack/franka_stack.yaml

# バッチ実行（複数タスク）
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  simgen-agent \
  --mode e2e-batch --config configs/docker/e2e_batch_smoke.yaml
```

### Azure OpenAI を使用する

OpenAI プラットフォームの代わりに Azure OpenAI を使用するには、`-e` フラグで Azure 資格情報を渡します：

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

| 変数 | 必須 | 説明 |
|------|------|------|
| `AZURE_OPENAI_API_KEY` | はい | Azure OpenAI API キー |
| `AZURE_OPENAI_BASE_URL` | はい | Azure エンドポイント（`/openai/v1/` サフィックス付き） |
| `AZURE_OPENAI_ENDPOINT` | はい | Azure リソースエンドポイント（Stage 1 で使用） |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | はい | Azure にデプロイされたモデル名（Stage 1 で使用） |

> `OPENAI_API_KEY` または上記の Azure 変数セットのいずれか一方のみ必要です。

### モデルの選択

デフォルトモデルは `gpt-5-mini` です。`OPENAI_MODEL` 環境変数で変更できます：

```bash
docker run --rm --gpus all \
  -v $(pwd)/artifacts:/workspace/artifacts \
  -v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs \
  -e OPENAI_API_KEY="your-key" \
  -e OPENAI_MODEL="gpt-4o" \
  simgen-agent \
  "Stack the blocks inside the tray on the table"
```

### ヘルプの表示

```bash
docker run --rm simgen-agent --help
```

### 実行時の表示内容

パイプラインが実行されると、ターミナルに分かりやすい進捗状況が表示されます：

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

各ステージの実行中はスピナーインジケーターが表示されます。詳細ログはファイルに保存され、ターミナルにはクリーンなステータス行のみが表示されます。

---

## ステップ 5：結果の確認

結果はホストマシンの `./artifacts/` に保存されます（コンテナ内部の `/workspace/artifacts` からマッピング）。

```bash
ls artifacts/
```

一般的な出力構造：
```
artifacts/
├── isaaclab/              # Stage 2：生成された環境コード
│   └── 20260402_*/        # タイムスタンプ付き実行ディレクトリ
│       ├── env_cfg.py     # 環境設定
│       ├── run_env.py     # 環境ランナー
│       ├── result.json    # 成功/失敗ステータス
│       └── debug/         # スクリーンショット（前面、上面、手首）
└── data_collection/       # Stage 3：収集されたエピソード
    └── TaskName_*/
        ├── collection_results.json
        ├── raw_dataset/   # エピソードデータ
        └── videos/        # 録画された動画
```

JSON 入力モードの場合、結果は `results/output.json` にも書き出されます。

---

## トラブルシューティング

### GPU が検出されない

```
Error: could not select device driver "nvidia"
```

**解決方法**：NVIDIA Container Toolkit をインストールまたは再インストールしてください：
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### IsaacSim のインストール中にビルドが失敗する

IsaacSim の pip パッケージは約 15GB です。ネットワークの問題で失敗した場合：
```bash
# キャッシュなしで再試行
docker build --no-cache -t simgen-agent .
```

### メモリ不足 (OOM)

IsaacLab シミュレーションには最低 **8GB の GPU メモリ**（Stage 2 のみ）または **16GB+**（Stage 3 を含むフルパイプライン）が必要です。OOM エラーが発生した場合：
- 他の GPU 集約型アプリケーションを終了してください
- `--episodes` を 1 に減らしてください
- `--mode isaac-lab` で先に Stage 2 だけをテストしてください

### API キーが機能しない

```
Either OPENAI_API_KEY or AZURE_OPENAI_API_KEY must be set.
```

**解決方法**：キーが `-e` で正しく渡されていることを確認してください：
```bash
# 方法 1：インライン
-e OPENAI_API_KEY="sk-your-key"

# 方法 2：.env ファイルから
-e OPENAI_API_KEY="$(grep OPENAI_API_KEY .env | cut -d= -f2)"

# 方法 3：先に export
export OPENAI_API_KEY="sk-your-key"
docker run ... -e OPENAI_API_KEY ...
```

---

## リファレンス

### 環境変数

| 変数 | 必須 | 説明 |
|------|------|------|
| `OPENAI_API_KEY` | はい（または Azure） | OpenAI プラットフォーム API キー |
| `AZURE_OPENAI_API_KEY` | はい（または OpenAI） | Azure OpenAI API キー |
| `AZURE_OPENAI_BASE_URL` | Azure 使用時 | Azure エンドポイント URL |
| `OPENAI_BASE_URL` | いいえ | カスタム OpenAI 互換エンドポイント |
| `HF_TOKEN` | いいえ | HuggingFace トークン（データセットアップロード用） |
| `OPENAI_MODEL` | いいえ | デフォルトモデルの上書き（デフォルト：gpt-5-mini） |

### `run_agent.sh` モード

| モード | 説明 |
|--------|------|
| *（位置引数）* | フルパイプライン：自然言語または JSON 入力 |
| `--mode e2e-batch` | 複数タスクのバッチ実行 |
| `--mode isaac-lab` | Stage 2 のみ：環境コード生成 |
| `--mode data-collection` | Stage 3 のみ：データ収集 |

### イメージ仕様

| コンポーネント | バージョン |
|----------------|------------|
| ベースイメージ | `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` |
| Python | 3.11 |
| Isaac Sim | 5.1.0 (pip) |
| IsaacLab | v2.3.2 (source) |
| PyTorch | 2.7.0 (CUDA 12.8) |

### 検証スクリプト

```bash
# 静的監査（シークレット、パス、構造）
python3 scripts/audit_release_repo.py

# 完全な Docker 検証（ビルド + 実行 + テスト）
python3 scripts/validate_docker_release.py

# 完全なソークテスト（全タスク）
python3 scripts/validate_docker_release.py --run-full-soak
```
