# Contributing to Simulation-Generation-Agent

Thank you for your interest in contributing! This project automates robotics simulation environment generation and data collection using LLMs and IsaacLab.

## How to Contribute

- **Bug Reports / Feature Requests**: Open a [GitHub Issue](../../issues)
- **Code Contributions**: Fork the repository and submit a Pull Request

## Development Setup

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo-url>
cd Simulation-Generation-Agent

# 2. Install dependencies
pip install -e .[data-collection]

# 3. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys (OPENAI_API_KEY at minimum)

# 4. Verify installation
python -c "from src.agent.isaac_lab.agent import IsaacLabAgent; print('OK')"
```

IsaacLab is required for full pipeline execution. See [docs/getting_started.md](docs/getting_started.md) for detailed setup instructions.

## Code Style

- Python 3.10+
- Type hints recommended for public APIs
- Follow existing code patterns in each module
- Keep imports organized: stdlib, third-party, local

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes and test locally
3. Ensure no secrets, API keys, or personal paths are included
4. Write a clear PR description explaining the change and motivation
5. Submit the PR for review

## Project Structure

```
src/agent/
  isaac_lab/     # Stage 2: YAML -> IsaacLab environment code
  data_collection/  # Stage 3: Environment -> Data collection
  common/        # Shared utilities (LLM client, token tracking)
scripts/         # CLI entry points
tasks/           # Task YAML definitions
```

See [docs/architecture.md](docs/architecture.md) for the full pipeline architecture.

## License

This project is licensed under the [MIT License](LICENSE). By contributing, you agree that your contributions will be licensed under the same license.
