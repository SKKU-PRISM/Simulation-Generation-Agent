# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT open a public issue.**
2. Email us at **prism.security@skku.edu** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
3. We will acknowledge your report within **48 hours** and aim to release a fix within **7 days** for critical issues.

## Security Best Practices

- Never commit API keys or tokens. Use `.env` files (excluded from git via `.gitignore`).
- When running Docker containers, pass secrets as runtime environment variables (`-e OPENAI_API_KEY=...`), never bake them into images.
- Review `.env.example` for the list of required environment variables.
