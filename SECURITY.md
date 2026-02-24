# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in RustyStats, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email security concerns to the maintainers via the contact information in the repository
3. Include a description of the vulnerability and steps to reproduce
4. Allow reasonable time for a fix before public disclosure

We aim to acknowledge reports within 48 hours and provide a fix within 7 days for critical issues.

## Scope

RustyStats is a statistical modeling library. Security concerns most relevant to this project include:

- **Deserialization safety**: `GLMModel.from_bytes()` uses pickle internally. Only load models from trusted sources.
- **Numerical stability**: Inputs that cause panics or undefined behavior in the Rust core.
- **Dependency vulnerabilities**: Issues in upstream crates or Python packages.
