# Security Policy

## Supported Versions

This project is released as a single version for the Google DeepMind Nano Banana Hackathon (2025). There are no versioned releases with separate support windows.

## Reporting a Vulnerability

If you discover a security vulnerability in Inspira, please report it responsibly.

**Do not** open a public GitHub issue for security concerns.

Instead, contact the maintainer directly:

- **GitHub**: https://github.com/sai009
- **Kaggle**: https://www.kaggle.com/wavesandpixelsfilms

### What to Include

- A description of the vulnerability
- Steps to reproduce it
- The affected file(s) and line number(s)
- Any suggested fix or mitigation

### What to Expect

- We will acknowledge your report within 7 days
- We will investigate and provide a status update within 14 days
- If a fix is needed, we will release it and credit you (unless you prefer anonymity)

## Security Best Practices for Users

1. **Never commit API keys.** Use a `.env` file (already in `.gitignore`) for your `GOOGLE_API_KEY` and `ELEVENLABS_API_KEY`.
2. **Review dependencies.** This project uses `gradio`, `Pillow`, `google-generativeai`, `requests`, and `python-dotenv`. Keep them updated.
3. **Run from a virtual environment.** Isolate dependencies with `venv` or `uv`.
4. **Inspect before running.** Read `comprehensive_studio.py` before executing, especially if you modify it.

## Disclaimer

This software is provided "AS IS" under the MIT License. The author is not liable for any damages arising from its use.
