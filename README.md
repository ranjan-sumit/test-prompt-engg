# 🧠 Prompt Engineering Libraries Explorer

An interactive Streamlit demo app covering the **7 most important open-source Python libraries** for prompt engineering in 2026.

## What's Inside

Each library tab has three sections:
- 📖 **Concept** — what makes this library unique, when to use it
- 💻 **Code** — real, copy-paste code examples
- 🎮 **Live Demo** — interactive demo powered by Claude

## Libraries Covered

| Library | What it does |
|---|---|
| 🧠 **DSPy** | Auto-optimize prompts (10–40% improvement) |
| 🔗 **LangChain** | Multi-step chains and agents |
| 📊 **Instructor** | Structured Pydantic outputs, zero parsing |
| 🌐 **LiteLLM** | Unified API for 100+ LLMs |
| 🏗️ **Mirascope** | Type-safe, function-based prompting |
| 🔒 **Promptfoo** | Security testing & red teaming |
| 📈 **Langfuse** | Production monitoring & prompt versioning |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

## Usage

1. Enter your **Anthropic API key** in the sidebar
2. Pick a **library tab** at the top
3. Read the concepts, study the code, run the live demo

## Architecture

All live demos use the **Anthropic SDK** as the underlying engine, simulating what each library does. This means:
- Only `streamlit` + `anthropic` + `pandas` are required
- No need to install heavy dependencies (DSPy, LangChain, etc.)
- Demos show real outputs alongside library-specific code

## Requirements

- Python 3.9+
- Anthropic API key (get one at console.anthropic.com)
