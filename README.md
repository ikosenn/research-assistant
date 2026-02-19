# Research Assistant

## Overview

This project is an attempt at implementing the **STORM** (Synthesis of Topic Outline through Retrieval and Multi-perspective questioning) research approach. It uses multiple AI analyst personas to explore a topic from different angles: each analyst conducts a simulated expert interview, writes a memo, and a final report synthesizes their findings into a single research document with citations.

## How to set up

1. **Clone the repository** (if you haven’t already):
   ```bash
   git clone <repository-url>
   cd research-assistant
   ```

2. **Create a virtual environment and install dependencies**
   Using [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync
   ```

3. **Configure environment variables**
   Copy the example env file and add your API keys:
   ```bash
   cp .example.env .env
   ```
   Edit `.env` and set the keys you need. They are all required

4. **Run the research pipeline**
   From the project root:
   ```bash
   uv run python main.py
   ```
   Or with an active venv:
   ```bash
   python main.py
   ```
   You’ll be prompted for the research topic, number of analysts, and maximum interview turns; the script will then generate analysts, run interviews, and produce a final report.
