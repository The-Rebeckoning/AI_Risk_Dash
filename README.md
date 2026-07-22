# Reported AI Cases Dashboard — Dash

An interactive Dash application for exploring reported AI incidents and hazards from the OECD AI Incidents and Hazards Monitor.

## Features

- National overview of reported AI cases
- Interactive year and stakeholder filters
- Industry and affected-stakeholder trends
- Methodology notes and downloadable OECD source data
- Optional AI-generated related case studies

## Run locally

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the dashboard:

```bash
python dash_app.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

The dashboard works without an OpenAI key using preloaded case-study content. To enable live case-study generation, set the key locally:

```bash
export OPENAI_API_KEY=your_key_here
```

Never commit API keys or local configuration files.

## Production

The Flask server exposed by Dash is available as `dash_app:server`:

```bash
gunicorn dash_app:server
```

## Data source

Data is sourced from the [OECD AI Incidents and Hazards Monitor](https://oecd.ai/en/incidents). Counts represent reported and coded cases, not every real-world AI harm.
