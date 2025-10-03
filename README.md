# TikTok Early Warning (Storyful MVP)

Minimal prototype:
- Clusters videos into narratives (sound/hashtag)
- Computes Narrative Risk Score (NRS) & Influence Score (IS)
- Emits alerts (engine) and supports ad-hoc brand search

## Setup
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env  # fill in values
