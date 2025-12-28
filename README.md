# Civil Mini-Apps: Submittal Checker + Schedule What-Ifs

Fast, practical tools for civil/CM workflows:
- **Submittal Checker**: compares a submittal to specs using a **hybrid score** (lexical + semantic + keyword coverage + penalties + section boost). Reviewer “must/nice/forbidden” lists, presets, and a **Memory Bank** to track runs (company, project, pass rate, quotes).
- **Schedule What-Ifs**: CPM with **FS/SS/FF lags**, optional **fast-track overlap**, full float breakdown (Total/Free/Independent/Interfering), **calendar mapping** (workweek + holidays), and a simple **crash-to-target** optimizer.

## Quick start

```bash
git clone https://github.com/yourname/civil-miniapps.git
cd civil-miniapps
cp .streamlit/secrets.example.toml .streamlit/secrets.toml   # edit with your values
./scripts/run_local.sh

