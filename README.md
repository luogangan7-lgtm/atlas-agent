# ATLAS — Autonomous Knowledge Agent

> *You read hundreds of articles, books, and conversations. But when you need to connect an idea from last quarter to what you learned yesterday — it's gone.*

ATLAS is a self-running background agent that captures everything that passes through your Claude sessions, organizes it into a living knowledge pyramid, and surfaces it exactly when you need it.

It runs on your machine, 24/7, without you lifting a finger.

---

## What it does

Every 15 minutes, ATLAS autonomously:

1. **Captures** — Every Claude conversation, web search, and note you input flows into L0 raw storage
2. **Organizes** (L0 → L1) — DeepSeek extracts structured knowledge nodes: concepts, arguments, procedures, principles
3. **Connects** (L1 → L2) — Finds cross-domain insights between unrelated fields (e.g. what sales psychology has in common with product design)
4. **Synthesizes** (L2 → L3) — When a domain has enough nodes, distills a high-level framework
5. **Evolves** (L3 → L4) — Tracks its own reasoning quality and adjusts its extraction strategies over time
6. **Asks when unsure** — Drops a Markdown file in your Obsidian inbox for decisions it can't make alone

Your Obsidian vault becomes a second brain that writes itself.

---

## The Knowledge Pyramid

```
L4  ████  Self-model & meta-learning (how the agent reasons)
L3  ████████  Domain frameworks (synthesized from 15+ nodes)
L2  ████████████  Cross-domain insights (surprising connections)
L1  ████████████████████  Structured knowledge nodes (concepts, facts, principles)
L0  ████████████████████████████  Raw captures (everything that comes in)
```

Current live stats (one person, ~6 months): **L1 ≈ 2,400 nodes · L2 ≈ 182 · L3 ≈ 163 · L4 ≈ 4**

---

## Architecture

```
Claude Code session
      │  (openclaw hook — captures every conversation turn)
      ▼
Qdrant (local vector DB)  ←──────────────────────┐
      │                                           │
      ▼                                           │
atlas-agent (Python, launchd daemon)              │
  ├── perception.py   — reads world state         │
  ├── decision.py     — priority-based task queue │
  ├── organize.py     — L0 → L1 (DeepSeek)        │
  ├── associate.py    — L1 → L2 (cross-domain)    │
  ├── synthesize.py   — L2 → L3 (frameworks)      │
  ├── evolution/      — L3 → L4 (self-improvement) │
  ├── inbox_monitor   — reads your Obsidian decisions
  └── mcp_server      — query interface for Claude ─┘

Obsidian vault (local Markdown)
  ├── L1/{category}/   — knowledge nodes
  ├── L2/{category}/   — cross-domain insights
  ├── L3/{category}/   — domain frameworks
  └── agent-brain/     — inbox, outbox, daily reports
```

**Two components, both required:**

| Component | Language | Role |
|-----------|----------|------|
| `atlas-agent/` (this repo) | Python | Autonomous background daemon |
| `hooks/atlas-memory/` | Node.js | openclaw hook — captures Claude sessions |

---

## Prerequisites

| Dependency | What it's for | Required |
|------------|---------------|----------|
| [Qdrant](https://qdrant.tech) | Vector storage | ✅ |
| [Ollama](https://ollama.com) + `bge-m3` | Local embeddings | ✅ |
| [DeepSeek API](https://platform.deepseek.com/) | LLM (organize, connect, synthesize) | ✅ |
| [Obsidian](https://obsidian.md) | Knowledge vault UI | ✅ |
| [openclaw](https://github.com/anthropics/claude-code) | Claude Code CLI hook runner | ✅ |
| [oMLX](https://github.com/smpanaro/more-rlhf) + `Qwen3.5-9B` | Faster local inference (Apple Silicon) | Optional |

---

## Switching Models

ATLAS uses two models: one for **reasoning** (cloud), one for **embeddings** (local).

### Reasoning model — edit `config.yaml`

```yaml
llm:
  primary: "deepseek"
  deepseek_model: "deepseek-v4-flash"   # ← change this
```

**Supported values:**

| Model string | Provider | Notes |
|---|---|---|
| `deepseek-v4-flash` | DeepSeek cloud | Default. Fast + cheap |
| `deepseek-chat` | DeepSeek cloud | More capable, slower |
| `deepseek-reasoner` | DeepSeek cloud | Best quality, expensive |
| Any Ollama model name | Local | Set `primary: "ollama"` |

To switch to a local Ollama model entirely:

```yaml
llm:
  primary: "ollama"
  ollama_model: "qwen2.5:14b"   # or any model you've pulled
```

The fallback chain is: DeepSeek → Ollama (configured in `utils/llm.py`).

### Embedding model — edit `config.yaml`

```yaml
embedding:
  model: "bge-m3"           # ← change this
  endpoint: "http://127.0.0.1:11434"
```

`bge-m3` produces 1024-dim vectors and works well for Chinese + English mixed content. If you switch models, you must recreate the Qdrant collection (all embeddings change dimension).

```bash
# Recreate collection after changing embedding model
curl -X DELETE http://127.0.0.1:6333/collections/atlas_memories_v2
# Then restart atlas-agent — it will recreate the collection on first run
```

---

## Setup

### 1. Start local services

```bash
# Qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Ollama + embedding model
brew install ollama
ollama serve &
ollama pull bge-m3
```

### 2. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/atlas-agent.git
cd atlas-agent

# Edit config.yaml — set your vault path and DeepSeek API key endpoint
cp .env.example .env
# Fill in DEEPSEEK_API_KEY in .env
```

### 3. Edit `config.yaml`

```yaml
storage:
  obsidian_vault: "/path/to/your/obsidian-vault"   # ← your vault
```

### 4. Initialize vault structure

```bash
python3 -c "from storage.obsidian_writer import ensure_vault_structure; ensure_vault_structure()"
```

### 5. Install as background daemon (macOS)

```bash
cp com.atlas.agent.plist.example com.atlas.agent.plist
# Edit com.atlas.agent.plist:
#   - Set your Python path (which python3)
#   - Set your DEEPSEEK_API_KEY
#   - Set your paths

cp com.atlas.agent.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.atlas.agent.plist
```

### 6. Install the openclaw hook

```bash
# Copy hooks/atlas-memory/ to your openclaw hooks directory
cp -r hooks/atlas-memory ~/.openclaw/hooks/atlas-memory

# Set env vars in your shell profile
echo 'export DEEPSEEK_API_KEY=sk-...' >> ~/.zshrc
echo 'export ATLAS_OBSIDIAN_VAULT=/path/to/your/vault' >> ~/.zshrc
```

---

## Daily operation

Once running, you do nothing. Just use Claude normally.

- **Check what the agent did**: Open `agent-brain/daily/YYYY-MM-DD.md` in Obsidian
- **Answer agent questions**: Check `agent-brain/inbox/` — tick `[x]` on choices, save
- **Manual evolution trigger**: `touch /your-vault/agent-brain/state/trigger_evolution.flag`
- **Logs**: `tail -f ~/Library/Logs/atlas-agent.log`

The agent will auto-confirm inbox items after 24 hours if you don't respond (except deletion operations — those always wait for you).

---

## Taxonomy

Knowledge is organized into 12 top-level categories, editable at:

```
agent-brain/state/taxonomy.json
```

Default categories: 商业营销 · 战略管理 · 金融投资 · 技术科学 · 认知学习 · 组织管理 · 数据分析 · 心理行为 · 个人成长 · 客户增长 · 生态科学 · 其他

---

## Roadmap & where I need help

I built this for myself over several months. It works — but it's still one person's system. Here's what I'd love help with:

### Looking for collaborators on:

- **🔧 Setup experience** — Right now setup takes ~30 minutes and requires touching 4 config files. Can we make it a 5-minute install?
- **🌍 Multi-vault support** — Currently hardcoded to one vault path. Config-based multi-user support would unlock teams using this
- **🔌 Alternative LLM backends** — Claude API, OpenAI, local-only (no cloud calls) mode
- **📊 Health dashboard** — A simple web UI to see pyramid stats without opening Obsidian
- **🪝 More capture hooks** — Obsidian Daily Notes, Readwise, browser extension
- **🧪 Test coverage** — Almost zero automated tests right now. It's embarrassing
- **📖 Better docs** — I know what everything does. You don't. Help me bridge that gap

### How to contribute

1. Open an issue describing what you're trying to do
2. Or just fork, build, and open a PR — I review everything
3. If you get stuck on setup, open an issue. Your confusion is my bug

**If you're using this or experimenting with it, I'd genuinely like to know.** Star the repo, open an issue, say hi. This project is more useful if more minds are working on it.

---

## Questions & support

- **Setup issues** → open a GitHub issue with your OS, Python version, and the error
- **Feature requests** → open an issue with your use case
- **Architecture questions** → start a Discussion

---

## License

MIT — use it, fork it, build on it.
