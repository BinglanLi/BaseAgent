# Knowledge Graph Builder — Frontend Prototype

Interactive frontend for visualizing multi-agent knowledge graph construction with real-time telemetry and human-in-the-loop controls.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run in demo mode (no API keys needed)
python server.py --demo

# Open http://127.0.0.1:8765 in your browser
# Click "Run Demo" to see the simulated pipeline
```

## Real Mode

Requires a configured LLM provider (e.g. `ANTHROPIC_API_KEY`):

```bash
python server.py --port 8765
```

Type a task in the input box and click "Send Message". The backend streams `BaseAgent.run_stream()` events over WebSocket.

## Architecture

```
browser  <──WebSocket──>  server.py (FastAPI)  ──>  BaseAgent.run_stream()
  │                            │
  ├─ D3.js graph canvas        ├─ Broadcasts state snapshots
  ├─ Telemetry log             ├─ Tracks cost accumulation
  ├─ HITL approve/reject       └─ Demo simulation mode
  └─ Status bar
```

### WebSocket Messages

**Client → Server:**
| type | description |
|------|-------------|
| `run_task` | Start agent with `{prompt: "..."}` |
| `demo` | Start demo simulation |
| `approve` | Approve pending code block |
| `reject` | Reject with `{feedback: "..."}` |
| `message` | Free-form HITL message `{text: "..."}` |
| `reset` | Clear all state |

**Server → Client:**
| type | description |
|------|-------------|
| `state` | Full state snapshot (graph, cost, telemetry, status) |
| `approval_required` | Code block pending review `{code, language}` |
| `error` | Error message |

## Files

- `server.py` — FastAPI backend with WebSocket, BaseAgent integration, and demo simulator
- `index.html` — Single-page frontend with D3.js, embedded CSS/JS
- `requirements.txt` — Python dependencies (FastAPI + uvicorn)
