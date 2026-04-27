"""
Knowledge Graph Builder — FastAPI backend.

Streams BaseAgent events over WebSocket and manages graph state.
Includes a demo mode (--demo) that simulates agent activity without LLM keys.

Usage:
    # Real mode (requires ANTHROPIC_API_KEY or other provider key):
    uvicorn server:app --reload

    # Demo mode (no API keys needed):
    python server.py --demo
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Add project root to path so we can import BaseAgent
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    id: str
    label: str
    node_type: str  # "SOURCE" or "ONTOLOGY"
    x: float | None = None
    y: float | None = None

@dataclass
class GraphLink:
    source: str  # node id
    target: str  # node id
    label: str = ""

@dataclass
class AppState:
    nodes: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)
    telemetry: list[dict] = field(default_factory=list)
    accumulated_cost: float = 0.0
    active_agent: str = "Idle"
    is_running: bool = False
    is_interrupted: bool = False
    pending_code: str | None = None
    thread_id: str | None = None

app_state = AppState()

# Track connected WebSocket clients
clients: set[WebSocket] = set()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Knowledge Graph Builder")

DEMO_MODE = False  # Set via CLI flag


def timestamp_str() -> str:
    return datetime.now().strftime("[%I:%M:%S %p]")


def add_telemetry(message: str, agent: str | None = None):
    entry = {
        "time": timestamp_str(),
        "message": message,
        "agent": agent or app_state.active_agent,
    }
    app_state.telemetry.append(entry)
    # Keep last 200 entries
    if len(app_state.telemetry) > 200:
        app_state.telemetry = app_state.telemetry[-200:]


def state_snapshot() -> dict:
    return {
        "type": "state",
        "graph": {"nodes": app_state.nodes, "links": app_state.links},
        "cost": round(app_state.accumulated_cost, 4),
        "active_agent": app_state.active_agent,
        "telemetry": app_state.telemetry[-50:],
        "is_running": app_state.is_running,
        "is_interrupted": app_state.is_interrupted,
    }


async def broadcast(message: dict):
    dead = set()
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


# ---------------------------------------------------------------------------
# BaseAgent integration
# ---------------------------------------------------------------------------

async def run_agent_task(prompt: str):
    """Run a BaseAgent task and stream events to all connected clients."""
    try:
        from BaseAgent import BaseAgent
        from BaseAgent.events import EventType
    except ImportError:
        add_telemetry("ERROR: BaseAgent package not found. Run from project root.")
        await broadcast({"type": "telemetry", "entries": app_state.telemetry[-1:]})
        return

    app_state.is_running = True
    app_state.active_agent = "Supervisor"
    add_telemetry(f"Starting task: {prompt[:80]}...")
    await broadcast(state_snapshot())

    try:
        agent = BaseAgent(llm="azure-claude-sonnet-4-5", require_approval="always")
        app_state.thread_id = None  # will be set by run_stream

        async for event in agent.run_stream(prompt):
            etype = event.event_type

            if etype == EventType.THINKING:
                app_state.active_agent = "Reasoning"
                add_telemetry(event.content[:120] + ("..." if len(event.content) > 120 else ""))

            elif etype == EventType.CODE_EXECUTING:
                app_state.active_agent = "Code_Executor"
                add_telemetry(f"Executing {event.metadata.get('language', 'python')} code...")

            elif etype == EventType.CODE_RESULT:
                add_telemetry(f"Result: {event.content[:120]}...")
                # Try to parse graph updates from code results
                _try_parse_graph_update(event.content)
                app_state.accumulated_cost += 0.002  # approximate per-step cost

            elif etype == EventType.APPROVAL_REQUIRED:
                app_state.is_interrupted = True
                app_state.pending_code = event.content
                app_state.active_agent = "Awaiting_Approval"
                add_telemetry(f"Approval needed for {event.metadata.get('language', 'python')} code")
                await broadcast({
                    "type": "approval_required",
                    "code": event.content,
                    "language": event.metadata.get("language", "python"),
                })

            elif etype == EventType.FINAL_ANSWER:
                app_state.active_agent = "Complete"
                add_telemetry(f"Final answer: {event.content[:120]}...")

            elif etype == EventType.ERROR:
                add_telemetry(f"ERROR: {event.content[:120]}")

            await broadcast(state_snapshot())

    except Exception as e:
        add_telemetry(f"Agent error: {str(e)[:120]}")
    finally:
        app_state.is_running = False
        app_state.active_agent = "Idle"
        await broadcast(state_snapshot())


def _try_parse_graph_update(content: str):
    """Try to extract graph nodes/links from agent output (JSON or heuristic)."""
    try:
        # Look for JSON blocks in the output
        import re
        json_match = re.search(r'\{[\s\S]*"nodes"[\s\S]*\}', content)
        if json_match:
            data = json.loads(json_match.group())
            if "nodes" in data:
                for node in data["nodes"]:
                    node_id = node.get("id", node.get("name", ""))
                    if node_id and not any(n["id"] == node_id for n in app_state.nodes):
                        app_state.nodes.append({
                            "id": node_id,
                            "label": node.get("label", node_id),
                            "type": node.get("type", "SOURCE"),
                        })
            if "links" in data or "edges" in data:
                for link in data.get("links", data.get("edges", [])):
                    app_state.links.append({
                        "source": link.get("source", link.get("from", "")),
                        "target": link.get("target", link.get("to", "")),
                        "label": link.get("label", link.get("relation", "")),
                    })
    except (json.JSONDecodeError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Demo mode — simulated agent activity
# ---------------------------------------------------------------------------

EXAMPLE_PROMPT = (
    "Set up a knowledge graph repository for 'Parkinson\u2019s disease'. "
    "disease_agent: Read config/project.yaml to understand the structure, then "
    "rewrite it for 'Parkinson\u2019s disease'. Update project.name, display_name, and all "
    "disease_scope fields (primary_terms, umls_cuis, doid_ids, mesh_ids). "
    "Keep ontology paths, node_types, edge_types, and graph_indexes unchanged. "
    "database_agent: Read config/databases.yaml to understand the structure, then "
    "update the enabled flags to reflect which sources are appropriate for 'Parkinson\u2019s disease'. "
    "Keep all args and notes fields unchanged."
)

# Schema nodes and edges are derived from parkinsons_disease-kg/config/databases.yaml
# (enabled sources) and parkinsons_disease-kg/config/ontology_mappings.yaml.
# Node types use SOURCE (blue) for biological entities, ONTOLOGY (purple) for
# ontology-defined term types.
DEMO_NODES = [
    {"id": "Gene",                "label": "Gene",                "type": "ONTOLOGY"},
    {"id": "Disease",             "label": "Disease",             "type": "ONTOLOGY"},
    {"id": "Drug",                "label": "Drug",                "type": "ONTOLOGY"},
    {"id": "TranscriptionFactor", "label": "TranscriptionFactor", "type": "ONTOLOGY"},
    {"id": "Pathway",             "label": "Pathway",             "type": "ONTOLOGY"},
    {"id": "BodyPart",            "label": "BodyPart",            "type": "ONTOLOGY"},
    {"id": "Symptom",             "label": "Symptom",             "type": "ONTOLOGY"},
    {"id": "DrugClass",           "label": "DrugClass",           "type": "ONTOLOGY"},
]

DEMO_LINKS = [
    {"source": "Gene",              "target": "Pathway",  "label": "geneInPathway"},
    {"source": "Gene",              "target": "Disease",  "label": "geneAssociatesWithDisease"},
    {"source": "Drug",              "target": "Gene",     "label": "chemicalBindsGene"},
    {"source": "TranscriptionFactor", "target": "Gene",   "label": "TFInteractsWithGene"},
    {"source": "Drug",              "target": "DrugClass","label": "drugInClass"},
    {"source": "Symptom",           "target": "Disease",  "label": "symptomManifestationOfDisease"},
    {"source": "Disease",           "target": "BodyPart", "label": "diseaseLocalizesToAnatomy"},
    {"source": "Disease",           "target": "Disease",  "label": "diseaseAssociatesWithDisease"},
]

DEMO_AGENTS = [
    "Supervisor",
    "disease_agent",
    "database_agent",
]

DEMO_TELEMETRY_MESSAGES = [
    ("Supervisor", [
        "Dispatching disease_agent to configure project identifiers...",
        "Dispatching database_agent to enumerate node and edge types...",
    ]),
    ("disease_agent", [
        "Reading config/project.yaml...",
        "Set disease scope: Parkinson\u2019s disease",
        "Set UMLS CUI: C0030567",
        "Set DOID: DOID:14330",
        "Set MeSH ID: D010300",
        "config/project.yaml updated successfully.",
    ]),
    ("database_agent", [
        "Reading config/databases.yaml \u2014 10 sources enabled...",
        "Reading config/ontology_mappings.yaml...",
        "aopdb \u2192 node: Pathway, Drug | edge: geneInPathway",
        "disgenet \u2192 node: Disease | edge: geneAssociatesWithDisease",
        "drugbank \u2192 node: Drug | edge: chemicalBindsGene",
        "ncbigene \u2192 node: Gene",
        "dorothea \u2192 node: TranscriptionFactor | edge: TFInteractsWithGene",
        "disease_ontology \u2192 node: Disease",
        "uberon \u2192 node: BodyPart",
        "mesh \u2192 node: Symptom",
        "drugcentral \u2192 node: DrugClass | edge: drugInClass",
        "medline \u2192 edge: symptomManifestationOfDisease, diseaseLocalizesToAnatomy",
        "Schema: 8 node types, 8 edge types across 10 enabled sources.",
    ]),
    ("Supervisor", [
        "Schema validated for Parkinson\u2019s disease KG.",
        "Node types: Gene, Disease, Drug, TranscriptionFactor, Pathway, BodyPart, Symptom, DrugClass",
        "KG repository ready. Run: python src/main.py run",
    ]),
]


async def run_demo_simulation():
    """Simulate a multi-agent knowledge graph building session."""
    app_state.is_running = True
    add_telemetry("Starting disease KG schema discovery for 'Parkinson\u2019s disease'...")
    await broadcast(state_snapshot())
    await asyncio.sleep(1.0)

    for agent_name, messages in DEMO_TELEMETRY_MESSAGES:
        app_state.active_agent = agent_name
        cost_per_msg = round(0.015 + 0.01 * (hash(agent_name) % 5) / 5, 4)

        # Dispatch from Supervisor to sub-agents
        if agent_name == "disease_agent":
            await broadcast({"type": "agent_message", "from": "Supervisor", "to": "disease_agent",
                              "text": "Configure project identifiers for Parkinson\u2019s disease"})
            await asyncio.sleep(0.8)
        elif agent_name == "database_agent":
            await broadcast({"type": "agent_message", "from": "Supervisor", "to": "database_agent",
                              "text": "Enumerate node and edge types from enabled sources"})
            await asyncio.sleep(0.8)

        for msg in messages:
            add_telemetry(msg, agent=agent_name)
            app_state.accumulated_cost += cost_per_msg
            await broadcast(state_snapshot())
            await asyncio.sleep(0.6 + 0.4 * (hash(msg) % 3) / 3)

        # Simulate a HITL interrupt after disease_agent finishes project.yaml
        if agent_name == "disease_agent" and "successfully" in messages[-1].lower():
            app_state.is_interrupted = True
            app_state.active_agent = "Awaiting_Approval"
            add_telemetry("disease_agent: config/project.yaml updated. Approve database configuration?")
            await broadcast({
                "type": "approval_required",
                "code": "Proceed with database_agent:\n- Enumerate node and edge types from 10 enabled sources\n- Validate schema against ontology_mappings.yaml",
                "language": "plan",
            })
            await broadcast(state_snapshot())

            # Wait for user response (or auto-approve after 15s in demo)
            for _ in range(150):
                if not app_state.is_interrupted:
                    break
                await asyncio.sleep(0.1)
            else:
                app_state.is_interrupted = False
                add_telemetry("Auto-approved after timeout.")

            # Broadcast cleared state before reporting back (fixes HITL ordering)
            await broadcast(state_snapshot())
            await asyncio.sleep(0.5)
            await broadcast({"type": "agent_message", "from": "disease_agent", "to": "Supervisor",
                              "text": "project.yaml updated \u2014 ready for database config"})
            await asyncio.sleep(0.5)

        # Report back from database_agent after completing its work
        if agent_name == "database_agent":
            await broadcast({"type": "agent_message", "from": "database_agent", "to": "Supervisor",
                              "text": "Schema validated: 8 node types, 8 edge types"})
            await asyncio.sleep(0.5)

    # Reveal the full KG schema after the conversation finishes
    app_state.nodes = list(DEMO_NODES)
    app_state.links = list(DEMO_LINKS)
    app_state.active_agent = "Complete"
    app_state.is_running = False
    add_telemetry("Pipeline complete.")
    await broadcast(state_snapshot())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        # Send current state on connect
        await websocket.send_json(state_snapshot())
        # Pre-fill the UI input with the example prompt (one-time, on connect)
        await websocket.send_json({"type": "example_prompt", "prompt": EXAMPLE_PROMPT})

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "run_task":
                prompt = data.get("prompt", "")
                if not prompt:
                    continue
                if app_state.is_running:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Agent is already running.",
                    })
                    continue

                if DEMO_MODE:
                    asyncio.create_task(run_demo_simulation())
                else:
                    asyncio.create_task(run_agent_task(prompt))

            elif msg_type == "approve":
                app_state.is_interrupted = False
                app_state.pending_code = None
                add_telemetry("User approved. Resuming...")
                await broadcast(state_snapshot())

            elif msg_type == "reject":
                feedback = data.get("feedback", "User rejected. Try a different approach.")
                app_state.is_interrupted = False
                app_state.pending_code = None
                add_telemetry(f"User rejected: {feedback[:100]}")
                await broadcast(state_snapshot())

            elif msg_type == "message":
                # Free-form HITL message from user
                text = data.get("text", "")
                add_telemetry(f"User: {text[:120]}", agent="User")
                await broadcast(state_snapshot())

            elif msg_type == "reset":
                app_state.nodes = []
                app_state.links = []
                app_state.telemetry = []
                app_state.accumulated_cost = 0.0
                app_state.active_agent = "Idle"
                app_state.is_running = False
                app_state.is_interrupted = False
                app_state.pending_code = None
                add_telemetry("Graph reset.")
                await broadcast(state_snapshot())

            elif msg_type == "demo":
                if not app_state.is_running:
                    asyncio.create_task(run_demo_simulation())

    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Knowledge Graph Builder server")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (no LLM keys needed)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    DEMO_MODE = args.demo
    if DEMO_MODE:
        print("Running in DEMO mode — no API keys required.")

    uvicorn.run(app, host=args.host, port=args.port)
