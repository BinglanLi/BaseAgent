"""
Train Node2Vec embeddings on the CardioKB graph.

Uses PecanPy (memory-efficient Node2Vec) to learn 128-dimensional
node embeddings from the exported edge list.
"""

import time
import json
import numpy as np
from pathlib import Path

from pecanpy import pecanpy

DATA_DIR = Path(__file__).resolve().parent / "data"
EDGELIST_PATH = DATA_DIR / "edgelist.txt"
EMBEDDING_PATH = DATA_DIR / "embeddings.npz"
SUMMARY_PATH = DATA_DIR / "export_summary.json"

# Node2Vec hyperparameters
DIMENSIONS = 128
WALK_LENGTH = 80
NUM_WALKS = 10
P = 1.0        # return parameter (1.0 = neutral)
Q = 0.5        # in-out parameter (<1 = BFS-like, captures local structure)
WORKERS = 4
WINDOW = 10
EPOCHS = 5
MIN_COUNT = 0  # keep all nodes


def main():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    print(f"Graph: {summary['num_nodes']:,} nodes, {summary['num_edges']:,} edges")

    print(f"\nNode2Vec config:")
    print(f"  dimensions={DIMENSIONS}, walk_length={WALK_LENGTH}, num_walks={NUM_WALKS}")
    print(f"  p={P}, q={Q}, window={WINDOW}, epochs={EPOCHS}")

    print(f"\nLoading graph from {EDGELIST_PATH}...")
    t0 = time.time()

    g = pecanpy.PreComp(p=P, q=Q, workers=WORKERS, verbose=True)
    g.read_edg(str(EDGELIST_PATH), weighted=False, directed=False)
    print(f"Graph loaded in {time.time() - t0:.1f}s")

    print(f"\nGenerating random walks...")
    t1 = time.time()
    walks = g.simulate_walks(num_walks=NUM_WALKS, walk_length=WALK_LENGTH)
    print(f"Generated {len(walks):,} walks in {time.time() - t1:.1f}s")

    print(f"\nTraining Word2Vec on walks...")
    t2 = time.time()
    from gensim.models import Word2Vec
    model = Word2Vec(
        walks,
        vector_size=DIMENSIONS,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,           # skip-gram
        workers=WORKERS,
        epochs=EPOCHS,
    )
    print(f"Training complete in {time.time() - t2:.1f}s")

    node_ids = np.array([int(w) for w in model.wv.index_to_key])
    embeddings = np.array([model.wv[str(nid)] for nid in node_ids])

    np.savez_compressed(
        EMBEDDING_PATH,
        node_ids=node_ids,
        embeddings=embeddings,
    )
    print(f"\nSaved embeddings: {embeddings.shape} to {EMBEDDING_PATH}")
    print(f"  {embeddings.shape[0]} nodes x {embeddings.shape[1]} dimensions")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
