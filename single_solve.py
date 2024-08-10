from jaxtyping import Int
from beartype import beartype
from beartype.typing import Dict, List, Tuple

import os

import clingo
import torch
from torch import Tensor
from tqdm import tqdm

from af import Semantics, SEMANTICS, APXProgram


FILTER_FILE = "lib/aspartix/filter.lp"
SEMANTICS_FILES: Dict[Semantics, str] = {
    "GR": "lib/aspartix/ground.dl",
    "CO": "lib/aspartix/comp.dl",
    "ST": "lib/aspartix/stable.dl",
    "PR": "lib/aspartix/prefex_gringo.lp",
    "SST": "lib/aspartix/semi_stable_gringo.lp",
    "STG": "lib/aspartix/stage_gringo.lp",
}

Nodes = Dict[str, int]
Edges = List[Tuple[int, int]]


@beartype
class Graph:
    """Simple graph representation."""

    name: str
    nodes: Nodes
    edges: Edges
    node_count: int
    edge_count: int

    def __init__(self, name: str, nodes: Nodes, edges: Edges):
        self.name = name
        self.nodes = nodes
        self.edges = edges
        self.node_count = len(nodes)
        self.edge_count = len(edges)


@beartype
def solve(nodes: int, prog: APXProgram, sem: Semantics, p: int) -> Int[Tensor, "E A"]:
    """Solves an argumentation problem with `p` processors."""
    ctl = clingo.Control("0")

    # Add argumentation framework
    ctl.add("base", [], prog)

    # Load semantics solvers
    semantics_file = SEMANTICS_FILES[sem]
    ctl.load(semantics_file)
    ctl.load(FILTER_FILE)

    # Ground
    ctl.ground([("base", [])])

    # Solve the program
    extensions = []
    ctl.configuration.parallel_mode = str(p)
    with ctl.solve(yield_=True) as handle:
        for m in tqdm(handle, desc=f"Solving {sem} semantics", unit=" exts"):
            m = m.symbols(shown=True)
            ids = []
            for sym in m:
                sym_str = str(sym)
                id = sym_str[3:-1]
                ids.append(id)

            extension = torch.zeros(nodes, dtype=torch.int)
            extensions.append(extension)
            handle.get()

    extensions = torch.stack(extensions)
    return extensions


@beartype
def read_apx(apx_path: str) -> Tuple[str, int, APXProgram]:
    """Reads an apx file and returns the name, node count, and program."""
    name = os.path.basename(apx_path)

    # Read the apx file
    with open(apx_path, "r") as file:
        lines = file.readlines()

    # Count the number of nodes
    node_count = 0
    for l in lines:
        if not l.startswith("arg"):
            break
        node_count += 1

    # Return the name, node count, and the program
    return name, node_count, "".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 6:
        print(
            "Usage: python single_solve.py <tgf_path> <apx_dir> <out_dir> <semantics> <n_p>"
        )
        sys.exit(1)

    tgf_path = sys.argv[1]
    apx_dir = sys.argv[2]
    out_dir = sys.argv[3]
    semantics = sys.argv[4]
    parallel = int(sys.argv[5])

    if not os.path.exists(tgf_path):
        print(f"File {tgf_path} does not exist.")
        sys.exit(1)

    if not os.path.exists(apx_dir):
        print(f"Directory {apx_dir} does not exist.")
        sys.exit(1)

    file_name = os.path.basename(tgf_path)
    apx_path = os.path.join(apx_dir, file_name + ".apx")
    if not os.path.exists(apx_path):
        print(f"File {apx_path} does not exist.")
        sys.exit(1)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if semantics not in SEMANTICS:
        print(f"Semantics must be one of {SEMANTICS}")
        sys.exit(1)

    name, node_count, prog = read_apx(apx_path)
    print(f"{name} ({node_count})")

    exts = solve(node_count, prog, semantics, p=parallel)

    out_file = os.path.join(out_dir, f"{name}.{semantics}.pt")
    torch.save(exts, out_file)
    print(f"Saved to {out_file}")
