# 1. Loop through all tgf files
# 2. Parse the tgf file into apx format
# 3. Solve the extensions
# 4. Store as data object
# 5. Save the data object


from typing import Dict, List, Literal, Tuple

import clingo
import torch
import os
from tqdm import tqdm
from torch_geometric.data import Data

# set parallelism


Nodes = Dict[str, int]
Edges = List[Tuple[int, int]]


class Graph:
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


SEMANTICS = ["GR", "CO", "ST", "SST", "STG", "PR"]
Semantics = Literal["GR", "CO", "ST", "SST", "STG", "PR"]
SEMANTICS_FILES: Dict[Semantics, str] = {
    "GR": "ground.dl",
    "CO": "comp.dl",
    "ST": "stable.dl",
    "PR": "prefex_gringo.lp",
    "SST": "semi_stable_gringo.lp",
    "STG": "stage_gringo.lp",
}


def solve_af_problem(
    node_count: int, prog: str, semantics: Semantics, parallel: int
) -> torch.Tensor:
    ctl = clingo.Control("0")

    # Add argumentation framework
    ctl.add("base", [], prog)

    # Load semantics solvers
    semantics_file = SEMANTICS_FILES[semantics]
    ctl.load(f"solver/{semantics_file}")
    ctl.load("solver/filter.lp")

    # Ground
    ctl.ground([("base", [])])

    # Set parallel
    ctl.configuration.parallel_mode = str(parallel)

    # Solve the program
    extensions = []
    with ctl.solve(yield_=True) as handle:
        for m in tqdm(handle, desc=f"Solving {semantics} semantics", unit=" exts"):
            m = m.symbols(shown=True)
            ids = []
            for sym in m:
                sym_str = str(sym)
                id = sym_str[3:-1]
                ids.append(id)

            extension = torch.zeros(node_count, dtype=torch.bool)
            extensions.append(extension)

            handle.get()

    extensions = torch.stack(extensions)

    return extensions


def read_apx(apx_path: str):
    name = os.path.basename(apx_path)

    # count number of args, and otherwise read into string
    with open(apx_path, "r") as file:
        lines = file.readlines()

    node_count = 0
    for l in lines:
        if not l.startswith("arg"):
            break
        node_count += 1

    return name, node_count, "".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 6:
        print("Usage: python aasgd.py <tgf_path> <apx_dir> <out_dir> <semantics> <n_p>")
        sys.exit(1)

    tgf_path = sys.argv[1]
    apx_dir = sys.argv[2]
    out_dir = sys.argv[3]
    semantics = sys.argv[4]
    parallel = sys.argv[5]

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

    exts = solve_af_problem(node_count, prog, semantics, parallel=parallel)

    out_file = os.path.join(out_dir, f"{name}.{semantics}.pt")
    torch.save(exts, out_file)
    print(f"Saved to {out_file}")
