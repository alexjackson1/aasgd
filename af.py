from beartype.typing import Dict, List, Tuple, Literal
from beartype import beartype
from jaxtyping import Float, Int

import os
from torch import Tensor
import torch

SEMANTICS = ["CO", "GR", "PR", "ST", "SST", "STG"]

Semantics = Literal["GR", "CO", "ST", "SST", "STG", "PR"]
APXProgram = str

Nodes = Dict[str, int]
Edges = List[Tuple[int, int]]


@beartype
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

    def to_apx(self) -> APXProgram:
        nodes, edges = self.nodes, self.edges
        s = ""
        for _, index in nodes.items():
            s += f"arg({index}).\n"

        for edge in edges:
            s += f"att({edge[0]}, {edge[1]}).\n"

        return s

    def to_tensors(self) -> Tuple[Float[Tensor, "Arg 1"], Int[Tensor, "2 Att"]]:
        x = torch.arange(self.node_count).float().unsqueeze(1)
        edge_index = torch.tensor(self.edges).t().long().contiguous()
        return x, edge_index


@beartype
def parse_tgf(file_path: str) -> Graph:
    """Parses a TGF file and returns a graph."""
    with open(file_path, "r") as file:
        lines = file.readlines()

    nodes = {}
    edges = []
    current_index = 0
    is_edge_section = False

    for line in lines:
        line = line.strip()
        if line == "":
            continue

        if line == "#":
            is_edge_section = True
            continue

        if is_edge_section:
            labels = line.split(maxsplit=1)
            if len(labels) != 2:
                raise ValueError(f"Invalid edge line: {line}")

            edge = nodes[labels[0]], nodes[labels[1]]
            edges.append(edge)
        else:
            nodes[line] = current_index
            current_index += 1

    name = os.path.basename(file_path)
    graph = Graph(name, nodes, edges)
    return graph
