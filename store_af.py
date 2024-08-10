from typing import Dict, List, Tuple

import os

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


def to_apx_string(graph: Graph) -> str:
    nodes, edges = graph.nodes, graph.edges
    s = ""
    for _, index in nodes.items():
        s += f"arg({index}).\n"

    for edge in edges:
        s += f"att({edge[0]}, {edge[1]}).\n"

    return s


def main(tgf_path: str, out_dir: str):
    if not os.path.exists(tgf_path):
        print(f"File {tgf_path} does not exist.")
        return

    if os.path.exists(os.path.join(out_dir, os.path.basename(tgf_path) + ".apx")):
        print(f"File {tgf_path + '.apx'} already exists.")
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    graph = parse_tgf(tgf_path)
    print(f"{graph.name} ({graph.node_count}, {graph.edge_count})")

    apx_string = to_apx_string(graph)

    with open(os.path.join(out_dir, graph.name + ".apx"), "w") as file:
        file.write(apx_string)
    print(f"Saved {graph.name}.apx")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python store_af.py <tgf_path> <out_dir>")
        sys.exit(1)

    tgf_path = sys.argv[1]
    out_dir = sys.argv[2]

    main(tgf_path, out_dir)
