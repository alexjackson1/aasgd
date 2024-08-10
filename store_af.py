import os

from af import parse_tgf


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

    apx_string = graph.to_apx()

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
