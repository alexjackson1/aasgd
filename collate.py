from beartype import beartype
from beartype.typing import Dict, Optional
from jaxtyping import Int

import os

import torch
from torch import Tensor
from af import parse_tgf, SEMANTICS

FILE_EXT = "pt"


@beartype
def check_ext(tgf_path: str, ext_dir: str, semantics: str) -> Optional[str]:
    ext_path = os.path.join(
        ext_dir, os.path.basename(tgf_path) + f".apx.{semantics}.{FILE_EXT}"
    )

    if os.path.exists(ext_path):
        return ext_path

    return None


def parse_extensions(
    paths: Dict[str, Optional[str]]
) -> Dict[str, Int[Tensor, "Args 1"]]:
    extensions = {}
    for sem, path in paths.items():
        if path is None:
            extensions[sem] = None
            continue

        vec = torch.load(path, weights_only=False, map_location="cpu").long()
        extensions[sem] = vec

    return extensions


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python collate.py <tgf_dir> <ext_dir>")
        sys.exit(1)

    tgf_dir = sys.argv[1]
    ext_dir = sys.argv[2]

    if not os.path.exists(tgf_dir):
        print(f"Directory {tgf_dir} does not exist.")
        sys.exit(1)

    if not os.path.exists(ext_dir):
        print(f"Directory {ext_dir} does not exist.")
        sys.exit(1)

    entries, file_count = [], 0
    counts, missing_all, missing_any = {}, [], []

    for sem in SEMANTICS:
        counts[sem] = 0

    for tgf_file in os.listdir(tgf_dir):
        if not tgf_file.endswith(".tgf"):
            continue

        tgf_path = os.path.join(tgf_dir, tgf_file)
        file_count += 1

        # check existence of corresponding extension files
        ext_paths = {sem: check_ext(tgf_path, ext_dir, sem) for sem in SEMANTICS}
        present = [sem for sem, path in ext_paths.items() if path is not None]
        missing = [sem for sem, path in ext_paths.items() if path is None]

        if len(missing) > 0:
            missing_any.append((tgf_file, missing))

        if len(missing) == len(SEMANTICS):
            print(f"No extensions found for {tgf_file}, skipping...")
            missing_all.append(tgf_file)
            continue

        for sem in present:
            counts[sem] += 1

        print(f"Processing {tgf_file}...")
        graph = parse_tgf(tgf_path)
        x, edge_index = graph.to_tensors()
        extensions = parse_extensions(ext_paths)

        entry = {"x": x, "edge_index": edge_index, "extensions": extensions}
        entries.append(entry)

    torch.save(entries, "data.pt")
    print(f"Saved {len(entries)} entries to data.pt.")

    with open("missing_all.txt", "w") as f:
        f.write("\n".join(missing_all))

    with open("missing_any.txt", "w") as f:
        for tgf_file, missing in missing_any:
            f.write(f"{tgf_file}: {', '.join(missing)}\n")

    with open("counts.txt", "w") as f:
        for sem, count in counts.items():
            print(f"{sem}: {count} ({count / file_count:0%})")
            f.write(f"{sem}: {count} ({count / file_count:0%})\n")
