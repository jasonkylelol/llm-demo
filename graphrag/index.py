import sys, os
import argparse
import shutil

from graphrag.index.cli import index_cli

# GRAPHRAG_INPUT_FILE_TYPE=csv GRAPHRAG_API_BASE=http://192.168.0.20:38063/v1 python index.py --root /workspace/test1 --input /workspace/ragtest/input/
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--root",
        help="Working index directory",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--input",
        help="Path of input files",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    workdir = os.path.join("/workspace", args.root)
    # os.makedirs(workdir, exist_ok=True)
    shutil.copytree("template", workdir, dirs_exist_ok=True)
    shutil.copytree(args.input, os.path.join(workdir, "input"), dirs_exist_ok=True)

    index_cli(
        root=workdir,
        verbose=False,
        resume=None,
        memprofile=False,
        nocache=False,
        reporter="print",
        config=None,
        emit=None,
        dryrun=False,
        init=False,
        overlay_defaults=False,
        cli=False,
    )