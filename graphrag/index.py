
import argparse
import shutil

from graphrag.index.cli import index_cli


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--root",
        help="If no configuration is defined, the root directory to use for input data and output data. Default value: the current directory",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    index_cli(
        root=args.root,
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
        cli=True,
    )