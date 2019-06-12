"""
Script to clean the docs
"""
import os

import shutil
from pathlib2 import Path


def main():
    doc_path = Path(__file__).absolute().parent.parent / "docs"
    # make api documentation
    cmd = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace"
    api_path = doc_path / "api"
    # clear API documentation
    if api_path.exists():
        shutil.rmtree(api_path)
    # execute all the notebooks
    for note_book_path in doc_path.rglob("*.ipynb"):
        cmd_to_run = cmd + " {}".format(str(note_book_path))
        result = os.system(cmd_to_run)
        if result != 0:
            msg = "failed to execute " + str(note_book_path)
            raise RuntimeError(msg)


if __name__ == "__main__":
    main()
