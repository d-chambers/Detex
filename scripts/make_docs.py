"""
Script to re-make the html docs and publish to gh-pages.
"""
import os
from contextlib import contextmanager
from queue import PriorityQueue

from pathlib2 import Path

from clean_docs import main as clean_docs


@contextmanager
def change_directory(new_path):
    here = Path(".")
    os.chdir(str(new_path))
    yield
    os.chdir(str(here))


def main():
    doc_path = Path(__file__).absolute().parent.parent / "docs"
    # # clean out all the old docs
    clean_docs()

    cmd = "jupyter nbconvert --ExecutePreprocessor.timeout=1200 --to notebook --execute --inplace "
    for note_book_path in doc_path.rglob("*.ipynb"):
        # skip any hidden directores, eg .ipynb_checkpoints
        if note_book_path.parent.name.startswith('.'):
            continue
        cmd_to_run = cmd + str(note_book_path)
        result = os.system(cmd_to_run)
        if result != 0:
            raise RuntimeError("failed to run: " + cmd_to_run)
    # run auto api-doc

    with change_directory(doc_path):
        api_doc_command = " sphinx-apidoc ../obsplus -o api"
        result = os.system(api_doc_command)
        if result != 0:
            print('failed to run: ' + api_doc_command)
        # make the rest of the docs
        doc_cmd = 'make html'
        result = os.system(doc_cmd)
        if result != 0:
            print('failed to run: ' + doc_cmd)


if __name__ == "__main__":
    main()
