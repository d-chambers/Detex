"""
Script to make the documentation and publish to gh-pages. This should
be done once for each release.
"""
import os
import tempfile
from contextlib import contextmanager
from pathlib2 import Path
# from subprocess import PIPE

from clean_docs import main as clean_docs
from make_docs import main as make_docs

import detex

version = detex.__version__


def run(cmd):
    """ Run a command on the shell. """
    result = os.system(cmd)
    if result != 0:
        print('failed to run command: ' + cmd)


@contextmanager
def change_directory(new_path):
    here = Path(".")
    os.chdir(str(new_path))
    yield
    os.chdir(str(here))


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    tmp = Path(tempfile.mkdtemp())
    html_path = (base / "docs" / "_build" / "html").absolute()
    detex_url = "https://github.com/d-chambers/detex"
    # make the docs
    make_docs()
    assert html_path.exists()
    # clone a copy of obplus to tempfile and cd there
    with change_directory(tmp):
        run("git clone " + detex_url)
    with change_directory(tmp / "detex"):
        # checkout gh-pages
        run("git checkout gh-pages")
        # reset to the first commit in gh-pages branch
        cmd = "git reset --hard `git rev-list --max-parents=0 HEAD | tail -n 1`"
        run(cmd)
        # copy all contents of html
        run("cp -R {}/* ./".format(html_path))
        # make a commit
        run("git add -A")
        run('git commit -m "{} docs"'.format(version))
        run("git push -f origin gh-pages")
    # clean up original repo
    clean_docs()
