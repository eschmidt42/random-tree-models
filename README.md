# random-tree-models
My implementation of a random selection of tree based models.

## layout

* `random_tree_models/` -> python implementation of tree algorithms
* `tests/` -> unit tests
* `nbs/` -> jupyter notebooks to play with the tree algorithms
* `config/` -> requirements.txt

## setup

In this project `pyenv` is used to provide the python version specified in `.python-version`. To manage the virtual environment `Makefile` and `pip-tools` are used.

### `pyenv` install

Install pyenv using [this guide](https://github.com/pyenv/pyenv#installation).

If you have it already installed or newly installed run

    pyenv update

Then install the python version of this project using

    pyenv install 3.10.11

Now you should have the required python version.

### virtual environment

To install the virtual environment run

    make install

Now you should be ready to interact with repo, e.g. run the notebooks in `nbs/` or execute `pytest -vx .`.

To changing requirements with `Makefile` and `pip-tools` first edit `pyproject.toml` and then run

    make compile

This will create a new `config/requirements.txt` file and not yet install those packages.

Install the dependencies as in `config/requirements.txt` run

    make update
