# random-tree-models

> Implementation of a random selection of tree based models.

## layout

* `src/random_tree_models/` -> python implementation of tree algorithms
* `tests/` -> unit tests
* `nbs/`
  * `nbs/core` -> core jupyter notebooks to play with the tree algorithms
  * `nbs/dev` -> other jupyter notebooks for other activities
* `config/` -> requirements.txt

## setup

This project is managed with [`uv`](https://docs.astral.sh/uv/getting-started/installation/) using the maturin backend to compile rust and create python bindings. To develop you'll need the [rust toolchain](https://www.rust-lang.org/tools/install) as well.

### virtual environment

To install the virtual environment for development run

    make install-dev

Now you should be ready to interact with repo, e.g. run the notebooks in `nbs/` or execute `make test`.

After changing rust code it needs to be compiled. This is taken care of with

    make update
