# random-tree-models
My implementation of a random selection of tree based models.

## `pyenv`

Install pyenv using [this guide](https://github.com/pyenv/pyenv#installation).

If you have it already installed or newly installed run

    pyenv update

Then install the python version of this project using 

    pyenv install 3.10.11

## virtual environment

### install from scratch

    make install

### changing requirements

First edit `pyproject.toml` and then run

    make compile

This will create a new `config/requirements.txt` file.

### updating the virtual environment

Install `config/requirements.txt` using 

    make update