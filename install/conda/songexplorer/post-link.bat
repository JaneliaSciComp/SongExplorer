: https://github.com/conda-forge/tensorflow-feedstock/issues/124
: google supports only WSL2 for tensorflow >2.10

python -m pip install –U pip
python -m pip install "tensorflow<2.11"

: https://github.com/soft-matter/pims/issues/425
pip3 install -e git+https://github.com/soft-matter/pims.git@7bd634015ecbfeb7d92f9f9d69f8b5bb4686a6b4#egg=pims -t $PREFIX/lib/python3.8/site-packages --upgrade --no-deps
