if [ `uname -m` == arm64 ] ; then
    python -m pip install –U pip
    python -m pip install tensorflow-macos
    python -m pip install tensorflow-metal
fi

# https://github.com/soft-matter/pims/issues/425
pip3 install -e git+https://github.com/soft-matter/pims.git@7bd634015ecbfeb7d92f9f9d69f8b5bb4686a6b4#egg=pims -t $PREFIX/lib/python3.8/site-packages --upgrade --no-deps
