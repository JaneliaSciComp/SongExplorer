if [ `uname -m` == arm64 ] ; then
    python -m pip install –U pip
    python -m pip install tensorflow-macos
    python -m pip install tensorflow-metal
fi
