mkdir $PREFIX/songexplorer
cp -R $SRC_DIR/* $PREFIX/songexplorer
mkdir $PREFIX/bin
executables=(songexplorer hetero accuracy.py activations.py classify.py cluster.py compare.py congruence.py ensemble.py ethogram.py freeze.py generalize.py loop.py misses.py mistakes.py time-freq-threshold.py train.py xvalidate.py)
for executable in ${executables[*]}; do
    ln -s $PREFIX/songexplorer/src/$executable $PREFIX/bin/$executable
done
ln -s $PREFIX/songexplorer/test/runtests.sh $PREFIX/bin/runtests.sh
