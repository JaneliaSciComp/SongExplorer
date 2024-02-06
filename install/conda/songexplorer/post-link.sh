cat << EOF >> ${PREFIX}/.messages.txt
********** IMPORTANT !!! **********
optionally, to use video in songexplorer cut and paste the following into the command line:
conda activate songexplorer
conda install av=8.1 git
cd \$CONDA_PREFIX/lib/python3.10/site-packages
pip3 install -e git+https://github.com/soft-matter/pims.git@7bd634015ecbfeb7d92f9f9d69f8b5bb4686a6b4#egg=pims
either way, to finish the installation cut and paste the following:
EOF

if [ `uname` == Darwin ] ; then
cat << EOF >> ${PREFIX}/.messages.txt
conda activate songexplorer
pip3 install tensorflow[tensorflow-io]
EOF
if [ `uname -m` == arm64 ] ; then
cat << EOF >> ${PREFIX}/.messages.txt
pip3 install tensorflow-metal
EOF
fi
else
cat << EOF >> ${PREFIX}/.messages.txt
pip3 install tensorflow-io==<VERSION>  # https://github.com/tensorflow/io?tab=readme-ov-file#tensorflow-version-compatibility
EOF
fi

cat << EOF >> ${PREFIX}/.messages.txt
********** IMPORTANT !!! **********
EOF
