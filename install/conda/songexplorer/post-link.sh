cat << EOF >> ${PREFIX}/.messages.txt
********** IMPORTANT !!! **********
EOF

if [ `uname` == Darwin ] ; then
cat << EOF >> ${PREFIX}/.messages.txt
cut and paste the following into the command line to finish the installation:
conda activate songexplorer
pip3 install tensorflow
EOF
if [ `uname -m` == arm64 ] ; then
cat << EOF >> ${PREFIX}/.messages.txt
pip3 install tensorflow-metal
EOF
fi
cat << EOF >> ${PREFIX}/.messages.txt
optionally, to use video in songexplorer cut and paste the following too:
EOF
else
cat << EOF >> ${PREFIX}/.messages.txt
optionally, to use video in songexplorer cut and paste the following into the command line:
conda activate songexplorer
EOF
fi

cat << EOF >> ${PREFIX}/.messages.txt
conda install av=8.1 git
pip3 install -e git+https://github.com/soft-matter/pims.git@7bd634015ecbfeb7d92f9f9d69f8b5bb4686a6b4#egg=pims
********** IMPORTANT !!! **********
EOF
