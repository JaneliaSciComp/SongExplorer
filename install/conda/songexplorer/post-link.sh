cat << EOF >> ${PREFIX}/.messages.txt
********** IMPORTANT !!! **********
cut and paste the following into the command line to finish the installation:
conda activate songexplorer
pip3 install -e git+https://github.com/soft-matter/pims.git@7bd634015ecbfeb7d92f9f9d69f8b5bb4686a6b4#egg=pims -t $PREFIX/lib/python3.8/site-packages --upgrade --no-deps
EOF

if [ `uname -m` == arm64 ] ; then
cat << EOF >> ${PREFIX}/.messages.txt
pip3 install tensorflow tensorflow-metal nitime
EOF
fi

cat << EOF >> ${PREFIX}/.messages.txt
********** IMPORTANT !!! **********
EOF
