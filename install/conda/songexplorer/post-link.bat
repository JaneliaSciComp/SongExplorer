: https://github.com/conda-forge/tensorflow-feedstock/issues/124
: google supports only WSL2 for tensorflow >2.10

echo ********** IMPORTANT !!! ********** >> %PREFIX%\.messages.txt
echo optionally, to use video in songexplorer cut and paste the following into the command line: >> %PREFIX%\.messages.txt
echo conda activate songexplorer >> %PREFIX%\.messages.txt
echo conda install av=8.1 git >> %PREFIX%\.messages.txt
echo cd %%CONDA_PREFIX%%\Lib\site-packages >> %PREFIX%\.messages.txt
echo pip3 install -e git+https://github.com/soft-matter/pims.git@7bd634015ecbfeb7d92f9f9d69f8b5bb4686a6b4#egg=pims >> %PREFIX%\.messages.txt
echo either way, to finish the installation cut and paste the following: >> %PREFIX%\.messages.txt
echo conda activate songexplorer >> %PREFIX%\.messages.txt
echo pip3 install "tensorflow<2.11" >> %PREFIX%\.messages.txt
echo ********** IMPORTANT !!! ********** >> %PREFIX%\.messages.txt
