setlocal

set tbpath=%~dp0\songexplorer

SET PATH=%tbpath%;%tbpath%\bin\songexplorer\src;%tbpath%\Library\mingw-w64\bin;%tbpath%\Library\usr\bin;%tbpath%\Library\bin;%tbpath%\Scripts;%tbpath%\bin;%PATH%

python.exe %~dp0\make-predictions.py %*

sleep 10
