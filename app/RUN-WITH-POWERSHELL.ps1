Set-Variable -Name tbpath -Value $PSScriptRoot\songexplorer

Set-Variable -Name SONGEXPLORER_BIN -Value ($tbpath + ";" + $tbpath + "\Library\mingw-w64\bin;" + $tbpath + "\Library\usr\bin;" + $tbpath + "\Library\bin;" + $tbpath + "\Scripts;" + $tbpath + "\bin;")
 
$env:Path = $SONGEXPLORER_BIN + $env:Path

python.exe $PSScriptRoot\songexplorer\bin\songexplorer\src\songexplorer $PSScriptRoot\configuration.py 8080

sleep 10
