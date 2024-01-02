@echo off
setlocal enabledelayedexpansion

REM Loop through all .dot files in the current directory
for %%i in (*.dot) do (
    REM Generate corresponding PNG file using Graphviz
    set "dot_file=%%i"
    set "png_file=!dot_file:.dot=.png!"
    dot -Tpng -Gdpi=300 "!dot_file!" -o "!png_file!"
    echo Generated !png_file!
)

endlocal
