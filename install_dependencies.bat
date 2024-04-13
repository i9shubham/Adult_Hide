@echo off
cd /D "%~dp0"
conda install --file requirements.txt
PAUSE