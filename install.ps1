$ErrorActionPreference = 'Stop'

$RepositoryRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $RepositoryRoot

python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
& .\.venv\Scripts\python.exe -m pip install -r requirements-full.txt
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe -m pip install -e third_party\sam2

Write-Host 'Installation complete.'
Write-Host 'Tests: .\.venv\Scripts\python.exe -m unittest discover -s tests -v'
Write-Host 'Model evaluation (after external assets are placed): .\.venv\Scripts\python.exe evaluate_representative.py'
