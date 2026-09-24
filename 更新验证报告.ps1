Set-Location -LiteralPath $PSScriptRoot
& '.\.venv\Scripts\python.exe' -X utf8 -B run.py verify
exit $LASTEXITCODE
