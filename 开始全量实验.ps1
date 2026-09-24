param(
    [string]$Name = ('full_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [int]$Workers = 2
)
Set-Location -LiteralPath $PSScriptRoot
& '.\.venv\Scripts\python.exe' -X utf8 -B run.py solve --name $Name --cases all --cores 5 --scenes 1 2 3 --device cuda --workers $Workers
exit $LASTEXITCODE
