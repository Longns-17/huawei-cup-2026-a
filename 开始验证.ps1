param(
    [string]$Name = ('validation_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [int]$Workers = 2
)
Set-Location -LiteralPath $PSScriptRoot
& '.\.venv\Scripts\python.exe' -X utf8 -B run.py validate --name $Name --workers $Workers
exit $LASTEXITCODE
