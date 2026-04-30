$venvPython = "d:\Web Projects\PalmRAG\PRISM-Palm-Reasoning-Interpretation-System-with-Models\Jupyter Notebook Implementation\env\Scripts\python.exe"
if (-Not (Test-Path $venvPython)) {
    Write-Error "Virtualenv python not found: $venvPython"
    exit 1
}
& $venvPython -m uvicorn main:app --reload
