@echo off
REM Start the PalmRAG backend using the project virtualenv Python.
"d:\Web Projects\PalmRAG\PRISM-Palm-Reasoning-Interpretation-System-with-Models\Jupyter Notebook Implementation\env\Scripts\python.exe" -m uvicorn main:app --reload
pause
