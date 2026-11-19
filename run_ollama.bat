@echo off
chcp 65001 >nul
rem Запуск С локальной моделью через Ollama. Панель: http://localhost:4000
rem Модель можно сменить:  set OLLAMA_MODEL=deepseek-coder-v2:16b  (до запуска этого файла)
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv || goto :err
  .venv\Scripts\python -m pip install -q -r requirements.txt || goto :err
)
set LLM_PROVIDER=ollama
if "%OLLAMA_MODEL%"=="" set OLLAMA_MODEL=qwen3.5:latest
echo Модель: %OLLAMA_MODEL%  (проверьте, что она есть в "ollama list")
.venv\Scripts\python -m uvicorn app.main:app --port 4000
goto :eof
:err
echo Не удалось подготовить окружение. Нужен Python 3.10+ (python.org).
pause
