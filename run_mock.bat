@echo off
chcp 65001 >nul
rem Запуск БЕЗ языковой модели (офлайн). Панель: http://localhost:4000
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv || goto :err
  .venv\Scripts\python -m pip install -q -r requirements.txt || goto :err
)
set LLM_PROVIDER=mock
.venv\Scripts\python -m uvicorn app.main:app --port 4000
goto :eof
:err
echo Не удалось подготовить окружение. Нужен Python 3.10+ (python.org).
pause
