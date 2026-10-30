@echo off
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
  set PY=py
) else (
  set PY=python
)
if not exist trend_test.py (
  echo trend_test.py is not in this folder.
  pause
  exit /b
)
if not exist discover_v2.py (
  echo discover_v2.py is not in this folder.
  pause
  exit /b
)
if not exist upbit_daily (
  echo upbit_daily folder is not here.
  pause
  exit /b
)
%PY% -m pip install pandas numpy scikit-learn
echo Running trend test (about one minute)...
%PY% trend_test.py > run_trend_log.txt 2>&1
type run_trend_log.txt
echo.
echo Finished. Send results_trend.txt  (if it is missing, send run_trend_log.txt)
pause
