@echo off
chcp 65001 >nul
title QA & Manual Label Editor - DATA2 (219,270 anh - Port 8900)
cd /d "E:\trend_AI"

echo ============================================================
echo   CONG CU KIEM TRA ^& SUA NHAN DATA2 (219,270 ANH)
echo   Dia chi Web: http://localhost:8900
echo ============================================================

netstat -ano | findstr ":8900 " | findstr "LISTENING" >nul
if %errorlevel%==0 (
    start "" "http://localhost:8900"
    timeout /t 2 >nul
    exit /b 0
)

start "" cmd /c "timeout /t 3 >nul & start http://localhost:8900"
python -X utf8 -u "E:\trend_AI\step3_qa_server.py" --dir "e:\trend_AI\data2" --port 8900
pause
