@echo off
chcp 65001 >nul
title QA & Manual Label Editor Server (Port 8899)
cd /d "E:\trend_AI"

echo ============================================================
echo   KHOI DONG CONG CU KIEM TRA ^& SUA NHAN DRONE (QA WEB UI)
echo   Dia chi Web: http://localhost:8899
echo   (Giu cua so nay mo khi dang su dung. Nhan Ctrl+C de tat)
echo ============================================================

:: Kiem tra neu port 8899 da chay san thi mo trinh duyet luon
netstat -ano | findstr ":8899 " | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo Server dang chay san tren cong 8899! Dang mo trinh duyet...
    start "" "http://localhost:8899"
    timeout /t 3 >nul
    exit /b 0
)

:: Mo trinh duyet sau 2 giay va khoi chay server Python
start "" cmd /c "timeout /t 2 >nul & start http://localhost:8899"
python -u "E:\trend_AI\step3_qa_server.py"
pause
