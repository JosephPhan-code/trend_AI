@echo off
chcp 65001 >nul
title YOLO26s-P2 INT8 Drone Detection & Tracking Test Web (Port 8888)
cd /d "E:\trend_AI"

echo ============================================================
echo   KHOI DONG WEB TEST DRONE DETECTION ^& TRACKING (INT8)
echo   Dia chi Web: http://localhost:8888
echo   (Giu cua so nay mo khi dang su dung. Nhan Ctrl+C de tat)
echo ============================================================

:: Kiem tra neu port 8888 da chay san thi mo trinh duyet luon
netstat -ano | findstr ":8888 " | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo Server dang chay san tren cong 8888! Dang mo trinh duyet...
    start "" "http://localhost:8888"
    timeout /t 3 >nul
    exit /b 0
)

:: Mo trinh duyet sau 2 giay va khoi chay server Python
start "" cmd /c "timeout /t 2 >nul & start http://localhost:8888"
python -u "E:\trend_AI\drone_test_web\app.py" --port 8888
pause
