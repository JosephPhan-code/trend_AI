@echo off
title Drone Unified Dataset QA Server (Port 8901)
cd /d e:\trend_AI
python -X utf8 -u step3_qa_server.py --dir e:\trend_AI\drone_dataset_unified --port 8901
pause
