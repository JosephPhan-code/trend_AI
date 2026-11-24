@echo off
cd /d "%~dp0"
echo ==================================================================
echo   FINETUNE 15 EPOCHS CHONG BAO GIA BAU TROI (WINDOWS CUDA)
echo   +10,000 anh nen bau troi rong (chim, may bay, may, dinh ang-ten)
echo ==================================================================
echo   Vi du: run_finetune_15ep_win.bat --weights path\to\best.pt --mode focused
echo ==================================================================
python -u finetune_sky_hn_15ep.py --epochs 15 %*
pause
