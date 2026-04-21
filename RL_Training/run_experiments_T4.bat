@echo off
REM Run MAPPO and PPO sequentially on a single GPU (T4).
REM Both experiments share cuda:0 — PPO runs after MAPPO finishes.

set TRAIN_DIR=%~dp0

echo Starting MAPPO on cuda:0 ...
python "%TRAIN_DIR%train.py" --algo MAPPO --device cuda:0

echo.
echo MAPPO finished. Starting PPO on cuda:0 ...
python "%TRAIN_DIR%train.py" --algo PPO --device cuda:0

echo.
echo Both experiments complete.
pause
