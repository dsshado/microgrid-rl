@echo off
REM Run MAPPO on GPU 0 and PPO on GPU 1 simultaneously.
REM Each experiment opens in its own terminal window.

set TRAIN_DIR=%~dp0

echo Starting MAPPO on cuda:0 ...
start "MAPPO-GPU0" cmd /k "cd /d %TRAIN_DIR% && python train.py --algo MAPPO --device cuda:0"

echo Starting PPO on cuda:1 ...
start "PPO-GPU1"   cmd /k "cd /d %TRAIN_DIR% && python train.py --algo PPO   --device cuda:1"

echo Both experiments launched. Close this window or press any key to exit.
pause
