@echo off
setlocal EnableExtensions
set "PROJECT_ROOT=%~dp0.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"
set "PYTHONUTF8=1"
set "CH4_DATA_ROOT=%PROJECT_ROOT%\data\chapter4_final"
if "%~1"=="" (set "MODE=run") else (set "MODE=%~1")
set "CH4_EXPERIMENT_FAMILY=stress_tests"
set "CH4_EXPERIMENT_STAGE=evaluation"
set "CH4_EXPERIMENT_ID=ch4_perf_rbe_vs_uniform_dr_stress_tiers_v1"
set "CH4_RUN_NAME=frozen_perf_rbe_vs_uniform_dr"
set "CH4_SEED=381,382,383"
set "CH4_PROTOCOL=radial_full9d_stress_tiers_v1"
set "CONSOLE_LOG_ROOT=%CH4_DATA_ROOT%\manifests\console_logs\%CH4_EXPERIMENT_ID%"
set "CONSOLE_LOG=%CONSOLE_LOG_ROOT%\%MODE%.console.log"
echo [StressTier] bat_version=20260801-v1
echo [StressTier] project_root=%PROJECT_ROOT%
echo [StressTier] mode=%MODE%
echo [StressTier] levels=0.00,0.25,0.50,0.75,1.00,1.25,1.50
echo [StressTier] seeds=%CH4_SEED%
echo [StressTier] formal_episodes=4200
python tools\ch4_run_and_log.py --log "%CONSOLE_LOG%" -- python tools\_internal\eval_ch4_perf_rbe_vs_uniform_dr_stress_tiers.py %MODE%
exit /b %ERRORLEVEL%
