@echo off
chcp 65001 >nul

REM ====== Hugging Face 缓存路径 ======
set HF_HOME=%CD%\hf_cache
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

REM ====== 设置 Python 路径 ======
SET PYTHON_PATH=%cd%\WZF311
SET PYTHON_EXECUTABLE=%PYTHON_PATH%\python.exe

REM ====== 清理系统干扰 ======
set PYTHONNOUSERSITE=1
set PYTHONHOME=
set PYTHONPATH=

REM ====== ffmpeg 路径 ======
SET FFMPEG_PATH=%cd%\WZF311\ffmpeg\bin
SET PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%FFMPEG_PATH%;%PATH%

REM ====== 可选：静音 cl 编码告警 ======
set CL=/utf-8

REM ====== 启动 ======
"%PYTHON_EXECUTABLE%" -s webui.py
pause
