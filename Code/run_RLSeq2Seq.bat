@echo off
setlocal EnableDelayedExpansion

REM Move to the folder containing this BAT file and the Python scripts
cd /d "%~dp0"

REM -------------------------------
REM Environment setup
REM -------------------------------
if not exist Logs mkdir Logs

REM Generate timestamp-based job id shared by this batch job
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set JOB_ID=%%i

REM -------------------------------
REM Common run settings
REM -------------------------------
REM launcher.py expects data as: %%DATA_LOC%%\%%DATA_NAME%%\train.txt, valid.txt, test.txt, rest.txt
set DATA_LOC=..\Data

set EPOCHS=50
set BATCH_SIZE=128
set MAX_SAMPLE=4
set NEG_SAMPLE_SIZE=31

REM These must match keys registered in Trainer.py
set ENCODER_NAME=Seq2seq
set DECODER_NAME=Seq2seq

REM seq_mode=[orig|rpw|OTSeq2Set|RL]
set SEQ_MODE=RL

REM seq_model = [RNN|LSTM|GRU]
set SEQ_MODEL=RNN

REM Output/log folders
set OUT_LOC=..\Logs

REM -------------------------------
REM Run each dataset
REM -------------------------------
for %%D in (FB15k-237 Nell-995 Nell-1115) do (
    set DATA_NAME=%%D

    set LOG_FILE=Logs\!DATA_NAME!_!ENCODER_NAME!_!SEQ_MODE!_!SEQ_MODEL!_!JOB_ID!.log

    echo.
    echo ============================
    echo Starting dataset: !DATA_NAME!
    echo Log file: !LOG_FILE!
    echo ============================

    python -u launcher.py ^
      --data_loc "!DATA_LOC!" ^
      --data_name "!DATA_NAME!" ^
      --n_epochs !EPOCHS! ^
      --batch_size !BATCH_SIZE! ^
      --No_max_sample !MAX_SAMPLE! ^
      --Neg_sample_size !NEG_SAMPLE_SIZE! ^
      --encoder_name "!ENCODER_NAME!" ^
      --decoder_name "!DECODER_NAME!" ^
      --out_loc "!OUT_LOC!" ^
      --seq_mode "!SEQ_MODE!" ^
      --seq_model "!SEQ_MODEl!" ^
      --seed 10 ^
      1> "!LOG_FILE!" 2>&1

    if errorlevel 1 (
        echo.
        echo ============================
        echo FAILED dataset: !DATA_NAME!
        echo See log file:
        echo !LOG_FILE!
        echo ============================
        type "!LOG_FILE!"
        goto :END
    ) else (
        echo.
        echo ============================
        echo Completed dataset: !DATA_NAME!
        echo Log file:
        echo !LOG_FILE!
        echo ============================
        type "!LOG_FILE!"
    )
)

echo.
echo ============================
echo All datasets completed!
echo Job ID: %JOB_ID%
echo ============================

:END
pause
endlocal
