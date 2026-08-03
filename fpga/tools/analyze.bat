@echo off
REM analyze.bat - VM-side routed checkpoint analysis dispatcher.
REM   analyze.bat summary|deep|gemm-acc <checkpoint.dcp> <output-dir>

set "MODE=%~1"
set "CHECKPOINT=%~2"
set "OUTPUT_DIR=%~3"
if "%MODE%"=="" goto usage
if "%CHECKPOINT%"=="" goto usage
if "%OUTPUT_DIR%"=="" goto usage

set "VIVADO_SETTINGS=C:\AMDDesignTools\2025.2.1\Vivado\settings64.bat"
if not exist "%VIVADO_SETTINGS%" (
  echo ERROR: VIVADO_SETTINGS not found: %VIVADO_SETTINGS%
  exit /b 1
)
call "%VIVADO_SETTINGS%"

if "%MODE%"=="summary" goto summary
if "%MODE%"=="deep" goto deep
if "%MODE%"=="gemm-acc" goto gemm_acc
goto usage

:summary
call vivado -mode batch -source tools/summarize.tcl -tclargs "%CHECKPOINT%" "%OUTPUT_DIR%"
exit /b %errorlevel%

:deep
call vivado -mode batch -source tools/report.tcl -tclargs "%CHECKPOINT%" "%OUTPUT_DIR%"
exit /b %errorlevel%

:gemm_acc
call vivado -mode batch -source tools/report_gemm_acc.tcl -tclargs "%CHECKPOINT%" "%OUTPUT_DIR%"
exit /b %errorlevel%

:usage
echo usage: analyze.bat summary^|deep^|gemm-acc checkpoint.dcp output-dir
exit /b 2
