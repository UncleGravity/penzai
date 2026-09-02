@echo off
REM build.bat - run on the Windows VM (Vivado). Invoked by build.sh over ssh, or
REM directly:  build.bat f225 clean <run-id> <git> <dirty> <source-hash>
REM Produces out\runs\<run-id>\penzai-<variant>.bit(.bin)

set "VARIANT=%~1"
if "%VARIANT%"=="" set "VARIANT=f225"
set "BUILD_MODE=%~2"
if "%BUILD_MODE%"=="" set "BUILD_MODE=clean"
set "RUN_ID=%~3"
if "%RUN_ID%"=="" set "RUN_ID=manual-%VARIANT%"
set "GIT_COMMIT=%~4"
if "%GIT_COMMIT%"=="" set "GIT_COMMIT=unknown"
set "GIT_DIRTY=%~5"
if "%GIT_DIRTY%"=="" set "GIT_DIRTY=unknown"
set "SOURCE_HASH=%~6"
if "%SOURCE_HASH%"=="" set "SOURCE_HASH=unknown"
if not "%BUILD_MODE%"=="clean" if not "%BUILD_MODE%"=="incremental" (
  echo ERROR: build mode must be clean or incremental
  exit /b 1
)

set "VIVADO_SETTINGS=C:\AMDDesignTools\2025.2.1\Vivado\settings64.bat"
set "BIT_PREFIX=penzai"
set "BIT_NAME=%BIT_PREFIX%-%VARIANT%"
set "OUT_DIR=out\runs\%RUN_ID%"

cd /d "%~dp0"

if not exist "%VIVADO_SETTINGS%" (
  echo ERROR: VIVADO_SETTINGS not found: %VIVADO_SETTINGS%
  echo Edit build.bat and set it to your Vivado settings64.bat
  exit /b 1
)
call "%VIVADO_SETTINGS%"

echo == [1/2] Vivado: synth/impl/bitstream variant=%VARIANT% mode=%BUILD_MODE% ==
call vivado -mode batch -source build.tcl -tclargs %VARIANT% %BUILD_MODE% %RUN_ID% %GIT_COMMIT% %GIT_DIRTY% %SOURCE_HASH%
if errorlevel 1 ( echo Vivado build failed & exit /b 1 )

echo == [2/2] bootgen: %BIT_NAME%.bit -^> %BIT_NAME%.bit.bin ==
> "%OUT_DIR%\bit.bif" echo all:
>> "%OUT_DIR%\bit.bif" echo {
>> "%OUT_DIR%\bit.bif" echo     [destination_device = pl] %OUT_DIR%/%BIT_NAME%.bit
>> "%OUT_DIR%\bit.bif" echo }
call bootgen -arch zynqmp -image "%OUT_DIR%\bit.bif" -process_bitstream bin -w on
if errorlevel 1 ( echo bootgen failed & exit /b 1 )
if not exist "%OUT_DIR%\%BIT_NAME%.bit.bin" ( echo ERROR: no %OUT_DIR%\%BIT_NAME%.bit.bin produced & exit /b 1 )

echo.
echo Done. Outputs in %OUT_DIR%\ :
echo   %BIT_NAME%.bit
echo   %BIT_NAME%.bit.bin
echo.
exit /b 0
