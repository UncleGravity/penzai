@echo off
REM build.bat - run on the Windows VM (Vivado). Invoked by build.sh over ssh, or
REM directly:  build.bat f100
REM Produces out\penzai-flash-v1-<variant>.bit(.bin)

set "VARIANT=%~1"
if "%VARIANT%"=="" set "VARIANT=f100"

set "VIVADO_SETTINGS=C:\AMDDesignTools\2025.2.1\Vivado\settings64.bat"
set "BIT_PREFIX=penzai-flash-v1"
set "BIT_NAME=%BIT_PREFIX%-%VARIANT%"

cd /d "%~dp0"

if not exist "%VIVADO_SETTINGS%" (
  echo ERROR: VIVADO_SETTINGS not found: %VIVADO_SETTINGS%
  echo Edit build.bat and set it to your Vivado settings64.bat
  exit /b 1
)
call "%VIVADO_SETTINGS%"

echo == [1/2] Vivado: synth/impl/bitstream variant=%VARIANT% ==
call vivado -mode batch -source build.tcl -tclargs %VARIANT%
if errorlevel 1 ( echo Vivado build failed & exit /b 1 )

echo == [2/2] bootgen: %BIT_NAME%.bit -^> %BIT_NAME%.bit.bin ==
> out\bit.bif echo all:
>> out\bit.bif echo {
>> out\bit.bif echo     [destination_device = pl] out/%BIT_NAME%.bit
>> out\bit.bif echo }
call bootgen -arch zynqmp -image out\bit.bif -process_bitstream bin -w on
if errorlevel 1 ( echo bootgen failed & exit /b 1 )
if not exist out\%BIT_NAME%.bit.bin ( echo ERROR: no out\%BIT_NAME%.bit.bin produced & exit /b 1 )

echo.
echo Done. Outputs in out\ :
echo   %BIT_NAME%.bit
echo   %BIT_NAME%.bit.bin
echo.
exit /b 0
