@echo off
REM build.bat - run from cmd on the Windows VM (Vivado/Vitis 2025.2.1).
REM   build.bat
REM Produces out\loopback.bit and out\loopback.bit.bin
REM
REM CONFIRM this path (you said you call it manually). It must be the Vivado
REM settings64.bat; bootgen needs to be on PATH after it (else also call the
REM Vitis settings64.bat below).
set "VIVADO_SETTINGS=C:\AMDDesignTools\2025.2.1\Vivado\settings64.bat"
REM bootgen ships under Vivado\bin, so it is on PATH after the Vivado settings.

cd /d "%~dp0"

if not exist "%VIVADO_SETTINGS%" (
  echo ERROR: VIVADO_SETTINGS not found: %VIVADO_SETTINGS%
  echo Edit build.bat and set it to your Vivado 2025.2 settings64.bat
  exit /b 1
)
call "%VIVADO_SETTINGS%"

echo == [1/2] Vivado: synth/impl/bitstream ==
REM 'call' is REQUIRED: vivado is vivado.bat. Without call, control transfers
REM to it and never returns, so step 2 (bootgen) would silently never run.
call vivado -mode batch -source build.tcl
if errorlevel 1 ( echo Vivado build failed & exit /b 1 )

echo == [2/2] bootgen: loopback.bit -^> loopback.bit.bin ==
> out\bit.bif echo all:
>> out\bit.bif echo {
>> out\bit.bif echo     [destination_device = pl] out/loopback.bit
>> out\bit.bif echo }
REM 'call' again: bootgen is bootgen.bat. Output lands at out\loopback.bit.bin.
call bootgen -arch zynqmp -image out\bit.bif -process_bitstream bin -w on
if errorlevel 1 ( echo bootgen failed & exit /b 1 )
if not exist out\loopback.bit.bin ( echo ERROR: no out\loopback.bit.bin produced & exit /b 1 )

echo.
echo Done. Outputs in fpga\out\ :
echo   loopback.bit
echo   loopback.bit.bin
echo.
echo Next (from the host):
echo   zig build deploy
echo   zig build verify
exit /b 0
