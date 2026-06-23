@echo off
REM ooc.bat - run on the Windows Vivado VM. Sources Vivado settings then runs the
REM OOC synth probe. Invoked by run.sh over ssh:
REM   ooc.bat <top> <period_ns> <out_prefix> <rtl...>
call "C:\AMDDesignTools\2025.2.1\Vivado\settings64.bat" >nul
call vivado -mode batch -source ooc_synth.tcl -tclargs %*
exit /b %errorlevel%
