@echo off
REM Registered engine OOC route entry point for the Vivado VM.
call "C:\AMDDesignTools\2025.2.1\Vivado\settings64.bat" >nul
call vivado -mode batch -source route.tcl -tclargs %*
exit /b %errorlevel%
