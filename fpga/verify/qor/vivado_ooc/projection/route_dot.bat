@echo off
call "C:\AMDDesignTools\2025.2.1\Vivado\settings64.bat" >nul
call vivado -mode batch -source route_dot.tcl -tclargs %*
exit /b %errorlevel%
