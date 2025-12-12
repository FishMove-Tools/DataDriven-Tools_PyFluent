echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="E:\Program Files\ANSYS Inc231\v231\fluent/ntbin/win64/winkill.exe"

"E:\Program Files\ANSYS Inc231\v231\fluent\ntbin\win64\tell.exe" DESKTOP-U4B4G55 56429 CLEANUP_EXITING
if /i "%LOCALHOST%"=="DESKTOP-U4B4G55" (%KILL_CMD% 38804) 
if /i "%LOCALHOST%"=="DESKTOP-U4B4G55" (%KILL_CMD% 33724) 
if /i "%LOCALHOST%"=="DESKTOP-U4B4G55" (%KILL_CMD% 30200) 
if /i "%LOCALHOST%"=="DESKTOP-U4B4G55" (%KILL_CMD% 45284) 
if /i "%LOCALHOST%"=="DESKTOP-U4B4G55" (%KILL_CMD% 33052) 
if /i "%LOCALHOST%"=="DESKTOP-U4B4G55" (%KILL_CMD% 33048)
del "E:\data_driven_cfd_test\Finial_Framework_version_4_action_fix\cfd_train_stage\fishmove\cleanup-fluent-DESKTOP-U4B4G55-33052.bat"
