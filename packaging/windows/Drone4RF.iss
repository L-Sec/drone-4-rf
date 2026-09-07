#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #error SourceDir must point to the packaged Drone4RF application directory
#endif
#ifndef OutputDir
  #define OutputDir "."
#endif

[Setup]
AppId={{70C1B60D-4506-4B43-8914-7D571644B373}
AppName=Drone 4-RF
AppVersion={#AppVersion}
AppPublisher=L-Sec
DefaultDirName={localappdata}\Programs\Drone 4-RF
DefaultGroupName=Drone 4-RF
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=Drone-4-RF-{#AppVersion}-Windows-x64-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
LicenseFile={#SourceDir}\LICENSE.txt
UninstallDisplayIcon={app}\Drone4RF.exe
SetupLogging=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Drone 4-RF"; Filename: "{app}\Drone4RF.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Drone 4-RF"; Filename: "{app}\Drone4RF.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Drone4RF.exe"; Description: "Launch Drone 4-RF"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Per-user observations and configuration live under LocalAppData\Drone4RF and
; are intentionally retained. Users must remove that directory deliberately.
