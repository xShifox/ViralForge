# Create a ViralForge launcher shortcut that carries the app icon.
#
# A .bat, .vbs or .ps1 cannot have its own icon: Windows draws the icon registered
# for the file type, which is a machine-wide setting. Changing that association to
# give one project file a different icon would alter every .bat on the computer.
#
# A shortcut can carry any icon, so this points one at the console-free launcher
# (ViralForge.vbs) and gives it the same mark as the rest of the product. This is
# generated rather than committed so the icon and the target cannot drift apart.
#
#   powershell -ExecutionPolicy Bypass -File scripts\make_launcher_shortcut.ps1

$ErrorActionPreference = 'Stop'

$root    = Split-Path -Parent $PSScriptRoot
# The .ico sits at the project root, not in assets/: it is the same file
# desktop.ini points at for the folder icon, and one icon should be one file.
$icon    = Join-Path $root 'clipforge-folder.ico'
$launcher = Join-Path $root 'ViralForge.vbs'
$wscript = Join-Path $env:SystemRoot 'System32\wscript.exe'

if (-not (Test-Path $launcher)) {
    throw "launcher not found: $launcher"
}
if (-not (Test-Path $icon)) {
    throw "icon not found: $icon - run scripts\make_folder_icon.py first"
}

$targets = @(
    @{ Path = (Join-Path $root 'ViralForge.lnk');      Desktop = $false },
    @{ Path = (Join-Path $env:USERPROFILE 'Desktop\ViralForge.lnk'); Desktop = $true }
)

$ws = New-Object -ComObject WScript.Shell

foreach ($target in $targets) {
    $dir = Split-Path -Parent $target.Path
    if (-not (Test-Path $dir)) { continue }

    $sc = $ws.CreateShortcut($target.Path)
    $sc.TargetPath       = $wscript
    $sc.Arguments        = '"' + $launcher + '"'
    $sc.WorkingDirectory = $root
    $sc.IconLocation     = "$icon,0"
    $sc.Description      = 'ViralForge - long video to vertical short clips'
    $sc.WindowStyle      = 7          # minimised; the VBS already hides the console
    $sc.Save()

    $size = (Get-Item $target.Path).Length
    Write-Output ("  {0}  ({1} bytes)" -f $target.Path, $size)
}

Write-Output ''
Write-Output '  start.bat, start.ps1 and ViralForge.vbs are left in place as the'
Write-Output '  underlying launchers; the shortcut is the icon-carrying front door.'
