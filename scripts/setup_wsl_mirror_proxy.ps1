param(
    [int] $Port = 9000,
    [string] $ListenAddress = "",
    [string] $WslAddress = ""
)

$ErrorActionPreference = "Stop"

function Get-PrimaryWindowsIPv4 {
    $candidate = Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object {
            $_.IPAddress -notlike "127.*" -and
            $_.IPAddress -notlike "169.254.*" -and
            $_.IPAddress -notlike "172.25.*" -and
            $_.InterfaceAlias -notlike "vEthernet*"
        } |
        Sort-Object SkipAsSource, InterfaceIndex |
        Select-Object -First 1

    if ($null -eq $candidate) {
        throw "Could not auto-detect the Windows LAN IPv4 address. Pass -ListenAddress explicitly."
    }

    return $candidate.IPAddress
}

function Get-WslIPv4 {
    $raw = wsl.exe -e sh -lc "hostname -I" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not query the WSL IPv4 address. Pass -WslAddress explicitly."
    }

    $address = ($raw | Out-String).Trim().Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries) |
        Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($address)) {
        throw "WSL did not return an IPv4 address. Pass -WslAddress explicitly."
    }

    return $address
}

if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)) {
    throw "Run this script from an elevated PowerShell window."
}

if ([string]::IsNullOrWhiteSpace($ListenAddress)) {
    $ListenAddress = Get-PrimaryWindowsIPv4
}

if ([string]::IsNullOrWhiteSpace($WslAddress)) {
    $WslAddress = Get-WslIPv4
}

Write-Host "Forwarding Windows $ListenAddress`:$Port -> WSL $WslAddress`:$Port"

netsh interface portproxy delete v4tov4 listenaddress=$ListenAddress listenport=$Port | Out-Null
netsh interface portproxy add v4tov4 listenaddress=$ListenAddress listenport=$Port connectaddress=$WslAddress connectport=$Port | Out-Null

$ruleName = "CognitiveSense Mirror $Port"
$existingRule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($null -eq $existingRule) {
    New-NetFirewallRule `
        -DisplayName $ruleName `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort $Port | Out-Null
}

Write-Host "Done."
Write-Host "Raspberry Pi target IP: $ListenAddress"
Write-Host "Mirror server port: $Port"
