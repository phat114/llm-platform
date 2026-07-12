<#
.SYNOPSIS
  Mở Windows Firewall cho Chat UI (:3000) + Gateway (:4000) — CHỈ trong LAN.

.DESCRIPTION
  Docker Desktop đã bind port ra 0.0.0.0, nhưng Windows Firewall chặn inbound
  (nhất là khi network profile = Public). Script này thêm 2 rule inbound TCP.

  Phạm vi: -RemoteAddress LocalSubnet → chỉ máy CÙNG subnet gọi được.
  Internet/máy ngoài không đụng tới được, kể cả khi router có port-forward.

  vLLM (:8000) CỐ TÌNH không mở — engine thô, LAN phải đi qua gateway.

.NOTES
  PHẢI chạy bằng PowerShell Administrator:
    powershell -ExecutionPolicy Bypass -File scripts\lan-firewall.ps1
  Gỡ bỏ:
    powershell -ExecutionPolicy Bypass -File scripts\lan-firewall.ps1 -Remove
#>
[CmdletBinding()]
param(
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "LOI: Script can quyen Administrator." -ForegroundColor Red
    Write-Host "  Mo PowerShell (Run as administrator) roi chay lai:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\lan-firewall.ps1" -ForegroundColor Yellow
    exit 1
}

$rules = @(
    @{ Name = 'LLM Platform - Chat UI (Open WebUI)'; Port = 3000 },
    @{ Name = 'LLM Platform - Gateway (LiteLLM)';    Port = 4000 }
)

foreach ($r in $rules) {
    # Xoá rule cũ trước → script chạy lại nhiều lần không sinh rule trùng
    Get-NetFirewallRule -DisplayName $r.Name -ErrorAction SilentlyContinue |
        Remove-NetFirewallRule

    if ($Remove) {
        Write-Host ("[-] Da go rule: {0} (TCP {1})" -f $r.Name, $r.Port) -ForegroundColor Yellow
        continue
    }

    New-NetFirewallRule `
        -DisplayName   $r.Name `
        -Direction     Inbound `
        -Action        Allow `
        -Protocol      TCP `
        -LocalPort     $r.Port `
        -RemoteAddress LocalSubnet `
        -Profile       Any `
        -Description   'LLM Platform: cho phep may trong cung LAN truy cap. Xem docs/lan-access.md' | Out-Null

    Write-Host ("[+] Cho phep TCP {0} tu LocalSubnet  ({1})" -f $r.Port, $r.Name) -ForegroundColor Green
}

if ($Remove) {
    Write-Host "`nXong. Stack chi con truy cap duoc tu chinh may nay." -ForegroundColor Cyan
    exit 0
}

$lanHost = "$($env:COMPUTERNAME).local".ToLower()
$lanIp = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.PrefixOrigin -eq 'Dhcp' -and $_.IPAddress -notlike '169.254.*' } |
    Select-Object -First 1 -ExpandProperty IPAddress)

Write-Host "`nXong. Tu may khac trong LAN:" -ForegroundColor Cyan
Write-Host ("  Chat UI : http://{0}:3000" -f $lanHost)
Write-Host ("  Gateway : http://{0}:4000/v1   (can LITELLM_MASTER_KEY)" -f $lanHost)
if ($lanIp) {
    Write-Host ("`n  (Neu hostname khong resolve, dung IP tam: http://{0}:3000)" -f $lanIp) -ForegroundColor DarkGray
}
