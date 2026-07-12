# Truy cập từ mạng LAN — dùng hostname cố định

Mục tiêu: máy khác trong nhà/văn phòng mở được Chat UI và gọi được Gateway, bằng một
**địa chỉ không bao giờ đổi** — không phải đi hỏi lại IP mỗi lần router cấp lại DHCP.

## Cái gì được lộ ra, cái gì không

| Service | Port | Bind | Ra LAN? | Bảo vệ bằng |
|---|---|---|---|---|
| Open WebUI (Chat UI) | 3000 | `0.0.0.0` | ✅ | `WEBUI_AUTH=true` — bắt đăng nhập |
| LiteLLM (Gateway) | 4000 | `0.0.0.0` | ✅ | `LITELLM_MASTER_KEY` — bắt Bearer token |
| vLLM (engine thô) | 8000 | `127.0.0.1` | ❌ | — (không lộ) |

vLLM **cố tình** giữ localhost-only. Client trong LAN đi qua gateway `:4000` để còn được
master-key, retry, fallback và log tập trung. Muốn lộ thẳng vLLM: đặt `VLLM_BIND=0.0.0.0`
trong `.env` (không khuyến nghị).

Cả 3 bind address đều là biến `.env` — `BIND_ADDR=127.0.0.1` rút toàn bộ stack về localhost-only.

## Tại sao hostname chứ không phải IP

IP của máy host là **DHCP** — router cấp lại là đổi, và mọi chỗ hardcode IP sẽ chết.
Hostname qua **mDNS** thì tự phân giải sang IP hiện tại, đổi bao nhiêu lần cũng đúng.

- **Windows 10/11**: có sẵn mDNS responder, tự trả lời `<COMPUTERNAME>.local`. Không cần cài gì.
- **Linux**: cần `avahi-daemon` (`sudo apt install avahi-daemon`) → `<hostname>.local`.

Máy này: `LAN_HOST=thinhphat.local` (khai trong `.env`).

## Cài đặt (một lần)

### 1. Mở firewall

Docker đã bind port ra `0.0.0.0` rồi, nhưng **Windows Firewall chặn inbound** — nhất là khi
network profile là `Public` (đúng trường hợp Wi-Fi của máy này). Đây gần như luôn là lý do
"máy khác vào không được".

Mở **PowerShell với quyền Administrator**:

```powershell
cd G:\work\llm-platform
powershell -ExecutionPolicy Bypass -File scripts\lan-firewall.ps1
```

Script mở TCP 3000 + 4000, giới hạn `-RemoteAddress LocalSubnet` → **chỉ máy cùng subnet**
gọi được, internet không đụng tới. Gỡ bỏ: thêm cờ `-Remove`.

### 2. Kiểm tra từ máy khác

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://thinhphat.local:3000
# → 200

curl -sS http://thinhphat.local:4000/v1/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" | jq '.data[].id'
# → brain, gpt-4o, brain-pro
```

Trỏ agent/app từ máy khác vào gateway:

```bash
export GATEWAY_URL=http://thinhphat.local:4000/v1
export LITELLM_MASTER_KEY=sk-...        # lấy từ .env của host
```

## Phát key riêng cho từng người (đừng chia sẻ master key)

Master key là quyền admin — ai cầm nó cũng tạo/xoá được key khác. Đừng đưa cho người dùng LAN.
Thay vào đó phát **virtual key** cho từng người/máy/app từ admin UI:

1. Mở `http://thinhphat.local:4000/ui` → đăng nhập bằng `UI_USERNAME` / `UI_PASSWORD` (xem `.env`).
   Đây là tài khoản **khác** với master key.
2. **Virtual Keys → + Create New Key** → chọn model được phép (`brain`, `gpt-4o`...), đặt hạn mức
   (budget / rate limit) nếu cần.
3. Đưa key `sk-...` đó cho người dùng. Họ dùng y như master key:
   ```bash
   export LITELLM_MASTER_KEY=sk-<virtual-key-cua-ho>
   ```

Lợi ích: xem được ai gọi bao nhiêu, chặn model đắt (`brain-pro`) với người không cần, và **thu hồi
từng key lẻ** mà không phải đổi master key rồi đi cấu hình lại toàn bộ máy.

UI này cần Postgres (service `postgres` trong compose). Không có DB thì login báo
`Authentication Error, Not connected to DB!`.

## Khi hostname không resolve

Máy này có **4 network adapter** (Wi-Fi LAN, WSL, VirtualBox, Tailscale), nên mDNS trả về
cả 4 IP cho `thinhphat.local`:

```
192.168.1.33     ← LAN thật, cái ta cần
172.29.16.1      ← WSL vEthernet
192.168.56.1     ← VirtualBox
100.75.67.51     ← Tailscale
```

Client thường thử lần lượt tới khi có cái chạy, nên phần lớn vẫn vào được — chỉ là có thể
chậm vài giây ở lần đầu. Nếu client cũ không chịu retry:

- **Cách nhanh** — ghim IP ở máy client (`/etc/hosts`, hoặc `C:\Windows\System32\drivers\etc\hosts`):
  ```
  192.168.1.33  thinhphat.local
  ```
  Nhược điểm: IP DHCP đổi là phải sửa lại — mất đúng cái lợi ta đang tìm.

- **Cách bền** — vào router đặt **DHCP reservation** cho MAC của card Wi-Fi máy này, ghim
  cứng `192.168.1.33`. Lúc đó hostname *và* IP đều cố định, đường nào cũng đúng.

- **Ngoài LAN** — máy này đã có Tailscale (`100.75.67.51`). Bật MagicDNS thì gọi được từ
  bất cứ đâu mà không cần mở port ra internet.

## Bảo mật

Cả hai cổng lộ ra LAN đều có auth (`WEBUI_AUTH`, `LITELLM_MASTER_KEY`) và rule firewall
giới hạn `LocalSubnet`. Hai điều cần nhớ:

- Traffic là **HTTP thuần** — master key đi trên dây không mã hoá. Trong LAN nhà thì chấp
  nhận được; ở mạng dùng chung (quán cà phê, co-working) thì `LocalSubnet` vẫn đồng nghĩa
  với "mọi người trong cùng mạng đó" → nên tắt bằng `lan-firewall.ps1 -Remove`.
- **Đừng port-forward** 3000/4000 ra internet. Cần truy cập từ xa thì dùng Tailscale.
