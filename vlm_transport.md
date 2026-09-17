# VLM Target 傳輸介面

> 狀態：**規劃中，client 與 server 都還沒實作**。下面的預設值（port、逾時、縮圖寬度等）是暫定值，要等實測來回延遲後再調整。

## 1. 流程總覽

```
上級描述 node ──► server（保存目前的描述，編 query_version）
                    ▲                 │
      ① JPEG + id   │                 │ ② mask + id + query_version
                    │                 ▼
client (orb_tracker_node)
  ├─ 送圖時把那一幀（灰階、深度、當下 TF）存進本地快取
  ├─ 等待期間繼續用舊 target 追蹤
  └─ mask 回來：從快取取出同一幀 → 用 mask 建新 target → 在「現在這一幀」確認後才替換
```

設計原則：

- **server 只回傳 mask，不回傳圖片。** client 手上已經有送出去的那一幀，只需要知道物件在哪裡。
- **不需要兩台機器的時鐘同步。** 用 `request_id` 對應請求和回應，時間一律用 client 的時鐘量。
- **同一時間只有一筆請求在路上。** 避免 WiFi 變慢時請求越堆越多，延遲越來越大。

## 2. 傳輸方式：ZeroMQ over TCP

### 為什麼不用 ROS 2 topic / service

- 目前走 WiFi。DDS discovery 依賴 multicast，在 WiFi 上不穩定。
- [config/cyclonedds.xml](../config/cyclonedds.xml) 的 `MaxMessageSize` 是 65500B，一張影像會被切成多個 UDP 封包，WiFi 上只要掉一片就得整張重傳或整張丟掉。
- ROS 2 service 在網路中斷時容易卡住等不到回應。
- 換成 rmw_zenoh 會影響整台機器人的所有 node，而且專案用的是 Humble，binary 支援度需要另外確認。

### 為什麼選 ZeroMQ

- TCP 長連線，斷線會自動重連。
- 多段（multipart）二進位訊息，JPEG 和 PNG 直接傳，不需要 base64，體積也不會多約 33%。
- C++（libzmq）和 Python（pyzmq）都很輕量，server 端大概率是 Python。

### Socket 類型

| 端 | Socket | 動作 |
|---|---|---|
| server | `ROUTER` | `bind("tcp://*:5555")` |
| client | `DEALER` | `connect("tcp://<server_ip>:5555")` |

**不要用 `REQ/REP`。** `REQ` 必須嚴格一送一收，只要一個回應在 WiFi 上遺失，socket 就會卡在等待狀態。

### Socket 設定

**client（DEALER）**

| 選項 | 值 | 用途 |
|---|---|---|
| `ZMQ_LINGER` | `0` | 關閉時不等待未送出的訊息 |
| `ZMQ_IMMEDIATE` | `1` | 連線還沒建立時直接送失敗，不排隊 |
| `ZMQ_SNDHWM` | `1` | 送出佇列最多一筆 |
| `ZMQ_RCVTIMEO` / poll | 依 `vlm.timeout_s` | 逾時判斷 |
| `ZMQ_TCP_KEEPALIVE` | `1` | 偵測 WiFi 斷線 |
| `ZMQ_TCP_KEEPALIVE_IDLE` | `5` | 秒 |
| `ZMQ_TCP_KEEPALIVE_INTVL` | `2` | 秒 |
| `ZMQ_RECONNECT_IVL` | `500` | ms |

**server（ROUTER）**

| 選項 | 值 | 用途 |
|---|---|---|
| `ZMQ_LINGER` | `0` | 關閉時不等待未送出的訊息 |
| `ZMQ_TCP_KEEPALIVE` | `1` | 偵測 WiFi 斷線 |

### 訊息的 frame 結構

DEALER 送出時**不加**空的分隔 frame（這點和 REQ 不同）。

| 方向 | 在 client 端看到的 frame | 在 server ROUTER 看到的 frame |
|---|---|---|
| request | `[header, jpeg]` | `[identity, header, jpeg]` |
| response（FOUND） | `[header, mask_png]` | server 送出時為 `[identity, header, mask_png]` |
| response（其他） | `[header]` | server 送出時為 `[identity, header]` |

## 3. 網路與 port 設定

| 項目 | 預設 | 說明 |
|---|---|---|
| Port | `5555/tcp` | client 參數 `vlm.endpoint` 可以改 |
| Server IP | 待定 | 建議在 WiFi AP 上固定 IP，或在 router 設 DHCP 保留 |
| 防火牆 | server 要開放 `5555/tcp` 入站 | 例如 `sudo ufw allow 5555/tcp` |
| Docker | client container 已經是 `network_mode: host`，不需要另外映射 port | server 若跑在 container 裡要映射或用 host 網路 |
| ROS_DOMAIN_ID | 不受影響 | 這條連線不走 DDS |

**建議**：機器人上的 ROS 2 流量盡量不要經過 WiFi。如果 server 也要透過 ROS 2 接收上級描述，那是另一條連線，要另外評估。

## 4. 訊息格式

`header` 一律是 UTF-8 JSON。

### 4.1 Request：client → server

```json
{
  "protocol_version": 1,
  "type": "detect",
  "request_id": 42,
  "client_stamp": 1726540000.123,
  "width": 640,
  "height": 362,
  "known_query_version": 3
}
```

| 欄位 | 型別 | 說明 |
|---|---|---|
| `protocol_version` | int | 目前是 `1` |
| `type` | string | `detect` 或 `ping`（見 4.3） |
| `request_id` | uint32 | client 遞增。server 原封不動帶回 |
| `client_stamp` | float | 這一幀的 ROS 時間戳（秒）。只拿來記錄，server 不要用它做判斷 |
| `width`, `height` | int | 上傳 JPEG 的尺寸 |
| `known_query_version` | int | client 目前的 target 是根據哪一版描述建的。還沒有 target 時填 `-1` |

第二個 frame：JPEG bytes，BGR 彩色影像，已經依 `vlm.upload_max_width` 縮小過。

### 4.2 Response：server → client

```json
{
  "protocol_version": 1,
  "request_id": 42,
  "status": "FOUND",
  "query_version": 3,
  "bbox": [212, 98, 120, 160],
  "score": 0.87,
  "server_ms": 850,
  "error": ""
}
```

| 欄位 | 型別 | 說明 |
|---|---|---|
| `request_id` | uint32 | 照抄 request 的值 |
| `status` | string | 見下表 |
| `query_version` | int | server 目前的描述版本。描述每換一次就加 1 |
| `bbox` | `[x, y, w, h]` int | **上傳影像的座標**。只有 `FOUND` 時有效 |
| `score` | float | VLM 或分割結果的信心值，0～1。只有 `FOUND` 時有效 |
| `server_ms` | int | server 從收到 request 到送出 response 花了多少毫秒 |
| `error` | string | `ERROR` 時放錯誤訊息，其他狀態是空字串 |

第二個 frame（只有 `FOUND` 時有）：mask PNG
- 單通道 8-bit，大小正好是 `w × h`（bbox 範圍，不是整張圖）。
- 物件是 `255`，背景是 `0`。
- 有多個候選物件時，只回傳分數最高的那一個。

| `status` | 意義 | client 的處理 |
|---|---|---|
| `FOUND` | 找到目標 | 建立候選 target |
| `NOT_FOUND` | 畫面中沒有目標 | 保持目前狀態，依送圖時機再送 |
| `NO_QUERY` | server 還沒收到上級描述 | 暫停送圖，等一段時間再試 |
| `ERROR` | server 端出錯 | 記錄 `error`，依送圖時機再送 |

### 4.3 Ping：量測網路延遲

`type: "ping"` 只有 header，沒有影像。server 立刻回傳：

```json
{"protocol_version": 1, "request_id": 43, "status": "PONG", "query_version": 3, "server_ms": 0, "error": ""}
```

用途有兩個：一是量 WiFi 的來回延遲，不包含 VLM 推論時間；二是在沒有追蹤需求時，也能提早發現 `query_version` 改變。

### 4.4 座標換算（client 端）

```
原始座標 = 上傳座標 × (原始寬度 / 上傳寬度)
```

server 不需要知道相機的原始解析度。

## 5. 時間差處理（client 端）

| 情況 | 處理 |
|---|---|
| 送圖 | 那一幀的灰階、深度，以及 camera→map TF 存進快取（最多 `vlm.cache_size` 幀）。TF 在送圖當下就存，因為 tf2 buffer 預設只保留 10 秒 |
| 等待中 | 繼續用舊 target 追蹤，不阻塞影像 callback（網路收發在背景 thread 處理） |
| 逾時（超過 `vlm.timeout_s`） | 放棄這一筆，允許送下一張。之後如果舊的回應才到，`request_id` 對不上就丟棄 |
| 收到 `FOUND` | 用快取裡那一幀加上 mask 建新 target：ORB 只在 mask（內縮幾 px）內抽特徵，發布點改用 mask 質心 |
| 新 target 驗證 | 先在「現在這一幀」偵測，成功才替換；失敗就保留舊 target，再試 N 幀 |
| `query_version` 改變 | 描述換了，舊 target 立刻作廢，馬上送圖 |

### server 端的注意事項

client 逾時後可能會送出新的 request，server 這時可能還在處理舊的那筆。server 處理完一筆後，應該先把 socket 裡排隊的 request 全部讀出來，**只處理最新的一筆**，其餘直接丟掉，不需要回覆。

## 6. 送圖時機（client 端）

符合任一條件、而且目前沒有請求在路上，就送最新的一幀：

1. 還沒有 target。
2. 連續 `vlm.lost_frames_trigger` 幀追丟。
3. 追蹤中，距離上次送圖超過 `vlm.refresh_period_s`。
4. 上一筆逾時。
5. `query_version` 改變。

之後可以再加的優化：在最近幾幀中挑最清楚的一幀送（例如 Laplacian variance 最高的，或相機靜止時的那一幀）。

## 7. Client 參數（暫定）

| 參數 | 預設 | 說明 |
|---|---|---|
| `vlm.enable` | `false` | 關閉時沿用 `target_image_path` 的靜態 target |
| `vlm.endpoint` | `tcp://192.168.0.100:5555` | server 位址，IP 待定 |
| `vlm.timeout_s` | `5.0` | 等待回應的上限 |
| `vlm.refresh_period_s` | `3.0` | 追蹤中定期送圖的間隔 |
| `vlm.lost_frames_trigger` | `10` | 連續追丟幾幀就送圖 |
| `vlm.upload_max_width` | `640` | 上傳前縮圖的最大寬度，應配合 VLM 的輸入尺寸 |
| `vlm.jpeg_quality` | `80` | JPEG 壓縮品質 |
| `vlm.cache_size` | `4` | 本地快取的幀數 |

## 8. 延遲量測

client 每筆請求記錄以下數值：

```
encode_ms   = JPEG 壓縮時間
rtt_ms      = 收到回應時間 − 送出時間
server_ms   = response 帶回的值
network_ms  = rtt_ms − server_ms
handoff_ms  = 建 target + 在目前這一幀驗證的時間
```

目前還沒有實測數據。`vlm.timeout_s` 和 `vlm.refresh_period_s` 要等量到 `rtt_ms` 的分布後再決定。

## 9. 開發順序建議

1. **mock server**（Python + pyzmq）：照這份協定回傳固定的 bbox 和 mask，或開視窗用滑鼠框選。可以加人工延遲和隨機不回應，模擬 WiFi。這樣 client 不必等真的 VLM 就能開發。
2. **client 網路 thread + 快取 + 逾時**，先用 `ping` 驗證連線。
3. **handoff**：用 mask 建 target，驗證後替換。
4. 接上真的 server，實測延遲後調整參數。

## 10. 相依套件與待確認事項

- client 的 Dockerfile 要加 `libzmq3-dev`。C++ header 版的 cppzmq 在 Ubuntu 22.04 的 apt 套件名稱還沒查證，找不到的話直接用 libzmq 的 C API。
- server 需要 `pyzmq`（假設 server 用 Python）。
- 待和 server 端約定：server IP、port、VLM 的輸入尺寸、`query_version` 的產生方式。
