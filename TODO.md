# TODO

1. [Client 交接策略：現有作法 vs 預追蹤](#1-client-交接策略現有作法-vs-預追蹤)（現有作法已上線，B 待數據決定）
2. [把 VLM 搬到雲端 AMD MI300X](#2-把-vlm-搬到雲端-amd-mi300x)（暫緩）

## 1. Client 交接策略：現有作法 vs 預追蹤

> 狀態：**現有作法已由 client 實作，並和真 server 聯測通過（2026-09-19）**。方案 B 待數據決定。
> 完整分析（含已丟棄的方案 A、比較表、判斷規則）已整理到 [server_progress.md](server_progress.md) 第 6 節，這裡只追蹤進度。

### 摘要

- 問題：bbox 回來時描述的是約 2 秒前（約 60 幀前）的畫面（聯測 rtt p50 1988 ms）。
- **現有作法**：client 快取送出的幀，bbox 回來後用同一幀建 target，在現在這一幀驗證後替換。
- **方案 B（預追蹤）**：送圖時撒 KLT 點、等待期間逐幀追，bbox 回來時直接取已在現在這一幀的點。
  最能應付位移，但等待期間每幀都要跑 KLT；client 改成連續送圖（`refresh_period_s = 0`）後等於一直在跑。
- **方案 A（server 抽特徵往下傳）**：已丟棄。
- **決策**：先用現有作法量三項數據（交接驗證失敗率、延遲期間位移、Pi 5 CPU 餘裕），依 server_progress.md 第 6 節的規則決定要不要加 B。

### 目前線索

- tracker debug 畫面：`TRACKING OK KLT`，每幀 6.8 ms、追 134 點，離 33 ms（30 fps）的預算還有空間。
  但 B 要在整張畫面追更多點，bridge 和相機驅動也在用 CPU，**還不能推論 B 可行**。

### 待辦

- [x] 實作現有作法（client：網路 thread + 快取 + 逾時 + 交接驗證）
- [x] 和真 server 聯測（DEBUG.md L0～L3 通過）
- [ ] client 參數改成 server_progress.md 第 5 節的建議（`refresh_period_s = 0.0` 等）
- [ ] 確認 tracker 用真 server 的 bbox 建 target 成功（handoff log）
- [ ] client 加上三項量測的 log
- [ ] 在 Pi 5 上量 `calcOpticalFlowPyrLK`（200／500 點）的每幀耗時
- [ ] 依數據決定是否加 B

## 2. 把 VLM 搬到雲端 AMD MI300X

> 狀態：**暫緩**。2026-09-18 決定先專注本地路線（Hackathon-gpu，Radeon 860M）。
> 這一節記錄已經查到的事實和要做的順序，之後要撿回來時不用重新調查。

### 為什麼考慮雲端

本地 server 在 Radeon 860M 上，`LA_MODE=fast`、Pi 相機畫面（640×362）的推論時間約 **1.9 秒**，
端到端 rtt 約 2.0 秒（2026-09-19 聯測，見 [vlm_transport.md](vlm_transport.md) §8）。
client 連續送圖時，target 約每 2 秒更新一次；推論時間大致和上傳像素數成正比。

MI300X 有加速的物理基礎：LocateAnything-3B 的解碼階段是**記憶體頻寬受限**，
860M 共用 LPDDR5X 約 120 GB/s，MI300X 是 HBM3 5.3 TB/s，理論差約 40 倍。
但 ggml 在 CDNA3 上的 kernel 成熟度達不到理論值，**實際加速倍率沒有量過，不要當成已知**。

### 前提條件（已確認）

- 雲端 server 可以自己裝 docker、自行編譯，價格不是考量。
- client 是 Raspberry Pi 5 Model B 8GB。
- Pi 5 **不可能自己跑 VLM**：模型權重 5.83 GiB，Pi 5 的 LPDDR4X 頻寬約 17 GB/s，
  解碼每個 token 都要讀一遍語言模型權重，一次推論會是數十秒級。VLM 一定在 server 端。

### 網路面：已分析，成本可接受

用實測的 JPEG 大小算（`tools/test_client.py` 的預設 640 寬、q80）：

| 項目 | 數值 |
|---|---|
| 上傳 JPEG | Pi 相機 640×362 約 27 KB；測試圖 44 KB（單目標）～ 87 KB（15 隻狗的複雜畫面），實測 |
| 回傳 | bbox JSON 約 200 bytes，可忽略 |
| Pi 上傳 87 KB | uplink 5 Mbps ≈ 140 ms；20 Mbps ≈ 35 ms |
| 到區域機房 RTT | 10～40 ms（跨區 150～250 ms） |
| **網路總成本估計** | **50～300 ms** |

結論：只要雲端推論能壓到 1 秒內，多付這些網路時間仍然比本地的 1.9 秒快。
當初分析用的 3.4 秒是 640×656 的測試圖；實際 Pi 畫面（640×362）只要 1.9 秒，所以搬到雲端的效益比當初估計的小。
（聯測時機器人上 WiFi 的 `network_ms` p50 為 103 ms，走外網只會更高。）

協定不需要改：`request_id` 對應、只處理最新一筆、逾時重送、TCP 自動重連都已經實作。
client 換 server 只是改 `vlm.endpoint`。

### 運算面：四條路，依序試

模型：NVIDIA **LocateAnything-3B**（Qwen2.5-3B LM + MoonViT 視覺塔），gguf q8_0，5.83 GiB。

**關鍵限制**：locate-anything.cpp v0.1.0 的 CMake 只有
`LA_GGML_CUDA` / `LA_GGML_METAL` / `LA_GGML_VULKAN` / `LA_GGML_CANN`，
**沒有 HIP/ROCm 選項**（已查 v0.1.0 的 CMakeLists.txt 與 README）。

| 路線 | 做法 | 評估 |
|---|---|---|
| **CPU 路線（先做）** | 不設 `LA_DEVICE`，用 CPU backend | 零 backend 工作量，最快拿到基準線。MI300X 機器通常配 EPYC，記憶體頻寬遠高於筆電，有機會贏過本地的 1.9 秒 |
| **Vulkan 路線** | 沿用現有 Dockerfile（`-DLA_GGML_VULKAN=ON`） | MI300X 是 gfx942 純運算卡，雲端映像通常只有 ROCm 沒有 Mesa。**能不能跑要上機驗證** |
| **HIP 路線** | ggml 上游有 `GGML_HIP`。LA 沒包裝這個選項，但它把 ggml 當子專案，可直接傳 `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx942` | 中等工作量，可能要改 CMake。Vulkan 不行就走這條 |
| **vLLM 路線（換模型）** | 改用 Qwen2.5-VL 這類原生支援 grounding 的模型，跑 vLLM ROCm | ROCm 支援最成熟，但等於換掉整個 VLM，框的品質要全部重新驗證。最後手段 |

**雲端 image 可以不裝 torch**：[vlm_server/pipeline.py](vlm_server/pipeline.py) 只在
`VLM_RETURN_MASK=1` 時才 import torch（`MaskPredictor.__init__` 內），
預設只回 bbox 的模式完全不需要。目前 image 9.6 GB 大部分是 ROCm torch，
雲端版可以砍到 1～2 GB，建置也簡單很多。

### 安全與維運（上公網前必做）

- **目前完全沒有認證**：ZMQ 是明文 TCP，HTTP `/api/query` 任何人打得到就能改追蹤目標。
  **絕對不要直接把 5555 / 8080 開到公網。**
- 建議 WireGuard 或 Tailscale：一次解決加密、認證、NAT 穿透，程式不用改。
- 5.83 GiB 模型要先上傳，或在雲端直接從 HuggingFace 下載。
- 冷啟動成本：容器啟動 + 模型載入約 3～5 秒（本地實測 `load_s` 約 2～3 秒）。
- **保留本地 fallback**：Demo 當天依賴會場網路有風險，切換只是改 `vlm.endpoint`。

### 執行順序

- [ ] **1. 先量網路（零風險，不需要 GPU 或模型）**
      雲端跑 mock server，從 Pi 量 RTT 和上傳時間，確認網路那一半可行：
      ```bash
      # 雲端（只需要 python + pyzmq + flask，不需要模型）
      python -m tools.mock_server --delay 0.5 --query "the cup"
      # Pi
      python -m tools.test_client --endpoint tcp://<cloud>:5555 --ping 20
      python -m tools.test_client --endpoint tcp://<cloud>:5555 --image frame.jpg --count 10
      ```
      要看的是 `rtt_ms − server_ms`（純網路）的 p50 和 p95。

- [ ] **2. 上機確認 backend**（決定走 Vulkan 還是 HIP）
      ```bash
      rocm-smi; ls -l /dev/kfd /dev/dri; vulkaninfo --summary 2>&1 | head -30; glxinfo -B 2>/dev/null
      ```

- [ ] **3. 建輕量 image 並量 CPU 基準（CPU 路線）**
      複製 Dockerfile，移除 torch / torchvision / sam2 / pyrealsense2 相關層，
      跑 `python -m scripts.smoke_pipeline <image> --target "..." --repeat 5`，
      用同樣 640×362 的圖比對本地的 1.9 秒（推論時間和像素數相關，尺寸要一致才能比）。

- [ ] **4. 依步驟 2 的結果試 Vulkan 或 HIP**，量同一組數字。

- [ ] **5. 決定是否切換**，並補上 VPN 與 HTTP API 的認證。

### 待確認

- 雲端 server 是常開還是按需啟動？
- 有公開 IP 還是要走跳板 / VPN？
- Pi 5 實際的上傳頻寬（會場 WiFi vs 有線）。
- MI300X 的 ROCm 版本與是否有 Mesa/Vulkan。
