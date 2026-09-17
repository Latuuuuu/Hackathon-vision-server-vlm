# TODO：把 VLM 搬到雲端 AMD MI300X

> 狀態：**暫緩**。2026-09-18 決定先專注本地路線（Hackathon-gpu，Radeon 860M）。
> 這份文件記錄已經查到的事實和要做的順序，之後要撿回來時不用重新調查。

## 為什麼考慮雲端

本地 server 在 Radeon 860M 上，單/雙目標、`LA_MODE=fast` 的推論時間約 **3.4 秒**（實測，見 [vlm_transport.md](vlm_transport.md) §8）。
對追蹤來說偏慢，client 的 `vlm.refresh_period_s` 只能設得比這個更長。

MI300X 有加速的物理基礎：LocateAnything-3B 的解碼階段是**記憶體頻寬受限**，
860M 共用 LPDDR5X 約 120 GB/s，MI300X 是 HBM3 5.3 TB/s，理論差約 40 倍。
但 ggml 在 CDNA3 上的 kernel 成熟度達不到理論值，**實際加速倍率沒有量過，不要當成已知**。

## 前提條件（已確認）

- 雲端 server 可以自己裝 docker、自行編譯，價格不是考量。
- client 是 Raspberry Pi 5 Model B 8GB。
- Pi 5 **不可能自己跑 VLM**：模型權重 5.83 GiB，Pi 5 的 LPDDR4X 頻寬約 17 GB/s，
  解碼每個 token 都要讀一遍語言模型權重，一次推論會是數十秒級。VLM 一定在 server 端。

## 網路面：已分析，成本可接受

用實測的 JPEG 大小算（`tools/test_client.py` 的預設 640 寬、q80）：

| 項目 | 數值 |
|---|---|
| 上傳 JPEG | 44 KB（單目標）～ 87 KB（15 隻狗的複雜畫面），實測 |
| 回傳 | bbox JSON 約 200 bytes，可忽略 |
| Pi 上傳 87 KB | uplink 5 Mbps ≈ 140 ms；20 Mbps ≈ 35 ms |
| 到區域機房 RTT | 10～40 ms（跨區 150～250 ms） |
| **網路總成本估計** | **50～300 ms** |

結論：只要雲端推論能壓到 1 秒內，多付這些網路時間仍然大贏本地的 3.4 秒。

協定不需要改：`request_id` 對應、只處理最新一筆、逾時重送、TCP 自動重連都已經實作。
client 換 server 只是改 `vlm.endpoint`。

## 運算面：四條路，依序試

模型：NVIDIA **LocateAnything-3B**（Qwen2.5-3B LM + MoonViT 視覺塔），gguf q8_0，5.83 GiB。

**關鍵限制**：locate-anything.cpp v0.1.0 的 CMake 只有
`LA_GGML_CUDA` / `LA_GGML_METAL` / `LA_GGML_VULKAN` / `LA_GGML_CANN`，
**沒有 HIP/ROCm 選項**（已查 v0.1.0 的 CMakeLists.txt 與 README）。

| 路線 | 做法 | 評估 |
|---|---|---|
| **D. 純 CPU（先做）** | 不設 `LA_DEVICE`，用 CPU backend | 零 backend 工作量，最快拿到基準線。MI300X 機器通常配 EPYC，記憶體頻寬遠高於筆電，有機會直接贏過 3.4 秒 |
| **A. Vulkan** | 沿用現有 Dockerfile（`-DLA_GGML_VULKAN=ON`） | MI300X 是 gfx942 純運算卡，雲端映像通常只有 ROCm 沒有 Mesa。**能不能跑要上機驗證** |
| **B. 改建 HIP** | ggml 上游有 `GGML_HIP`。LA 沒包裝這個選項，但它把 ggml 當子專案，可直接傳 `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx942` | 中等工作量，可能要改 CMake。A 不行就走這條 |
| **C. 換模型 + vLLM** | 改用 Qwen2.5-VL 這類原生支援 grounding 的模型，跑 vLLM ROCm | ROCm 支援最成熟，但等於換掉整個 VLM，框的品質要全部重新驗證。最後手段 |

**雲端 image 可以不裝 torch**：[vlm_server/pipeline.py](vlm_server/pipeline.py) 只在
`VLM_RETURN_MASK=1` 時才 import torch（`MaskPredictor.__init__` 內），
預設只回 bbox 的模式完全不需要。目前 image 9.6 GB 大部分是 ROCm torch，
雲端版可以砍到 1～2 GB，建置也簡單很多。

## 安全與維運（上公網前必做）

- **目前完全沒有認證**：ZMQ 是明文 TCP，HTTP `/api/query` 任何人打得到就能改追蹤目標。
  **絕對不要直接把 5555 / 8080 開到公網。**
- 建議 WireGuard 或 Tailscale：一次解決加密、認證、NAT 穿透，程式不用改。
- 5.83 GiB 模型要先上傳，或在雲端直接從 HuggingFace 下載。
- 冷啟動成本：容器啟動 + 模型載入約 3～5 秒（本地實測 `load_s` 約 2～3 秒）。
- **保留本地 fallback**：Demo 當天依賴會場網路有風險，切換只是改 `vlm.endpoint`。

## 執行順序

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

- [ ] **2. 上機確認 backend**（決定走 A 還是 B）
      ```bash
      rocm-smi; ls -l /dev/kfd /dev/dri; vulkaninfo --summary 2>&1 | head -30; glxinfo -B 2>/dev/null
      ```

- [ ] **3. 建輕量 image 並量 CPU 基準（路線 D）**
      複製 Dockerfile，移除 torch / torchvision / sam2 / pyrealsense2 相關層，
      跑 `python -m scripts.smoke_pipeline <image> --target "..." --repeat 5`，
      比對本地的 3.4 秒。

- [ ] **4. 依步驟 2 的結果試 A 或 B**，量同一組數字。

- [ ] **5. 決定是否切換**，並補上 VPN 與 HTTP API 的認證。

## 待確認

- 雲端 server 是常開還是按需啟動？
- 有公開 IP 還是要走跳板 / VPN？
- Pi 5 實際的上傳頻寬（會場 WiFi vs 有線）。
- MI300X 的 ROCm 版本與是否有 Mesa/Vulkan。
