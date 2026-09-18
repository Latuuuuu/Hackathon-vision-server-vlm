# TODO

1. [Client 交接策略：現有作法 vs 預追蹤](#1-client-交接策略現有作法-vs-預追蹤)（進行中，client 還沒實作）
2. [把 VLM 搬到雲端 AMD MI300X](#2-把-vlm-搬到雲端-amd-mi300x)（暫緩）

## 1. Client 交接策略：現有作法 vs 預追蹤

> 狀態：**待決定**。protocol 不受影響，兩種作法 server 都只回 bbox（mask 選配）。

### 要解決的問題

server 回傳時，bbox 描述的是 **約 3.4 秒前**送出的那一幀
（`LA_MODE=fast` WiFi 實測 rtt p50 3.44 s、max 3.71 s，以 30 fps 計約 100 幀前）。
這段時間機器人和物體都可能在動，client 必須把「舊畫面上的 bbox」轉成「現在畫面上的 target」。

### 已排除：server 端抽特徵往下傳（方案 A）

曾考慮 server 在 bbox/mask 內抽特徵傳給下游，讓下游直接做 KLT、不存舊圖。**已丟棄**，理由：

- KLT 是逐幀追蹤，需要前一幀的影像 patch，而且只能處理數十 px 的位移。
  座標屬於約 100 幀前的畫面，直接跳到現在會追丟；逐幀推又得存 100 幀，比現在更多。
- 改傳 ORB 描述子雖然可行，但特徵是從 q80 縮圖 JPEG 抽的，比對率會比 client 原始畫面差；
  server 還得跟 tracker 的 ORB 參數與 OpenCV 版本綁死。換來的只是省下約 1.2 MB 灰階快取。

### 兩種候選作法

**現有作法：client 快取 + 當前幀驗證**（[vlm_transport.md](vlm_transport.md) §5）

1. 送圖時把該幀的灰階、深度、camera→map TF 存進快取（`vlm.cache_size = 4`）。
2. bbox 回來：從快取取出同一幀，在 bbox 內用深度切出物件、抽 ORB 建新 target。
3. 在「現在這一幀」偵測新 target，成功才替換舊 target；失敗就保留舊的，再試 N 幀。
4. 可選強化：用 t0 的深度 + TF 把目標投影到現在的畫面，當作搜尋起點（只對靜態物件有效）。

**方案 B：client 預追蹤**

1. 送圖的同時，在 t0 整張畫面撒一批 KLT 點（例如 `goodFeaturesToTrack` 或規則網格）。
2. 等待期間逐幀用 `calcOpticalFlowPyrLK` 追這些點，建議加 forward-backward 檢查剔除壞點。
3. bbox 回來：挑出「起點落在 bbox 內、而且還活著」的點。這些點**已經在現在這一幀的位置上**。
4. 用這批點在 t0 與現在之間的位移估出變換（similarity / homography），把 bbox 搬到現在的畫面，
   再用**現在這一幀的深度**切出物件，直接開始追蹤。

> 更正先前的分析：B 不需要保留 t0 的深度和 TF。點已經在當前幀，深度切割和 3D 發布都用當前幀的資料即可。

### 比較

| 面向 | 現有作法 | 方案 B：預追蹤 |
|---|---|---|
| 快取 | 4 幀灰階 + 深度 + TF，約 3.7 MB | 不需要影像快取，只存點軌跡 |
| 應付 3.4 秒位移 | 靠外觀在當前幀重新偵測；視角、尺度變化大時可能失敗 | **最好**：點是一路追過來的，不怕大位移 |
| 移動中的物體 | 外觀比對可處理；TF 投影強化只對靜態物件有效 | 可處理 |
| Pi 5 運算 | 只在交接時抽一次 ORB、比對一次 | **等待期間每幀都要跑 KLT**（約 3.4 秒 × 30 fps），負擔最重，**未實測** |
| 失敗模式 | 驗證失敗 → 保留舊 target，再試 | 目標上的點全數追丟（遮擋、快速轉動、模糊）→ 這筆請求作廢，只能重送，再等約 3.4 秒 |
| 實作複雜度 | 較低，流程直觀 | 較高：要管理點的生命週期、去除壞點、估計變換 |
| server 端 | 不用改 | 不用改 |

### 怎麼選：先做現有作法，用數據決定要不要加 B

**建議順序：先實作現有作法，量數據，再決定。** 理由：

- 現有作法比較簡單，也是驗證整條 pipeline 的最短路徑。
- B 追丟時沒有後備，實務上很可能還是需要現有作法當 fallback。先做現有作法不會白做。
- protocol 不用改，之後再加 B 只動 client。

實作現有作法時，順手記錄這三個數據：

| 要量的 | 怎麼量 | 看到什麼就該考慮 B |
|---|---|---|
| **交接驗證失敗率** | 統計第 3 步「在現在這一幀偵測新 target」的成功與失敗次數 | 失敗率明顯偏高，而且失敗多發生在機器人或物體移動時 |
| **延遲期間的位移** | 用 TF 算送圖到收到回應之間相機平移與旋轉量；或用 bbox 中心在影像上的位移 | 常常超過 bbox 寬度的一半，或相機轉角大到物體快出畫面 |
| **Pi 5 CPU 餘裕** | 在 Pi 5 上用實際解析度量 `calcOpticalFlowPyrLK`（例如 200 / 500 點）每幀耗時，加上現有 tracker 的負載 | 這是 B 的前提：每幀 KLT 加上 tracker 要能在 33 ms 內跑完（30 fps），否則 B 不可行或要降點數、降幀率 |

判斷規則：

- **失敗率可接受** → 維持現有作法，B 不做。
- **失敗率高、主因是位移，而且 Pi 5 有 CPU 餘裕** → 加 B，現有作法保留當 fallback。
- **失敗率高、但 Pi 5 沒有餘裕** → 先試「用 t0 深度 + TF 投影當搜尋起點」（成本低，限靜態物件），
  或降低機器人在等待期間的移動（例如送圖時先減速）。
- **失敗主因是外觀**（光線、角度、遮擋），不是位移 → B 也救不了，改善 tracker 的特徵或驗證條件。

### 待辦

- [ ] 實作現有作法（client：網路 thread + 快取 + 逾時 + 交接驗證）
- [ ] 加上三項量測的 log
- [ ] 在 Pi 5 上量 `calcOpticalFlowPyrLK` 的每幀耗時
- [ ] 依數據決定是否加 B

## 2. 把 VLM 搬到雲端 AMD MI300X

> 狀態：**暫緩**。2026-09-18 決定先專注本地路線（Hackathon-gpu，Radeon 860M）。
> 這一節記錄已經查到的事實和要做的順序，之後要撿回來時不用重新調查。

### 為什麼考慮雲端

本地 server 在 Radeon 860M 上，單/雙目標、`LA_MODE=fast` 的推論時間約 **3.4 秒**（實測，見 [vlm_transport.md](vlm_transport.md) §8）。
對追蹤來說偏慢，client 的 `vlm.refresh_period_s` 只能設得比這個更長。

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
| 上傳 JPEG | 44 KB（單目標）～ 87 KB（15 隻狗的複雜畫面），實測 |
| 回傳 | bbox JSON 約 200 bytes，可忽略 |
| Pi 上傳 87 KB | uplink 5 Mbps ≈ 140 ms；20 Mbps ≈ 35 ms |
| 到區域機房 RTT | 10～40 ms（跨區 150～250 ms） |
| **網路總成本估計** | **50～300 ms** |

結論：只要雲端推論能壓到 1 秒內，多付這些網路時間仍然大贏本地的 3.4 秒。

協定不需要改：`request_id` 對應、只處理最新一筆、逾時重送、TCP 自動重連都已經實作。
client 換 server 只是改 `vlm.endpoint`。

### 運算面：四條路，依序試

模型：NVIDIA **LocateAnything-3B**（Qwen2.5-3B LM + MoonViT 視覺塔），gguf q8_0，5.83 GiB。

**關鍵限制**：locate-anything.cpp v0.1.0 的 CMake 只有
`LA_GGML_CUDA` / `LA_GGML_METAL` / `LA_GGML_VULKAN` / `LA_GGML_CANN`，
**沒有 HIP/ROCm 選項**（已查 v0.1.0 的 CMakeLists.txt 與 README）。

| 路線 | 做法 | 評估 |
|---|---|---|
| **CPU 路線（先做）** | 不設 `LA_DEVICE`，用 CPU backend | 零 backend 工作量，最快拿到基準線。MI300X 機器通常配 EPYC，記憶體頻寬遠高於筆電，有機會直接贏過 3.4 秒 |
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
      比對本地的 3.4 秒。

- [ ] **4. 依步驟 2 的結果試 Vulkan 或 HIP**，量同一組數字。

- [ ] **5. 決定是否切換**，並補上 VPN 與 HTTP API 的認證。

### 待確認

- 雲端 server 是常開還是按需啟動？
- 有公開 IP 還是要走跳板 / VPN？
- Pi 5 實際的上傳頻寬（會場 WiFi vs 有線）。
- MI300X 的 ROCm 版本與是否有 Mesa/Vulkan。
