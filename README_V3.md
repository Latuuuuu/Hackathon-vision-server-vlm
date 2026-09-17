# V3：AMD GPU 低延遲追蹤

本版暫停 NPU；只用 Locate Vulkan GPU＋SAM ROCm GPU。保留 ROCm 10 gfx1152 已通過的套件組合。主目標是降低畫面延遲、縮短真實 SAM mask 推論時間，而不是用舊遮罩貼到新畫面。

## 根據你貼的日誌

- 相機約 29.97 FPS，SAM 單步約 1886 ms；速度差在模型端，不是相機缺影格。
- SAM 從約 1.01 秒逐步增加到約 1.86 秒，符合記憶填滿後 attention 工作量增加的可能情形。這是根據程式結構的推論，V3 加入分段計時確認，不能只靠總時間定論。
- V2 會先完成 Locate、SAM 初始化，再依序重播最多 8 張歷史影格。依你每步約 1–2 秒的紀錄，這段本身就可能再花很多秒，使輸出一直落後。V3 完全移除歷史重播。
- 你提供的 JSON 已是 idle／output_captured=null／tracking_fps=0，其他時間是殘留值；它不是活動中 17 秒延遲的快照。V3 在停止／切換／重連時清除相關欄位。
- V2 的 completion_age 約 1893 ms 接近 sam_step 約1887 ms，該筆已完成結果沒有呈現17秒的推論等待；17秒可能來自先前追趕、畫面傳輸／瀏覽器緩衝或不同時間點。V3 同時顯示「伺服器完成時的年齡」與「瀏覽器目前顯示影格的年齡」，不混為同一個指標。

## 修改內容

1. 初始化保留官方 init_state 已算好的特徵，移除原本清快取導致的第二次相同影像編碼。
2. 使用 Locate 的原始影格建立目標記憶；不把這張歷史初始 mask 當直播輸出。初始化完成後直接處理最新相機影格，不排8張舊圖。
3. 瀏覽器從 MJPEG 串流改為最新影格拉取。每個視窗最多一個未完成請求，伺服器回傳最新完成的 JPEG；帶影格序號／generation／年齡，網路慢時不積累整串畫面。保留舊 /video 路由供相容，但新版網頁不使用它。
4. 改變 SAM 真正的內部輸入尺寸：預設512，並調整模型的 prompt embedding 尺寸；不是只把相機影像縮小再放回1024。SAM 官方 RoPE 原始碼會依查詢特徵尺寸重建位置編碼。本專案的低解析度設定仍需要硬體 shape smoke test 與遮罩品質驗證。
5. 預設 attention memory 從官方7減到3、object pointers 上限從16減到4。先依原尺寸載入 checkpoint，再保留最近一段 learned temporal embeddings；模型仍使用真正影片記憶，但較短視窗可能降低遮擋後恢復能力。
6. 分段計時：image encoder、memory attention、mask decoder、memory encoder。GPU 使用 device events，避免只量到 Python 排程時間；CPU 使用 wall time。它们不包括所有前後處理，總步耗時仍獨立記錄。
7. 增加24張、排除前8張暖機的 benchmark，避免之前只跑2張傳播，誤判長時間效能。

## 啟動步驟

ZIP 根目錄仍為 locate-sam2-d435i-v1，內容已為V3。將其中程式覆蓋進你目前的 locate-sam2-d435i-v2 資料夾即可；保留 .env、models、output。不要載入 compose.npu.yaml。

```bash
docker compose -f compose.yaml -f compose.rocm.yaml down
docker compose -f compose.yaml -f compose.rocm.yaml build tracker
docker compose -f compose.yaml -f compose.rocm.yaml run --rm tracker python -m scripts.smoke_sam
docker compose -f compose.yaml -f compose.rocm.yaml run --rm tracker python -m scripts.benchmark_sam --output /output/benchmark-512.json
docker compose -f compose.yaml -f compose.rocm.yaml up -d tracker
```

smoke_sam 或 benchmark 失敗就先處理錯誤，不要把結果當成完成加速。image tag 已改成 locate-sam2:rocm10-v3，確保不是舊映像。

網頁 http://localhost:8080，請重新整理（必要時 Ctrl+Shift+R）。先保持物體静止完成定位，看到第一張 live mask 後再移動。直接跳最新影格減少延遲，但Locate期間物體快速移動／離開視野，仍可能導致初始化身分追蹤失敗。

## 速度／品質設定

在 .env 加入，重建容器即可；不用重新下載模型或重新 build：

```dotenv
SAM_IMAGE_SIZE=512
SAM_MEMORY_FRAMES=3
SAM_OBJECT_POINTERS=4
SAM_PROFILE=1
```

```bash
docker compose -f compose.yaml -f compose.rocm.yaml up -d --force-recreate tracker
```

|設定|尺寸|memory|pointers|用途|
|---|---:|---:|---:|---|
|預設|512|3|4|先測低延遲與可用精度|
|較快實驗|384|3|4|512仍慢時；小物體／邊界細節可能更差|
|較高品質|768|5|8|速度夠但遮罩不穩時|
|原始容量比較|1024|7|16|比較原模型計算量；仍保有V3佇列與快取修正|

上表為本專案工程選項，不是官方效能／精度保證。尺寸變小及縮短記憶會改變輸出，需目視／標註比較。沒有保證達到10、15或30FPS；先看 benchmark 與實機數據。

只針對一次 benchmark 比較384，不改 .env：

```bash
docker compose -f compose.yaml -f compose.rocm.yaml run --rm -e SAM_IMAGE_SIZE=384 tracker python -m scripts.benchmark_sam --output /output/benchmark-384.json
```

## 可選：實驗 attention kernel

你目前的 PyTorch 明確提示 gfx1152 的 Flash／Memory Efficient attention 尚屬實驗。V3預設不開，另附獨立覆寫檔做A/B測試，不默默啟用。

```bash
docker compose -f compose.yaml -f compose.rocm.yaml stop tracker
docker compose -f compose.yaml -f compose.rocm.yaml -f compose.attention.yaml run --rm tracker python -m scripts.gpu_probe
docker compose -f compose.yaml -f compose.rocm.yaml -f compose.attention.yaml run --rm tracker python -m scripts.benchmark_sam --output /output/benchmark-512-experimental.json
```

只在通過並比較更快、遮罩無明顯退化後，使用三份 compose 啟動服務。設環境變數只是允許選擇實驗後端，不保證所有 attention 都會用到 Flash kernel。若出現崩潰／NaN，回到只帶compose.yaml與compose.rocm.yaml的命令即可，不需要改主機驅動。

## 讀取結果

- 網頁顯示年齡來自**實際顯示的那張JPEG**，加上客戶端經過時間。傳輸計時採近似保守值，不是相機曝光至螢幕的硬體精準延遲。
- stage_ms.memory_attention_ms 隨暖機增長可支持記憶計算瓶頸；若 image_encoder_ms 佔大多數，輸入尺寸更重要；若總步時間遠大於這些GPU時間，查前處理／CPU排程／傳输與共享記憶體競爭。
- 畫面年齡自然會在兩次模型輸出之間增加。即使不排隊，每張花1秒，仍不可能顯示每張都只有幾十毫秒的真實SAM結果。
- V3 CSV 用 output/tracking-v3.csv，避免與舊格式混用。benchmark JSON 記錄設定、每張時間與穩態P50／P95／FPS；是合成圖的模型測試，不包含Locate、相機與网页。
- 可交回 benchmark-512.json、tracking-v3.csv 以及**tracking活動中**的網頁JSON（包含browser_display與stage_ms），即可更精確定位下一個瓶頸。

目前本環境僅能做靜態／單元測試，沒有你的GPU或模型，尚未實測新版512／384設定、實驗attention相容性或速度提升。原始模型的成功不等於新版低解析度品質已驗證。

參考：[SAM-2模型與記憶原始碼](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py)、[RoPE與SDPA原始碼](https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam/transformer.py)。延遲分析與時間數據取自你本次日誌；改善幅度待實測。
