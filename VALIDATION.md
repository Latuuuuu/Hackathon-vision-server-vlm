# 驗證紀錄

## V3 更新

12 項核心單元測試通過，包含停止後統計重設、最新影格／相機epoch篩選及速度設定邊界。Python AST、Compose YAML 結構、viewer.js JavaScript 語法檢查通過。未在本環境執行 Docker build、完整瀏覽器端端到端串流、512／384 SAM 真實模型推論或 ROCm 效能測量。此版速度與品質需要目標硬體驗證。

## V2 更新

新增 GPU 子程序錯誤傳遞、缺少 PASS 拒絕、成功結果解析、NPU／CPU 報告區分、encoder shape／dtype／有限值檢查，共 9 個核心單元測試通過。所有 Python AST、Compose YAML 結構與 NPU entrypoint shell 語法檢查通過。

使用者已提供 V1 CPU 真實 SAM smoke test PASS；V2 GPU 使用官方文件列出的 ROCm 10 gfx1152 wheel，NPU 使用官方文件的 Vitis AI API。尚未實際下載／建置這些套件、匯出模型、進行 NPU 編譯、測試混合追蹤或測量新版本 FPS。NPU 原型不是已完成硬體驗證的功能。

## V1 紀錄

已在交付環境執行：

- 所有 Python 檔案 AST／compileall 語法檢查通過。
- compose.yaml、compose.rocm.yaml 的 YAML 結構檢查通過；不是 Docker Compose 實際解析／建置測試。
- 瀏覽器 JavaScript 通過 Node `--check`；尚未以相機串流做瀏覽器端端到端測試。
- `unittest discover -s tests -v`：4／4 通過，涵蓋只改變遮罩內像素、重連時歷史隔離、記憶清理保留 conditioning anchor、非法 bbox 拒絕。

尚未執行：Docker build、RealSense 開流、Locate C ABI 實際呼叫、SAM 真實權重初始化／影片傳播、ROCm 完整 kernel、長時間記憶體及追蹤品質測試。交付環境沒有 Docker、D435i 或目標 AMD 硬體。以上項目不能用靜態檢查取代。

README 提供 `doctor`、`smoke_sam` 與完整啟動命令，讓目標主機逐一完成驗證。最終速度及硬體可行性需以該主機產生的日誌與 CSV 判定。
