# Hackathon Vision Server（VLM）

機器人視覺追蹤的「找目標」服務：client（Pi 5 上的 `vlm_bridge_node`）用 ZeroMQ 傳 JPEG 上來，
server 依上級描述（例如 `the white paper cup`）用 **LocateAnything-3B** 找出目標，回傳 bbox（可選 SAM2 mask），
由 client 在 bbox 內用深度切出物件並追蹤。

- 協定：[vlm_transport.md](vlm_transport.md)
- 進度：[server_progress.md](server_progress.md)（server 端）、[client_progress.md](client_progress.md)（client 端）
- 聯測與除錯：[DEBUG.md](DEBUG.md)
- 待辦與規劃：[TODO.md](TODO.md)

## 目錄

| 路徑 | 內容 |
|---|---|
| `vlm_server/` | server 本體：ZMQ ROUTER（5555）、HTTP 描述 API（8080）、推論 pipeline |
| `app/` | 從 V3 繼承的程式。`vlm_server` 目前用到 `models.py`（Locator）、`core.py`、`gpu_check.py` |
| `tools/` | `mock_server.py`（不需要模型的假 server）、`test_client.py`（量延遲、畫結果） |
| `scripts/` | `smoke_pipeline.py`（在容器內用真模型跑單張圖）等 |
| `tests/` | 單元測試（`tests/test_vlm_server.py` 不需要 GPU 與模型） |
| `deploy/` | `deploy.sh`：在 Hackathon-gpu 上部署 |
| `eval/images/` | 現場實拍的評估圖片 |
| `compose.yaml` + `compose.rocm.yaml` + `compose.vlm.yaml` | 三份一起用才是 VLM server；`compose.vlm.yaml` 設定 port、`LA_MODE` 等 |

## 部署（Hackathon-gpu）

server 跑在 `wildbot@192.168.50.125:~/Documents/vlm-server`（ssh alias `Hackathon-gpu`），
那是這個 repo 的 git 工作目錄。模型不在 repo 裡，放在 `~/models/vlm`（`.env` 的 `LOCATE_MODELS_DIR`）。

### 日常部署

```bash
git push                                                   # 本機
ssh Hackathon-gpu '~/Documents/vlm-server/deploy/deploy.sh'
```

`deploy.sh` 會依序：checkout `origin/main` → build image → 在新 image 裡跑單元測試 → 重啟 → 等模型就緒 → 把 commit 記到 `output/DEPLOYED`。
build 或測試失敗時不會重啟，server 繼續跑舊版本。

**重啟後描述會消失**，要重新設定（或在 `.env` 設 `VLM_INITIAL_QUERY`）：

```bash
curl -X POST http://192.168.50.125:8080/api/query -H 'Content-Type: application/json' -d '{"text":"the white paper cup"}'
```

### 回滾

```bash
ssh Hackathon-gpu '~/Documents/vlm-server/deploy/deploy.sh <commit>'
ssh Hackathon-gpu 'cat ~/Documents/vlm-server/output/DEPLOYED'     # 目前部署的 commit
```

### 在新主機上第一次部署

```bash
git clone <repo> ~/Documents/vlm-server && cd ~/Documents/vlm-server
cp .env.example .env            # 設定 LOCATE_MODELS_DIR
deploy/deploy.sh
```

模型檔：`locate-anything-q8_0.gguf`（5.83 GiB），mask 模式另需 `sam2.1_hiera_tiny.pt`，放進 `LOCATE_MODELS_DIR`。

## 本機開發

```bash
python3 -m venv .venv
.venv/bin/pip install pyzmq numpy opencv-python-headless Pillow Flask waitress
.venv/bin/python -m unittest tests.test_vlm_server tests.test_core

# 不需要模型的假 server，給 client 開發用
.venv/bin/python -m tools.mock_server --query "the cup" --delay 1.9
.venv/bin/python -m tools.test_client --endpoint tcp://127.0.0.1:5555 --ping 5 --image eval/images/lab-desk-cup-bottle.png
```
