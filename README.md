# comfy-agent

構想（文字和/或圖片）→ **OpenRouter** 免費模型產生 prompt → **Jev**（TypeSafe）評審 → **ComfyUI** 生圖。

```
構想 (text / image)
      │
      ▼
OpenRouter 產生 positive / negative prompt ◄──────┐
      │                                           │ 不及格：帶著 Jev 的評語重寫
      ▼                                           │（最多 max_attempts 次）
Jev 用 4 個面向評分 ── 任一面向低於門檻 ──────────────┘
      │ 全部面向都達到門檻
      │（或次數用完 → 改送分數最高的一版，並標記「未達標」）
      ▼
ComfyUI：有圖片走 img2img，沒有圖片走 txt2img → 圖片存到 server 的 output/
```

## 需求

- Python ≥ 3.12，使用 [uv](https://docs.astral.sh/uv/) 管理環境
- 本機要能連到 ComfyUI server（預設 `http://10.0.0.15:8188`，需要 VPN）
- API key：[OpenRouter](https://openrouter.ai/keys)、[TypeSafe](https://console.typesafe.ai/)

## 安裝

```bash
uv sync
cp .env.example .env   # 填入 OPENROUTER_API_KEY、TYPESAFE_API_KEY
uv run comfy-agent check
```

`check` 會分別測試 ComfyUI、OpenRouter 和 TypeSafe 的連線，三項都要顯示 `[ok]`。

## 使用方式

### CLI

```bash
# 列出免費模型（--vision 只列出能看圖的模型）
uv run comfy-agent models
uv run comfy-agent models --vision

# 純文字 → txt2img（沒給 --model 時，會列出免費模型讓你選）
uv run comfy-agent run --idea "a girl reading under cherry blossoms"

# 文字 + 參考圖 → img2img
uv run comfy-agent run --idea "same girl at night" --image ref.png --model qwen/qwen3.8-27b:free

# 只產生 prompt 並評分，不送去生圖
uv run comfy-agent run --idea "..." --no-render
```

`run` 可以用的參數（沒有指定的就用 `config.yaml` 的值）：

| 類別 | 參數 |
| --- | --- |
| 模型／流程 | `--model/-m`, `--max-attempts` |
| Jev 門檻（0–1） | `--t-fidelity`, `--t-format`, `--t-completeness`, `--t-negative` |
| 生圖 | `--width`, `--height`, `--batch-size`, `--seed`（-1 = 隨機）, `--steps`, `--cfg`, `--sampler`, `--scheduler`, `--denoise`（只有 img2img 用得到） |
| LoRA | `--lora name=strength`（可以重複指定多個） |
| 其他 | `--config path/to/config.yaml` |

### WebUI

```bash
uv run comfy-agent webui            # http://127.0.0.1:7860
uv run comfy-agent webui --port 8000
```

- 模型下拉選單會依有沒有上傳圖片，自動只列出能看圖的模型；↻ 可以重新整理清單。
- 「Jev 門檻」和「生圖參數」兩個折疊區塊，可以調整每次執行的設定。
- 右側會即時顯示每一輪的分數、最後送出的 prompt，以及直接從 server 讀取的結果圖。

## 設定：`config.yaml`

| 區塊 | 重點 |
| --- | --- |
| `comfyui` | `url`、`workflow`（API 格式的 workflow template）、`timeout_s` |
| `openrouter` | `default_model`（空白 = 每次執行時再選）、`temperature` |
| `judge` | `model`（`jev-latest`）、`max_attempts`（預設 5）、`thresholds`（每個面向的門檻） |
| `prompt` | `positive_prefix`／`negative_base`：每次都會自動加上的固定 tag，例如品質 tag 和 LoRA 觸發詞 |
| `generation` | 生圖參數預設值；`lora_strengths` 以 `lora_name` 為 key |
| `runs_dir` | 本機存放執行紀錄的資料夾 |

## Jev 評分標準

每一版 prompt 會送一次 request，同時評 4 個 Score 面向。每個面向分 4 個等級，分數正規化成 0–1：

| 面向 | 評什麼 |
| --- | --- |
| `fidelity` | positive prompt 有沒有忠實呈現構想（有參考圖時，也對照圖片描述） |
| `format` | 是不是英文、逗號分隔的 danbooru tag，沒有句子，也沒有互相矛盾的 tag |
| `completeness` | 主體、外觀、姿勢／構圖、場景／光線有沒有交代完整 |
| `negative` | negative prompt 是否合理，有沒有排除掉構想要的東西 |

- **及格**：4 個面向都達到各自的門檻（預設都是 0.67）。
- **不及格**：把沒過的面向、分數和 Jev 判斷的等級寫成評語，連同上一版 prompt 一起交給 OpenRouter 重寫。
- **次數用完**：比較每一版最弱的那個面向，挑這個分數最高的一版送出，並標記為「未達標」。
- Jev 只吃文字。有參考圖時，會先由 OpenRouter 產生一段圖片描述，交給 Jev 當判斷依據。
- 評分題目和各等級的描述定義在 `src/comfy_agent/judge.py` 的 `QUESTIONS`。

> 門檻的預設值只是起點。建議先跑幾次，看 `runs/` 裡的實際分數分布再調整。

## ComfyUI workflow

Template 是 `workflows/anima_flow.json`，從 `example flow.json` 複製過來。程式會依節點的類型和連線找出要改的節點，不寫死節點 ID。每次執行時會做以下修改：

- **txt2img**（沒有參考圖）：KSampler 改接 `EmptyLatentImage`，denoise 設成 1.0；用不到的 LoadImage 那一串節點會被移除。
- **img2img**（有參考圖）：把圖片上傳到 ComfyUI 的 `input/`，再走 `LoadImage → VAEEncode → LatentUpscale → KSampler`，使用你設定的 denoise。
- `PreviewImage` 會換成 `SaveImage`，圖片存到 server 的 `ComfyUI/output/<filename_prefix>_xxxxx_.png`。
- 會寫入 prompt、seed、steps、cfg、sampler、scheduler、尺寸和 LoRA 強度。

要換 workflow 的話，在 ComfyUI 用 **Export (API)** 匯出 JSON，再修改 `config.yaml` 的 `comfyui.workflow`。新的 workflow 需要包含：一個 `KSampler`（positive／negative 各接一個文字編碼節點）、`EmptyLatentImage`（txt2img 用）、`LoadImage` 加 `VAEEncode`（img2img 用），以及一個輸出節點（PreviewImage 或 SaveImage）。

## 輸出

- **圖片**：只存在 server 的 `output/`，不會下載到本機。CLI 會印出 `/view` 連結，WebUI 則直接顯示。
- **紀錄**：`runs/<時間>.json`，內容包括每一輪的 prompt、各面向的分數／信心值／等級、最後選用的是第幾輪、是否達標、生圖參數、seed 和 ComfyUI 的 prompt_id。

## 專案結構

```
config.yaml               預設設定
workflows/anima_flow.json ComfyUI API workflow template
src/comfy_agent/
  config.py      讀取 config.yaml 和 .env
  openrouter.py  列出免費模型、產生 prompt（支援圖片輸入）
  judge.py       Jev 評分題目、及格判定、評語
  comfyui.py     修改 workflow、上傳圖片、送出並等待生圖
  pipeline.py    生成 → 評審 → 生圖的主流程，並寫出執行紀錄
  cli.py         CLI（typer）
  webui.py       WebUI（Gradio）
tests/           單元測試（HTTP 全部 mock）
```

## 測試

```bash
uv run pytest
```

## 常見問題

| 狀況 | 處理方式 |
| --- | --- |
| `check` 顯示 ComfyUI 失敗 | 確認 VPN 已連線，並且 `config.yaml` 的 `comfyui.url` 正確 |
| OpenRouter 回應 429 或 503 | 免費模型常被限流，程式會自動重試幾次；還是不行就換一個模型 |
| `Model did not return valid prompt JSON` | 這個模型不太會照格式輸出 JSON，換一個模型 |
| ComfyUI rejected the workflow | 錯誤訊息會附上 node_errors，通常是 server 上缺少模型或 LoRA 檔案，或是 sampler、scheduler 名稱打錯 |
| 每次都不及格 | 查看 `runs/` 裡是哪個面向偏低，再調整門檻或 `judge.py` 裡的評分標準 |
