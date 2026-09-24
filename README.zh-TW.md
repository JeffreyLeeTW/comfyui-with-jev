# comfy-agent

[![English](https://img.shields.io/badge/lang-English-lightgrey)](README.md) [![繁體中文](https://img.shields.io/badge/lang-%E7%B9%81%E9%AB%94%E4%B8%AD%E6%96%87-blue)](README.zh-TW.md)

構想（文字和/或圖片）→ **OpenRouter** 免費模型或本地 **Ollama** 產生 prompt → **Jev**（TypeSafe）評審 → **ComfyUI** 生圖。

```
構想 (text / image)
      │
      ▼
LLM 產生 positive / negative prompt ◄─────────────┐
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
- 選用：[Ollama](https://ollama.com/) server（預設 `http://10.0.0.15:11434`），可以取代 OpenRouter，不需要 API key

## 安裝

```bash
uv sync
cp .env.example .env   # 填入 OPENROUTER_API_KEY、TYPESAFE_API_KEY
uv run comfy-agent check
```

`check` 會分別測試 ComfyUI、OpenRouter、Ollama 和 TypeSafe 的連線。你實際會用到的服務都要顯示 `[ok]`；只用其中一個 provider 的話，另一個顯示失敗沒關係。

## 使用方式

### CLI

```bash
# 列出模型（--vision 只列出能看圖的模型；--provider/-p 選 openrouter 或 ollama，預設看 config.yaml）
uv run comfy-agent models
uv run comfy-agent models --vision
uv run comfy-agent models -p ollama

# 改用本地 Ollama 產生 prompt
uv run comfy-agent run --idea "..." -p ollama --model gemma4:e4b

# 純文字 → txt2img（沒給 --model 時，會列出模型讓你選）
uv run comfy-agent run --idea "a girl reading under cherry blossoms"

# 文字 + 參考圖 → img2img
uv run comfy-agent run --idea "same girl at night" --image ref.png --model qwen/qwen3.8-27b:free

# 只產生 prompt 並評分，不送去生圖
uv run comfy-agent run --idea "..." --no-render

# 指定內容分級（WebUI 用「內容分級」選項）：sfw 或 nsfw，不加就不指定
uv run comfy-agent run --idea "..." --rating sfw

# Prompt 風格（WebUI 用「Prompt 風格」選項）：natural（英文自然語言，預設）或 tags（danbooru tag）
uv run comfy-agent run --idea "..." --style tags

# Jev 模式：on（預設）／off（產生一次就生圖，Jev 只評分供參考）／ab（兩種各生一組、同 seed 對照）
uv run comfy-agent run --idea "..." --judge ab --report

# 把任一次的紀錄匯出成 HTML 報告（存到 runs/reports/；--lang en 或 zh-TW）
uv run comfy-agent report runs/20260924-120000.json --lang zh-TW
```

`run` 可以用的參數（沒有指定的就用 `config.yaml` 的值）：

| 類別 | 參數 |
| --- | --- |
| 模型／流程 | `--provider/-p`（`openrouter`／`ollama`）, `--model/-m`, `--max-attempts`, `--rating`（`sfw`／`nsfw`）, `--style`（`natural`／`tags`）, `--judge`（`on`／`off`／`ab`）, `--report`（另外匯出 HTML 報告）, `--lang`（報告語言：`en`／`zh-TW`） |
| Jev 門檻（0–1） | `--t-fidelity`, `--t-format`, `--t-completeness`, `--t-negative` |
| 生圖 | `--width`, `--height`, `--batch-size`, `--seed`（-1 = 隨機）, `--steps`, `--cfg`, `--sampler`, `--scheduler`, `--denoise`（只有 img2img 用得到） |
| LoRA | `--lora name=strength`（可以重複指定多個） |
| 其他 | `--config path/to/config.yaml` |

### WebUI

```bash
uv run comfy-agent webui            # http://127.0.0.1:7860
uv run comfy-agent webui --port 8000
```

- 右上角可以切換介面語言（English／繁體中文），預設值是 `config.yaml` 的 `ui.language`；匯出的報告也會使用目前的語言。
- 「Prompt 產生器」可以切換 OpenRouter／Ollama，模型清單會跟著更新。
- 模型下拉選單會依有沒有上傳圖片，自動只列出能看圖的模型；↻ 可以重新整理清單。
- 「Jev 評審」可選使用 Jev／不用 Jev／A-B 對照，說明見下方〈Jev 模式〉。
- 「Jev 門檻」和「生圖參數」兩個折疊區塊，可以調整每次執行的設定。
- 「連線設定」可以改 ComfyUI 和 Ollama 的 IP 與 port。按「儲存並測試連線」會寫入 `.env` 的 `COMFYUI_URL`／`OLLAMA_URL`（優先於 `config.yaml`，CLI 也會讀），並立刻測試連線；下一次執行就生效，不用重開。
- 右側會即時顯示每一輪的分數、最後送出的 prompt，以及直接從 server 讀取的結果圖。
- 跑完後按「匯出報告（HTML）」可以下載這次的報告。

## 設定：`config.yaml`

| 區塊 | 重點 |
| --- | --- |
| `comfyui` | `url`、`workflow`（API 格式的 workflow template）、`timeout_s` |
| `llm` | `provider`：預設用哪個產生 prompt（`openrouter`／`ollama`），每次執行時都可以改 |
| `openrouter` | `default_model`（空白 = 每次執行時再選）、`temperature` |
| `ollama` | `url`、`default_model`、`temperature`、`think`（thinking 模型要不要先思考，預設關掉以加快速度）、`timeout_s`（包含把模型載入 VRAM 的時間） |
| `judge` | `model`（`jev-latest`）、`max_attempts`（預設 5）、`thresholds`（每個面向的門檻） |
| `prompt` | `style`：預設的 prompt 風格，`natural`（anima_baseV10 看得懂英文句子）或 `tags`。`positive_prefix`／`negative_base`：兩種風格都會自動加在最前面的固定 tag，例如品質 tag 和 LoRA 觸發詞。`ratings.sfw`／`ratings.nsfw`：選擇內容分級時強制加入的 positive／negative tag，選擇的分級也會告訴生成模型和 Jev。tag 風格下，模型寫在相反一側的分級 tag 會被移除；自然語言風格不改動句子，衝突交給 Jev 的 `fidelity`／`negative` 檢查 |
| `generation` | 生圖參數預設值；`lora_strengths` 以 `lora_name` 為 key |
| `runs_dir` | 本機存放執行紀錄的資料夾 |
| `ui` | `language`：WebUI 和 HTML 報告的預設語言（`en`／`zh-TW`，預設 `en`） |

## Jev 評分標準

每一版 prompt 會送一次 request，同時評 4 個 Score 面向。每個面向分 4 個等級，分數正規化成 0–1：

| 面向 | 評什麼 |
| --- | --- |
| `fidelity` | positive prompt 有沒有忠實呈現構想（有參考圖時，也對照圖片描述） |
| `format` | tag 風格：是不是英文、逗號分隔的 danbooru tag，沒有句子，也沒有互相矛盾的 tag。自然語言風格：固定 tag 之後是不是通順、具體的英文句子，不是 tag 清單，也沒有矛盾 |
| `completeness` | 主體、外觀、姿勢／構圖、場景／光線有沒有交代完整 |
| `negative` | negative prompt 是否合理，有沒有排除掉構想要的東西 |

- **及格**：4 個面向都達到各自的門檻（預設都是 0.67）。
- **不及格**：把沒過的面向、分數和 Jev 判斷的等級寫成評語，連同上一版 prompt 一起交給 LLM 重寫。
- **次數用完**：比較每一版最弱的那個面向，挑這個分數最高的一版送出，並標記為「未達標」。
- Jev 只吃文字。有參考圖時，會先由 LLM 產生一段圖片描述，交給 Jev 當判斷依據。
- 評分題目和各等級的描述定義在 `src/comfy_agent/judge.py` 的 `QUESTIONS`。

> 門檻的預設值只是起點。建議先跑幾次，看 `runs/` 裡的實際分數分布再調整。

### Jev 模式

| 模式 | 流程 | Jev 花費 |
| --- | --- | --- |
| 使用 Jev（`on`，預設） | 上面描述的評分／重寫迴圈 | 每一版 1 次 |
| 不用 Jev（`off`） | 產生 1 次就送去生圖。Jev 仍會評 1 次分，只記錄、不影響流程；沒有 `TYPESAFE_API_KEY` 或評分失敗時就跳過 | 1 次 |
| A-B 對照（`ab`） | 跑完 Jev 迴圈後，用**同一個 seed** 分別生「Jev 最後送出的版本」和「第 1 版」（也就是不用 Jev 時會送出的版本）。第 1 版就及格時兩組相同，只生一次圖 | 和 `on` 相同 |

A-B 對照直接拿 Jev 迴圈的第 1 版當「無 Jev」組，兩組從同一個起點出發，差別只在有沒有經過 Jev 回饋重寫，而且不會多花 LLM 和 Jev 的額度。

### 報告

`runs/reports/<紀錄檔名>.html` 是單一 HTML 檔，內容有構想、參數、每一輪的分數與 prompt、最後的圖；A-B 對照時會左右並排，並附各面向的分數差異。圖片從 ComfyUI 的 `/view` 讀取，不會下載到本機，所以要連 VPN 才看得到。

## ComfyUI workflow

Template 是 `workflows/anima_flow.json`，從 `example flow.json` 複製過來。程式會依節點的類型和連線找出要改的節點，不寫死節點 ID。每次執行時會做以下修改：

- **txt2img**（沒有參考圖）：KSampler 改接 `EmptyLatentImage`，denoise 設成 1.0；用不到的 LoadImage 那一串節點會被移除。
- **img2img**（有參考圖）：把圖片上傳到 ComfyUI 的 `input/`，再走 `LoadImage → VAEEncode → LatentUpscale → KSampler`，使用你設定的 denoise。
- `PreviewImage` 會換成 `SaveImage`，圖片存到 server 的 `ComfyUI/output/<filename_prefix>_xxxxx_.png`。
- 會寫入 prompt、seed、steps、cfg、sampler、scheduler、尺寸和 LoRA 強度。

要換 workflow 的話，在 ComfyUI 用 **Export (API)** 匯出 JSON，再修改 `config.yaml` 的 `comfyui.workflow`。新的 workflow 需要包含：一個 `KSampler`（positive／negative 各接一個文字編碼節點）、`EmptyLatentImage`（txt2img 用）、`LoadImage` 加 `VAEEncode`（img2img 用），以及一個輸出節點（PreviewImage 或 SaveImage）。

## 輸出

- **圖片**：只存在 server 的 `output/`，不會下載到本機。CLI 會印出 `/view` 連結，WebUI 則直接顯示。
- **紀錄**：`runs/<時間>.json`（`kind` 是 `run` 或 `compare`；`compare` 的兩組分別在 `with_jev`／`without_jev`），內容包括每一輪的 prompt、各面向的分數／信心值／等級、最後選用的是第幾輪、是否達標、生圖參數、seed 和 ComfyUI 的 prompt_id。模型拒絕時會多一個 `refused` 欄位記錄原因。
- **拒絕紀錄**：`runs/refusals.jsonl`，每行一筆（時間、provider、模型、txt2img／img2img、第幾輪、原因、構想）。`models` 指令和 WebUI 的模型選單會標出拒絕過幾次。

## 專案結構

```
config.yaml               預設設定
workflows/anima_flow.json ComfyUI API workflow template
src/comfy_agent/
  config.py      讀取 config.yaml 和 .env
  i18n.py        WebUI 與報告的英文／繁體中文字串
  openrouter.py  共用的 prompt 產生邏輯（PromptWriter）與 OpenRouter 後端
  ollama.py      Ollama 後端（原生 /api/chat，可關掉 thinking）
  llm.py         依 provider 建立對應的後端
  refusals.py    模型拒絕紀錄
  judge.py       Jev 評分題目、及格判定、評語
  comfyui.py     修改 workflow、上傳圖片、送出並等待生圖
  pipeline.py    生成 → 評審 → 生圖的主流程，並寫出執行紀錄
  cli.py         CLI（typer）
  report.py      把紀錄轉成 HTML 報告
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
| `check` 顯示 ComfyUI 失敗 | 確認 VPN 已連線，並且位址正確（`.env` 的 `COMFYUI_URL` 優先，其次是 `config.yaml` 的 `comfyui.url`） |
| OpenRouter 回應 429 或 503 | 免費模型常被限流，程式會自動重試幾次；還是不行就換一個模型 |
| `Model did not return valid prompt JSON` | 這個模型不太會照格式輸出 JSON，換一個模型 |
| `refused ... stopping (nothing rendered)` | 模型拒絕產生 prompt（它自己回報拒絕、供應商的內容過濾擋下，或回覆中出現常見的拒絕句型）。整個流程會立刻停止、不生圖，並記錄到 `runs/refusals.jsonl`。換一個模型 |
| ComfyUI rejected the workflow | 錯誤訊息會附上 node_errors，通常是 server 上缺少模型或 LoRA 檔案，或是 sampler、scheduler 名稱打錯 |
| 每次都不及格 | 查看 `runs/` 裡是哪個面向偏低，再調整門檻或 `judge.py` 裡的評分標準 |
