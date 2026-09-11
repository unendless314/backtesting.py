# 每日自動化數據更新與動態買點雷達系統提案 (RFC)

- **文件日期**: 2026-09-11
- **文件狀態**: 提案討論中 (Draft / Ready for Review)
- **目標讀者**: 專案成員與協同工程師
- **相關文件**: `proposals/2026-09-11_scripts_tech_debt_notes.md`

---

## 1. 背景與核心動機 (Context & Motivation)

目前專案內的量化分析體系（包含期限結構階梯、買入持有回測、逢低抄底策略）已具備良好的分析模型。然而在實際使用場景中，整體流程屬於**「手動研究模式」**：
1. 人工手動下指令抓取資料（Binance / Yahoo Finance）。
2. 人工個別呼叫分析腳本產出 Markdown 研報。
3. 人工手動比對數值，確認目前價格是否便宜。

當長期關注的標的擴展至 5～15 檔（涵蓋加密貨幣如 BTC、ETH，以及美股 ETF/個股如 VOO、BABA、QQQ 等）時，手動更新成本顯著增加。本提案旨在設計一套**「無人值守的自動化定時管線」**，讓系統每天自動抓取最新數據、以自適應的動態分位數計算當日買點狀態，並透過便利的管道隨時查閱。

---

## 2. 關鍵量化邏輯升級：動態百分位數 (Dynamic Percentile Rank)

在早期的靜態測試中，逢低買入策略採用寫死的數值門檻（例如 BABA 的 `-16.90%`、BTC 的 `-11.15%`）。

### 痛點：門檻隨時間動態漂移
每天資料更新時，都會新增一筆交易日與新的過去 1 年報酬率樣本，導致歷史的 Q1（25% 悲觀分位數）、中位數與 Q3 隨時間**緩慢漂移**。若在設定檔中固定數值，過了一兩年市場週期變動後，靜態數值將偏離「歷史超跌區」的定義。

### 解決方案：改採「動態分位點 (Percentile Rank)」
每天系統更新資料後，不比較固定趴數，而是進行兩步計算：
1. **計算今日漲跌幅**：$R_{\text{today}} = \frac{\text{Close}_{\text{today}} - \text{Close}_{\text{1 year ago}}}{\text{Close}_{\text{1 year ago}}}$
2. **計算歷史分位數**：將 $R_{\text{today}}$ 放入該標的歷史所有 1 年報酬率母體中，計算其**百分位數（Percentile）**。

#### 優勢：
* **零人工維護成本**：設定檔不用為每檔標的手動查填趴數，統一採用百分位邏輯。
* **跨資產天然自適應**：自動調和高波動資產（加密幣）與低波動資產（美股指數）的波動率差異。
* **分級燈號明確**：
  * **⚪ 觀望 / 多頭**（分位數 > 25%）：價格未進入歷史超跌區。
  * **🟢 適合建倉 (Q1 超跌)**（分位數 $\le 25\%$）：落入歷史最差 25% 區間，滿足均值回歸條件。
  * **🔥 重度超跌 (極端機會)**（分位數 $\le 10\%$）：歷史前 10% 罕見底部（如重大股災、週期極底）。

---

## 3. 系統整體架構概念 (Conceptual Architecture)

```
[ 標的監控設定檔 (Watchlist Config) ]
                  │
                  ▼
[ 每日自動化總控腳本 (Daily Pipeline Runner) ]
  ├── 1. 增量/全量更新數據 (Binance / Yahoo Finance)
  ├── 2. 今日訊號雷達 (計算各標的今日 1年報酬 & 歷史分位數)
  └── 3. 自動刷新研報與彙整看板 (Markdown / Dashboard)
                  │
                  ▼
[ 結果呈現與即時交付 (Delivery Channel) ]
  ├── 選項 A: 通訊推播 (Telegram / LINE / Discord)
  ├── 選項 B: 雲端網頁儀表板 (Streamlit / 靜態 HTML)
  └── 選項 C: GitHub 看板 (自動 Commit & Push)
```

---

## 4. 模組分工與設計原則 (High-Level Modules)

### 模組 A：標的監控清單 (Watchlist Config)
* 採用輕量格式（如 JSON 或 YAML）。
* 欄位概念：
  * `symbol`: 標的代碼（如 `BTC`, `BABA`, `VOO`）。
  * `source`: 資料源（`binance` 或 `yfinance`）。
  * `active`: 是否啟用每日監控。
  * `alert_percentile`: 警示分位門檻（預設 0.25）。

### 模組 B：每日自動管線 (Daily Pipeline)
* 負責協調整個流程的 Python 入口腳本。
* 具備容錯與日誌機制：若某一檔標的網路請求失敗，不應中斷其他標的的更新。
* 輸出產物：
  * 更新後的 `data/raw/*.csv`。
  * 更新後的 `research/reports/*.md`。
  * 每日買點摘要彙整（例如 `research/reports/DAILY_RADAR.md`）。

### 模組 C：排程與交付 (Delivery & Scheduling)
* 定時於收盤後或固定時段觸發（如每日清晨）。
* 將「今日雷達摘要」透過最省力的方式呈現給使用者。

---

## 5. 與工程師討論時的技術選型決策點 (Discussion Points for Engineering)

具體採用的技術與實作方式，建議在實作前與工程師針對以下議題評估：

### 決策點 1：部署環境與排程方案
* **選項 1（自有 Linux VPS）**：
  * *優點*：掌控度最高、無執行時間限制、易於部署即時 Web 服務或常駐 Bot。
  * *排程機制*：Linux `crontab` 或 `systemd timer`。
* **選項 2（GitHub Actions 無伺服器架構）**：
  * *優點*：完全免費（善用免費每月份額）、免維護主機安全性與環境。
  * *排程機制*：GitHub Workflow cron schedule，跑完自動 commit 回 repo 並推播通知。

### 決策點 2：結果查閱介面（使用者體驗）
* **選項 1（通訊軟體 Bot 主動推播 - 最推薦）**：
  * 串接 Telegram Bot API 或 Discord Webhook。
  * 優點：使用者最無負擔，每天早上被動接收「今日買點燈號清單」，無需手動打開任何網站。
* **選項 2（輕量 Web 介面）**：
  * 使用 Streamlit、Gradio 或靜態 HTML（Nginx 託管）。
  * 優點：視覺化走勢圖、歷史百分位分佈圖較豐富。
* **選項 3（純 Markdown / GitHub Pages）**：
  * 每日更新一頁 `DAILY_RADAR.md`，手機直接用 GitHub App 查閱。

### 決策點 3：資料儲存與重構相依性
* **資料持久化**：目前維持 CSV 是否足夠？或隨著標的增多，是否考慮 Parquet / SQLite？
* **程式碼體質對齊**：是否先落實 `proposals/2026-09-11_scripts_tech_debt_notes.md` 中提議的 `scripts/_stats_utils.py` 抽離共用統計邏輯，以方便 Pipeline 乾淨調用？

---

## 6. 階段性實作建議 (Phased Roadmap)

1. **Phase 1（核心邏輯與本地驗證）**：
   * 建立 `watchlist.json`。
   * 完成 `run_daily_pipeline.py`（支援批量更新數據 + 計算今日動態分位數 + 輸出終端/Markdown 看板）。
2. **Phase 2（推播通知機制）**：
   * 串接 Telegram Bot / Webhook，確認通知格式簡潔實用。
3. **Phase 3（雲端部署與自動排程）**：
   * 決定採用 VPS 或 GitHub Actions，設定定時排程，進入全自動無人值守運作。
