# 技術債清理備忘錄：scripts/ 目錄體質改善

- **文件日期**: 2026-09-11
- **文件狀態**: 待處理（Backlog，尚未排定時程）
- **來源**: 2026-09-11 實作「統一日期與持有期模型」（見 `2026-09-10_unified_start_end_days_support.md`）過程中發現的既有問題
- **範圍**: 主要是 `scripts/` 目錄；核心套件 `backtesting/` 品質良好，不在此列
- **給執行者的話**: 本文件每一節都是獨立任務，可分開實作、分開 commit。建議依「優先級」順序處理，每項做完跑 `python -m unittest scripts.test_scripts` 確認無迴歸。

---

## 背景說明

這個 repo 由開源專案 backtesting.py 複製而來，核心套件（`backtesting/`）有完整測試與一致的風格；`scripts/` 是後來自行加入的研究腳本，長期未經 lint 與測試把關，累積了一些技術債。2026-09-11 的功能變更（統一 `--start` / `--end` / `--days`）已修掉其中一個實際 bug（錯誤路徑崩潰），其餘問題記錄於此。

---

## 優先級 P1：測試與正確性

### 1. 測試用的合成資料與真實資料格式不一致（時區）

**問題**：`scripts/test_scripts.py` 的合成 CSV 使用 tz-naive 日期（`pd.date_range(..., freq="D")`），但 Binance 下載的真實 CSV 日期帶 `+00:00` 時區（tz-aware）。tz-naive 與 tz-aware 的日期在 pandas 中**不能互相比較**，會直接拋 `TypeError`。

**實際後果**：2026-09-11 的變更在測試全過的情況下，於真實資料執行時崩潰，需額外加入 `entry_window_mask()` 做時區相容處理。也就是說，**現有測試無法擋住「只在真實資料上才出現」的錯誤**。

**建議作法**：將測試合成資料的日期改為 tz-aware（例如 `pd.date_range(..., freq="D", tz="UTC")`），與 `data/raw/` 的真實 CSV 對齊。改完後現有 16 個測試應仍全數通過；若有測試因此失敗，代表該處程式碼確實有時區處理缺口，應一併修正。

**相關檔案**：`scripts/test_scripts.py`（`TestKellyCriterion`、`TestScriptDateFiltering` 的 `_write_rising_csv` 與各測試的 DataFrame 建立處）

---

### 2. 三支腳本間的統計計算程式碼大量重複

**問題**：`scripts/generate_asset_report.py` 與 `scripts/test_strategy_dip_buy.py` 中，「勝率 / 賠率 / 平均值 / 中位數 / 四分位 / 凱利公式四種變體 / 偏度」這一整段（約 100 行）幾乎逐行相同；期限結構腳本也有一小段重複。

**風險**：日後修改算法（例如調整凱利公式的保守情境定義）時必須改多個地方，容易漏改導致三份報告口徑不一致——這正是 2026-09-11 變更中抽出 `_range_utils.py` 想要避免的「語義漂移」。

**建議作法**：比照 `_range_utils.py` 的模式，新增 `scripts/_stats_utils.py`（命名可再議），抽出如 `compute_return_stats(roi_series)` 之類的純函式，回傳包含勝率、四分位、凱利倉位等欄位的 dict；三隻腳本改用雙軌匯入（try 相對匯入 / except 直接匯入，寫法見 `_range_utils.py` 與現有三腳本開頭）。Markdown 表格的**排版**可留在各腳本，只抽「計算」。

**驗收重點**：抽取前後，對同一支真實 CSV 產生的報表統計數字必須完全一致（可用 git stash 對比法，或先存舊報表再比對）。

---

## 優先級 P2：工程衛生

### 3. `ruff check scripts` 基線約 270 項警告，從未通過

**問題**：`scripts/` 目錄自建立以來未納入 lint 把關，2026-09-11 實測基線約 269–271 項錯誤。分佈（依數量）：`F541`（無佔位符的 f-string）、`RUF001`（字串中的全形標點）、`W293`（空白行含空白字元）、`E501`（行超過 100 字元）、`T201`（使用 print）、`RUF003`（註解中的全形標點）等。

**附帶問題**：`pyproject.toml` 的 ruff 設定使用已棄用的頂層寫法（`select` / `ignore` / `pep8-naming`），新版 ruff 會警告應移至 `[tool.ruff.lint]` 區段。

**建議作法**（分兩個 commit，降低 review 雜訊）：
1. 先跑 `ruff check scripts --fix` 自動修復可修項（約 140+ 項，主要是空白與 import 排序），人工檢視 diff。
2. 再決定剩下的類別如何處理：
   - `RUF001` / `RUF003`（全形標點）：腳本輸出為繁體中文報表，全形標點是**有意的**，建議在 pyproject 中對 `scripts/` 關閉這兩條規則（per-file-ignores），而不是把中文標點改成半形。
   - `T201`（print）：這些腳本本就是 CLI 工具，print 是正常輸出管道，同樣建議對 `scripts/` 關閉。
   - `F541`、`E501`：逐一修正（f-string 拿掉多餘的 `f` 前綴；長行以字串串接或換行處理）。
3. 更新 `pyproject.toml` 的 ruff 設定寫法到 `[tool.ruff.lint]`。

**目標**：`ruff check scripts` 回歸 0 錯誤，並可考慮加入 CI 或 commit 前檢查，避免再度累積。

---

### 4. 測試會將報告寫入真實的 `research/reports/` 目錄

**問題**：`scripts/test_scripts.py` 的測試直接呼叫腳本函式，而腳本固定把報告寫到 `research/reports/`；目前依賴 `tearDown` 的 glob 清理（`TEST_*.md`）避免殘留。

**風險**：清理規則若與檔名規則脱节（例如未來又改檔名格式），測試殘骸會留在真實報告目錄中；測試中斷（例如強制終止）也會留下檔案。

**建議作法**：讓三隻腳本的輸出目錄可參數化（例如函式加 `out_dir` 參數，預設 `research/reports`），測試時指向 `tempfile.TemporaryDirectory()`，即可完全移除 tearDown 的 glob 清理。此改動可與第 2 項（抽共用統計模組）一併進行，因為都會動到函式簽名。

---

### 5. 腳本假設 `research/reports/` 目錄已存在

**問題**：三隻腳本寫檔時直接 `open(Path("research/reports") / filename, "w")`，全新 clone 的環境若該目錄不存在會拋 `FileNotFoundError`。

**建議作法**：寫檔前加一行 `out_path.parent.mkdir(parents=True, exist_ok=True)`。若實作了第 4 項的 `out_dir` 參數，在共用入口統一處理即可。

---

## 優先級 P3：版控與文件慣例（需使用者決策）

### 6. 產生的報告是否該進版控？

**現況**：`research/reports/*.md` 有 commit 進 git 的慣例（歷史上有 `DATA:` 類 commit），但報告是「資料 × 程式」的產物——資料持續更新，報告就會持續漂移。實際案例：git 中的 `BTC_Strategy_DipBuy_11p15pct_Drop.md` 是用 2026-02 的資料算的，與目前資料（2026-09）結果不同，已是過期文件。

**選項**：
- A. 維持現狀：報告進版控，接受會過期，定期以 `DATA:` commit 批次更新。
- B. 報告不進版控：把 `research/reports/` 加入 `.gitignore`（與 `data/**` 一致），報告純粹是本地產出物。
- C. 折衷：只 commit「有紀念價值」的分析（特定研究結論），日常重跑的報告不進版控。

**建議**：若報告主要用途是自己閱讀而非對外發表，選 B 最省事；若想保留「某時間點的分析結論」可查，選 C。此為使用習慣問題，無標準答案。

---

### 7. `proposals/` 目錄未納入版控

**現況**：`proposals/` 整個目錄仍是 untracked，但裡面的定稿計畫（含審閱紀錄）是很重要的決策文件。

**建議**：commit 起來（可用 `DOC:` 前綴），讓設計決策與程式碼演化一起留痕。本文件亦同。

---

### 8. 雜項

- **`.DS_Store` 已進版控**：macOS 系統檔（`backtesting/.DS_Store`、`doc/.DS_Store` 等）被 commit 進來了。建議在 `.gitignore` 加上 `.DS_Store`，並用 `git rm --cached` 移除已追蹤的檔案。
- **未使用的 import**：本次已順手移除兩個 `import numpy as np`；清理 lint（第 3 項）時 ruff 會把剩下的抓出來。
- **README 指令不一致**：`scripts/README.md` 用 `python3`，`AGENTS.md` 用 `python`，對新手來說啟動 venv 後兩者通常等價，但可統一以减少困惑。

---

## 建議執行順序

| 順序 | 項目 | 性質 | 預估工作量 |
| :--- | :--- | :--- | :--- |
| 1 | P1-1 測試資料時區對齊 | 防禦未來 bug | 小 |
| 2 | P2-5 輸出目錄 mkdir | 一行修正 | 極小 |
| 3 | P1-2 抽共用統計模組（可併入 P2-4 的 `out_dir` 參數） | 去重、防語義漂移 | 中 |
| 4 | P2-3 lint 大掃除 + pyproject ruff 設定更新 | 工程衛生 | 中 |
| 5 | P3-6 / P3-7 / P3-8 | 版控慣例 | 小，但需使用者先決策 |

## 注意事項

- 每項任務完成後的最低驗證：`python -m unittest scripts.test_scripts` 全數通過；涉及報表內容的改動（項目 2、3）需加做「同一真實 CSV 的統計數字前後一致」對比。
- 2026-09-11 變更已建立的慣例請沿用：錯誤時印中文訊息、不寫檔、exit 1；共用邏輯放 `scripts/_*.py` 並用雙軌匯入。
