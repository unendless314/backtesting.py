# Known Issues（已知問題與技術債）

記錄「已發現、但暫不處理」的問題，等有空時再逐項清理。每項都獨立，可分開處理、分開 commit。

- 處理完的項目：把狀態改成 ✅ 並註明 commit，或直接刪除該項。
- 新發現的問題：直接加在對應區塊的最後面，寫清楚「問題、影響、建議作法」。
- 較早（2026-09-11）整理的 `scripts/` 技術債有更完整的說明，見
  [`proposals/2026-09-11_scripts_tech_debt_notes.md`](proposals/2026-09-11_scripts_tech_debt_notes.md)；
  下方「舊清單摘要」只列標題，避免同一件事記兩次。

最後更新：2026-11-07

---

## 新增項目

### KI-1 找資料檔的規則在 4 支腳本裡各寫一份
- **來源**：2026-11-07 價格 PR 工具審查
- **問題**：「`data/raw/{代號}USDT_1d.csv` → `data/raw/{代號}_1d.csv`」這條找檔規則，
  分別寫在 `calculate_price_pr.py`（`resolve_csv_path`）、`analyze_term_structure.py`、
  `generate_asset_report.py`、`test_strategy_dip_buy.py`。
- **影響**：以後改命名規則時容易漏改，導致某支腳本找不到檔案。
- **建議作法**：把 `resolve_csv_path()` 移到共用模組（例如 `scripts/_range_utils.py`，
  或新建 `scripts/_data_paths.py`），四支腳本都改用它。

### KI-2 其他三支研報腳本沒有檢查 `--symbol` 的內容
- **來源**：2026-11-07 價格 PR 工具審查
- **問題**：`calculate_price_pr.py` 已用 `safe_file_stem()` 把代號轉成安全檔名
  （`BTC/USDT` → `BTC_USDT`，`^GSPC`、`EURUSD=X` 保持原樣），但另外三支腳本
  仍直接把 `--symbol` 放進輸出檔名。代號裡如果有 `/` 或 `../`，報告會被寫到
  `research/reports/` 以外的地方。
- **影響**：只有自己打錯字才會發生，不是安全漏洞，但可能覆寫到其他檔案。
- **建議作法**：與 KI-1 一起做，把 `safe_file_stem()` 移進共用模組，四支腳本共用。
  注意不要改成「只允許英數字」：那樣會擋掉 `^GSPC`、`EURUSD=X` 等正式的 Yahoo 代號。

### KI-3 舊腳本的測試仍會在真實報告資料夾寫入、刪除檔案
- **來源**：2026-11-07 價格 PR 工具審查（與舊清單 #4 同一件事，這次再次確認）
- **問題**：`TestScriptDateFiltering` 的 `tearDown` 會刪除 `research/reports/TEST_*.md`。
  如果你自己剛好有 `TEST` 開頭的報告，跑完測試就會被刪掉。
- **已示範的修法**：`calculate_price_pr.main()` 加了 `report_dir` 參數，測試時寫到暫存資料夾；
  其他三支腳本可以照這個方式改。

### KI-4 沒有實際在 Python 3.9 上跑過
- **問題**：`setup.py` 宣稱支援 Python >= 3.9，但 `.venv` 是 3.13，也沒有自動測試（CI）
  在 3.9 上跑。2026-11-07 曾發生 `zip(strict=True)`（3.10 才支援）被寫進程式、
  測試全部通過卻沒被發現的情況。
- **建議作法**：二選一。(a) 裝一個 3.9 環境定期跑 `python -m unittest scripts.test_scripts`；
  (b) 如果不打算支援舊版，把 `python_requires` 提高到實際使用的版本。

### KI-5 `flake8` 沒有安裝在 `.venv`
- **問題**：AGENTS.md 把 `flake8 backtesting` 列為第二道風格檢查，但環境裡沒有安裝，
  目前只有 ruff 在檢查。
- **建議作法**：`pip install -e '.[test,doc,dev]'` 補齊開發工具；或從 AGENTS.md 移除 flake8。

---

## 舊清單摘要（詳見 proposals/2026-09-11_scripts_tech_debt_notes.md）

| 編號 | 項目 | 狀態 |
| :--- | :--- | :--- |
| #1 | 測試合成資料沒有時區，跟真實 Binance 資料不一致 | 待處理 |
| #2 | 三支腳本的統計計算（勝率、凱利公式等）重複約 100 行 | 待處理 |
| #3 | `ruff check scripts` 約 270 項警告；pyproject 的 ruff 設定寫法已過時 | 待處理（`calculate_price_pr.py` 已單獨通過） |
| #4 | 測試把報告寫進真實的 `research/reports/` | 部分完成（PR 工具已改；見 KI-3） |
| #5 | 腳本假設 `research/reports/` 已存在 | 部分完成（PR 工具已會自動建立） |
| #6 | 產生的報告是否進版控 | ✅ 2026-11-07 決定不進版控（選項 B）：`.gitignore` 忽略 `research/reports/*`，只保留 `.gitkeep` |
| #7 | `proposals/` 納入版控 | ✅ 已在 commit 4c506db 完成 |
| #8 | `.DS_Store` 進了版控、README 中 `python`／`python3` 寫法不一致 | 部分完成（`.DS_Store` 已不在版控；指令寫法仍不一致） |
