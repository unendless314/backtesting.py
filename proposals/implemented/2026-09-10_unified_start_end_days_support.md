# 實作計畫：統一日期與持有期模型（--start / --end / --days）

- **計畫日期**: 2026-09-10
- **計畫狀態**: 已定稿 (Finalized，審閱紀錄見第 11 節)
- **取代關係**: 本計畫取代舊提案 `2026-09-10_custom_date_range_support.md`（該文件已依使用者指示刪除。其內容僅為兩隻腳本加上日期過濾；本計畫改為三腳本統一參數模型，並補強語義定義、報表揭露與錯誤處理）
- **影響範圍**:
  - `scripts/_range_utils.py` (新增，小型共用模組)
  - `scripts/analyze_term_structure.py`
  - `scripts/test_strategy_dip_buy.py`
  - `scripts/generate_asset_report.py`
  - `scripts/test_scripts.py` (新增測試類別)
  - `scripts/README.md` (文件更新)

---

## 1. 設計目標（使用者視角）

使用者的核心問題：**「不同的進場區間 × 不同的持有期，對投資績效造成什麼影響？」**

為了用簡單一致的方式回答這個問題，本計畫將三隻研報腳本統一為相同的「日期區間 + 持有期」參數模型：

| 參數 | 白話意義 | 範例 |
| :--- | :--- | :--- |
| `--start` | 最早可以進場的日期 | `--start 2021-09-10` |
| `--end` | 最晚可以進場的日期 | `--end 2022-12-31` |
| `--days` | 每筆進場往後持有幾天（前瞻持有期） | `--days 730` |

> 例外：期限結構腳本不提供 `--days`，它以固定的 1 季到 20 季（5 年）梯子一次呈現所有持有期（見 5.1）。

**日常使用流程（兩步驟）**：
```bash
# 第一步：用期限結構梯子表，看這個區間「抱多久」比較安全
python scripts/analyze_term_structure.py --symbol BNB --start 2021-09-10

# 第二步：對選定的持有期，用單期報告看細節（凱利倉位、情景分析）
python scripts/generate_asset_report.py --symbol BNB --start 2021-09-10 --days 730
```

---

## 2. 統一參數語義（核心設計，三隻腳本必須一致）

### 2.1 進場視窗語義（Entry-Window Filtering）
- `--start` / `--end` 篩選的是**進場日**，不是整筆交易的存活期間。
- **已決議**：進場日在 `--end` 之前、但持有結果發生在 `--end` 之後的交易**算數**（允許動用 end 之後的資料計算結果）。理由：使用者研究的是「在這段期間進場的表現」；這也是量化分析常規做法。
- 報表中必須**明確揭露**此語義（見第 4 節），避免誤導。

### 2.2 回溯緩衝（Lookback Buffer）— 防截斷
- 所有指標（`past_return`、`future_roi`）先在**全量資料**上計算完畢，最後才套用進場日篩選。
- 這樣指定 `--start 2021-09-10` 時，2021-09-10 當天的「過去一年漲跌幅」仍能引用 2020 年的歷史資料，不會出現指定區間第一年沒有訊號的問題。

### 2.3 右截尾處理（Right-Censoring）
- 若「進場日 + 持有天數」超過資料末端，該筆進場算不出結果，**直接捨棄**。
- 報表必須顯示「實際有效進場範圍」（捨棄後的首尾日），讓使用者知道最末端有幾天沒被算進去。

### 2.4 輸入驗證與錯誤處理
- 日期**嚴格限定 `YYYY-MM-DD` 格式**（例如 `2021-09-10`）：先做格式檢查，不合格直接報錯；不接受 `2021/09/10`、`2021-9-10` 等其他寫法。理由：與文件說明完全一致，並避免 `/` 寫法帶來的月日順序歧義。
- 以下情況給出**清楚的中文錯誤訊息並結束，不寫出報告檔、不拋例外堆疊**：
  - 日期格式錯誤（含非 YYYY-MM-DD 寫法、不存在的日期如 2021-02-30）
  - `start > end`
  - 指定區間內沒有任何有效進場樣本
- 日期超出資料範圍時不報錯，以資料的實際範圍為準（並在報表中可見）。

---

## 3. 共用模組：`scripts/_range_utils.py`（新增）

為保證三隻腳本語義完全一致，集中實作兩個小函式（約 40 行，無新依賴）：

```python
from __future__ import annotations
import re
import pandas as pd

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

def parse_date_range(start: str | None, end: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """解析並驗證 --start / --end。

    嚴格要求 YYYY-MM-DD：先以 _DATE_RE.fullmatch 檢查格式，
    再用 pd.to_datetime(..., format='%Y-%m-%d') 確認日期真實存在。
    格式錯誤或 start > end 時 raise ValueError(中文訊息)。
    """

def range_suffix(start: str | None, end: str | None) -> str:
    """產出報告檔名用的區間字串：
    都沒給 -> 'All_Time'
    只給 start -> '{start}_to_Now'
    只給 end -> 'Start_to_{end}'
    都給 -> '{start}_to_{end}'
    """
```

### 匯入方式（必要實作細節，不可省略）

`scripts/` 目錄**沒有** `__init__.py`，而三隻腳本有兩種使用方式：

1. **直接執行**：`python scripts/xxx.py`（日常用法，README 中的所有指令皆是）
2. **測試匯入**：`from scripts.xxx import ...`（`python -m unittest scripts.test_scripts` 時）

為同時支援兩者，三隻腳本匯入共用模組時**必須**使用以下雙軌寫法：

```python
try:
    from ._range_utils import parse_date_range, range_suffix
except ImportError:  # 直接執行 python scripts/xxx.py 時走這裡
    from _range_utils import parse_date_range, range_suffix
```

- 直接執行時，`sys.path` 會包含 `scripts/` 目錄本身，except 分支（`from _range_utils import ...`）生效。
- 測試匯入時，腳本作為 `scripts` 命名空間套件的一部分被載入，try 分支（相對匯入）生效。
- 注意：單純新增 `scripts/__init__.py` **不能**解決直接執行的問題（直接執行時腳本是 `__main__`，沒有套件上下文，相對匯入仍會失敗），因此此雙軌寫法為必要。

各腳本套用篩選的統一寫法：

```python
# 1. 先在全量資料算好指標（past_return / future_roi）
# 2. 再篩進場日
mask = pd.Series(True, index=df.index)
if start_ts is not None:
    mask &= df['datetime'] >= start_ts
if end_ts is not None:
    mask &= df['datetime'] <= end_ts
analysis_df = df.loc[mask].dropna(subset=[...])  # 右截尾在此自然捨棄
```

---

## 4. 報表開頭資訊統一（三隻腳本一致）

每份報告開頭固定四行：

```
**數據來源**: `BTCUSDT_1d.csv`
**完整數據範圍**: 2017-08-17 至 2026-09-09
**指定進場區間**: 2021-09-10 至 2026-09-09（或「全歷史」）
**實際有效進場範圍**: 2021-09-10 至 2025-09-10（持有 365 天，末端不足持有期的進場已捨棄）
```

並在「分析說明與風險提示」固定加一句：
> 本報告的進場日皆在指定區間內，但持有結果可能動用指定結束日之後的資料。

---

## 5. 各腳本改造細節

### 5.1 `scripts/analyze_term_structure.py`

**變更**：新增 `--start` / `--end`。**不加 `--days`**——此腳本的持有期維度就是 1Q–20Q 的梯子本身，維持固定梯子不變。

- 日期篩選放在每季迴圈內、`shift(-bars)` 計算完 ROI **之後**：
  ```python
  temp_df['future_close'] = temp_df['close'].shift(-bars)
  temp_df['roi'] = (temp_df['future_close'] - temp_df['close']) / temp_df['close']
  valid_df = temp_df.loc[mask].dropna(subset=['roi'])
  if valid_df.empty:
      continue  # 短區間時長天期自動跳過，不崩潰
  ```
- 現行的 `if len(df) <= bars: continue` 前置判斷可保留作為快速略過，但真正的空值判斷以過濾後的 `valid_df` 為準。
- 表格的 `samples` 欄位自然反映各天期的右截尾差異；在表格下方加一行說明：「較長天期的實際進場末端會提早，因為需要完整的未來資料才能計算結果」。
- 資產類型自動偵測（365/252）維持用全量資料前 50 筆，不受篩選影響。
- 匯入 `_range_utils` 時使用第 3 節的雙軌寫法。
- **檔名變更**：`{symbol}_Term_Structure_Analysis.md` → `{symbol}_Term_Structure_{range}.md`

### 5.2 `scripts/test_strategy_dip_buy.py`

**變更**：新增 `--start` / `--end`（`--days` 已存在，語義不變）。

- `past_return = df['close'].pct_change(periods=hold_days)` 維持在全量資料計算（現有程式碼順序已正確，只需在 `dropna` 後套用進場日篩選）。
- 函數簽名改為：
  ```python
  def run_dip_buy_test(symbol, csv_path, hold_days=None, threshold_pct=-0.1115,
                       start_date=None, end_date=None):
  ```
  （`hold_days` 預設值由 365 改為 None，讓「自動偵測資產類型」成為一致的預設行為；`main()` 本來就明確傳入 `args.days`，行為不變。）
- 報表開頭換成第 4 節的統一四行（原有的「分析進場區間」一行由「指定進場區間 + 實際有效進場範圍」取代）。
- 匯入 `_range_utils` 時使用第 3 節的雙軌寫法。
- **檔名變更**：`{symbol}_Strategy_DipBuy_{threshold}.md` → `{symbol}_Strategy_DipBuy_{threshold}_{range}.md`

### 5.3 `scripts/generate_asset_report.py`

**變更**：三參數已存在，本次做語義對齊與錯誤處理修正。

- 日期解析與檔名區間字串改用 `_range_utils` 共用函式（使用第 3 節的雙軌匯入寫法）。
  - 修正現有小瑕疵：只給 `--end` 不給 `--start` 時，目前檔名會錯誤地仍為 `All_Time`；改由 `range_suffix()` 統一處理。
- **修 Bug（錯誤路徑崩潰）**：`analyze_crypto()` 在「檔案不存在 / 區間無資料」時回傳純字串，但 `main()` 用 `report_content, hold_days = analyze_crypto(...)` 解包，會直接拋 `ValueError` 崩潰。改為：錯誤時回傳 `(None, hold_days)` 並附上錯誤訊息，`main()` 檢查後印出訊息並結束，**不寫出報告檔**。
- 報表開頭換成第 4 節的統一四行（目前已有「完整數據範圍」與「分析進場區間」，調整命名並補上指定區間）。
- **檔名不變**：`{symbol}_{days}d_Hold_{range}.md`（已符合統一規範）。

### 5.4 檔名規範對照表

| 腳本 | 現在（全歷史） | 改後（全歷史） | 改後（指定區間） |
| :--- | :--- | :--- | :--- |
| generate_asset_report | `BTC_365d_Hold_All_Time.md` | 不變 | `BTC_365d_Hold_2021-09-10_to_Now.md`（不變） |
| analyze_term_structure | `BTC_Term_Structure_Analysis.md` | `BTC_Term_Structure_All_Time.md` | `BTC_Term_Structure_2021-09-10_to_Now.md` |
| test_strategy_dip_buy | `BTC_Strategy_DipBuy_11p15pct_Drop.md` | `BTC_Strategy_DipBuy_11p15pct_Drop_All_Time.md` | `BTC_Strategy_DipBuy_11p15pct_Drop_2021-09-10_to_Now.md` |

**注意**：後兩隻腳本的預設檔名會改變，舊報告檔不會被覆蓋、也不會自動刪除，會並存於 `research/reports/`。使用者需知悉這是有意的行為變更。

---

## 6. 測試計畫（`scripts/test_scripts.py` 新增 `TestScriptDateFiltering`）

沿用現有模式：合成日線資料寫入 `tempfile` 暫存 CSV，直接呼叫函數、斷言回傳的報表字串。測試檔以 `from scripts.xxx import ...` 匯入各腳本（此時第 3 節雙軌匯入的相對分支生效）。新增測試類別，案例如下：

1. **進場日篩選精確生效**：2000 天遞增資料，`--start` / `--end` 各取中間一段，斷言報表中「實際有效進場範圍」的首日 = 指定 start。
2. **回溯緩衝保留**：`run_dip_buy_test` 指定 start 後，start 當天的 `past_return` 仍引用 start 之前的歷史（即指定區間第一年內就能產生訊號，不會空白一年）。
3. **Entry-Window 語義**：end 設在資料中段，驗證接近 end 的進場（其結果落在 end 之後）仍被納入統計（樣本數符合預期）。
4. **右截尾捨棄**：不指定 end 時，資料最後 `days` 天的進場被捨棄，報表的「實際有效進場範圍」末日 = 資料末日 − days。
5. **無效輸入不崩潰**：`start > end`、非 YYYY-MM-DD 格式（如 `2021/09/10`）、區間無資料三種情況，皆回傳/印出中文錯誤訊息且不寫出報告檔。
6. **期限結構短區間**：只給 2 年區間時，3 年、5 年等長天期被跳過，程式正常結束且報表只含有效天期。

**清理**：`tearDown` 的報告檔清理 glob 需涵蓋新檔名（`TEST_Term_Structure_*.md`、`TEST_Strategy_DipBuy_*.md`、`TEST_*d_Hold_*.md`）。

---

## 7. 文件更新（`scripts/README.md`)

- 三隻腳本的說明各加上統一參數表（`--start` / `--end` / `--days`）。
- 新增「建議工作流程」小節：期限結構梯子 → 單期細節的兩步驟範例。
- 用一小段白話說明語義：日期篩的是進場日、結果可能用到 end 之後的資料、末端不足一個持有期的進場會被捨棄。

---

## 8. 驗收標準（Acceptance Criteria）

1. `python -m unittest scripts.test_scripts` 全數通過（目前基線為 9 tests OK；此指令同時驗證測試匯入路徑正常）。
2. **向後相容**：不帶日期參數跑三隻腳本，統計數字與改版前完全一致（僅 5.4 節所述檔名變更）。
3. 以下範例指令在真實資料（`data/raw/`）上實際執行成功（同時驗證直接執行路徑正常），且報表開頭四行資訊正確：
   ```bash
   python scripts/analyze_term_structure.py --symbol BNB --start 2021-09-10
   python scripts/test_strategy_dip_buy.py --symbol BNB --threshold -0.1578 --start 2021-09-10
   python scripts/generate_asset_report.py --symbol BTC --start 2021-11-10 --end 2022-11-21 --days 365
   ```
4. 錯誤輸入（`--start 2025-01-01 --end 2024-01-01`、`--start 2021/09/10`）印出中文錯誤訊息、不寫檔、不拋例外。
5. `ruff check scripts` 通過（行寬 ≤ 100）。

---

## 9. 明確不做（Out of Scope）

- 不更動核心統計算法（每天進場模擬、勝率/中位數/四分位/凱利公式維持原樣）。
- 不做圖表或互動式儀表板。
- 不新增第三方依賴（僅用既有的 pandas）。
- 不改 `scripts/data/fetch_binance.py` 等資料取得腳本。
- 期限結構的梯子維持固定 1Q–20Q，不開放自訂天期列表（若未來有需要再議）。

---

## 10. 給審閱者的重點問題（Open Questions）

1. **Entry-Window 語義**（結果允許超出 end）是否認同？若改為「整筆交易須落在區間內」，統計結果會不同，且短區間會損失大量樣本。
2. 後兩隻腳本的**預設檔名變更**是否可接受（舊檔並存不覆蓋）？
3. 抽出共用模組 `_range_utils.py`  vs. 三隻腳本各自複製十幾行代碼——本計畫選前者以避免語義漂移，審閱者若有維護性疑慮可提出。
4. 期限結構維持固定梯子（不自訂 `--days` 列表）是否符合使用需求？

---

## 11. 審閱紀錄（2026-09-10）

### 第一輪審閱
本計畫經第二方 AI 審閱，結論：**以本文件為準進行實作**。審閱意見與處理：

- 第 10 節問題 1（Entry-Window 語義）：**認同**，維持原設計——回答的是「在某段期間進場後，持有 N 天會如何」，只要報告清楚揭露即可。
- 第 10 節問題 2（預設檔名變更）：審閱未反對，視為接受；實作時依 5.4 節執行。
- 第 10 節問題 3（共用模組）：**認同**——日期解析與檔名規則集中於 `_range_utils.py`，避免三腳本日後出現不一致。
- 第 10 節問題 4（固定梯子）：**認同**——期限結構以固定 1Q–20Q 梯子取代 `--days` 是合理設計。
- 採納的兩點修正（已反映於本文件）：
  1. 標題由「統一三參數模型」改為「統一日期與持有期模型」，避免誤以為期限結構腳本也有 `--days`。
  2. 日期格式由「`pd.to_datetime` 通用解析」改為「嚴格 YYYY-MM-DD 驗證」（見 2.4 與第 3 節），與文件說明一致並避免月日歧義。

### 第二輪審閱
- 確認兩項修正皆已正確反映，計畫的範圍、語義、錯誤處理與測試標準已足夠清楚，**同意定稿**。
- 唯一補充（已納入第 3 節「匯入方式」）：`scripts/` 無 `__init__.py`（已核實），日常使用為直接執行 `python scripts/xxx.py`，故三隻腳本必須以雙軌匯入（try 相對匯入 / except 直接匯入）引入 `_range_utils`，以同時支援直接執行與測試匯入。
- 舊提案文件 `2026-09-10_custom_date_range_support.md` 依使用者指示刪除，避免與本文件並存造成誤解。
