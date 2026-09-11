"""共用日期區間工具：統一 --start / --end 的解析、驗證與檔名區間字串。

三隻研報腳本（analyze_term_structure / test_strategy_dip_buy / generate_asset_report）
共用此模組以保證語義一致。僅依賴 pandas。
"""

from __future__ import annotations

import re

import pandas as pd

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_date_range(
    start: str | None, end: str | None
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """解析並驗證 --start / --end。

    嚴格要求 YYYY-MM-DD：先以 _DATE_RE.fullmatch 檢查格式，
    再用 pd.to_datetime(..., format='%Y-%m-%d') 確認日期真實存在。
    格式錯誤或 start > end 時 raise ValueError(中文訊息)。
    """
    start_ts = _parse_single(start, '--start')
    end_ts = _parse_single(end, '--end')
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(f'起始日期不可晚於結束日期：--start {start} 晚於 --end {end}')
    return start_ts, end_ts


def _parse_single(value: str | None, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    if not _DATE_RE.fullmatch(value):
        raise ValueError(f'日期格式錯誤：{name} 必須為 YYYY-MM-DD 格式（收到：{value}）')
    try:
        return pd.to_datetime(value, format='%Y-%m-%d')
    except ValueError:
        raise ValueError(f'日期不存在：{name}={value}，請確認為真實存在的日期') from None


def entry_window_mask(
    df: pd.DataFrame, start_ts: pd.Timestamp | None, end_ts: pd.Timestamp | None
) -> pd.Series:
    """產生進場日篩選 mask（True = 進場日在指定區間內）。

    若資料的 datetime 欄是 tz-aware（例如 Binance CSV 帶 +00:00），
    會先把 tz-naive 的 start/end 本地化到相同時區再比較。
    """
    dates = df['datetime']
    tz = getattr(dates.dt, 'tz', None)
    if tz is not None:
        if start_ts is not None:
            start_ts = start_ts.tz_localize(tz)
        if end_ts is not None:
            end_ts = end_ts.tz_localize(tz)
    mask = pd.Series(True, index=df.index)
    if start_ts is not None:
        mask &= dates >= start_ts
    if end_ts is not None:
        mask &= dates <= end_ts
    return mask


def range_suffix(start: str | None, end: str | None) -> str:
    """產出報告檔名用的區間字串：
    都沒給 -> 'All_Time'
    只給 start -> '{start}_to_Now'
    只給 end -> 'Start_to_{end}'
    都給 -> '{start}_to_{end}'
    """
    if start is None and end is None:
        return 'All_Time'
    if end is None:
        return f'{start}_to_Now'
    if start is None:
        return f'Start_to_{end}'
    return f'{start}_to_{end}'
