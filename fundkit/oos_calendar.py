"""Source-backed, narrowly scoped historical calendar corrections.

Do not treat unexplained missing quotes as suspension or invent trading prices.
"""
import pandas as pd

CALENDAR_SOURCE = "https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20251222_10802510.shtml"
SUSPENSION_SOURCE = "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2025-05-23/508068_20250523_4CJP.pdf"
VERIFIED_SUSPENSIONS = {"508068": ("2025-05-23",)}


def complete_calendar(calendar, asof):
    """Append only the known Nov 10-20 2026 exchange workdays.

    The official published annual holiday schedule has no closures in
    this interval. The patch does not extrapolate beyond Nov 20 and is
    not evidence of fund-specific trading or NAV publication.
    """
    days=pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
    if not len(days):
        raise ValueError("empty market calendar")
    start=pd.Timestamp("2026-11-10")
    end=min(pd.Timestamp(asof),pd.Timestamp("2026-11-20"))
    if end>=start and days.max()>=pd.Timestamp("2026-11-09"):
        days=days.union(pd.bdate_range(start,end)).sort_values()
    return days
