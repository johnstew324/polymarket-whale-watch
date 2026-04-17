import numpy as np
import pandas as pd

from ..config import FIN_PATH


def _get_7day_return(series, date):
    try:
        end = series.asof(date)
        start = series.asof(date - pd.Timedelta(days=7))
        if pd.isna(end) or pd.isna(start) or start == 0:
            return np.nan
        return (end - start) / start
    except Exception:
        return np.nan


def attach_financial_features(positions):
    positions = positions.copy()
    positions['entry_date'] = pd.to_datetime(positions['entry_date'], utc=True, format='mixed')

    fin = pd.read_csv(FIN_PATH, index_col=0, parse_dates=True)
    fin.index = pd.to_datetime(fin.index, utc=True, format='mixed')
    fin = fin.sort_index().ffill()

    for col in fin.columns:
        positions[f'{col}_7d'] = positions['entry_date'].apply(
            lambda d: _get_7day_return(fin[col], d)
        )
        positions[f'{col}_7d'] = positions[f'{col}_7d'].fillna(0.0)

    return positions