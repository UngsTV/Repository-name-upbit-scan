# -*- coding: utf-8 -*-
"""교집합 안 패턴 A / A1 / B / C 판정 (일봉 DataFrame: date, open, high, low, close, volume, value). pandas만 사용.
A : 이격도20 > 27.05% 이고 5일 수익률 > +47.64%                       (이미 급등 중, 신호 다음날 시가 진입 기준)
A1: A 이고 20일 일수익률 표준편차 <= 11.67%
B : 이격도20 > 27.05% 이고 5일 <= +47.64% 이고 저점(90일) 대비 <= +128.3%   (급등 후 완만한 상승)
C : 볼린저폭/120일 중앙값 > 2.119 이고 저점(90일) 대비 <= +173.4% 이고 OBV20 > 0.5592   (신호+4일 시가 진입 기준)
숫자는 학습 구간(2023-04~2025-03) 결정트리에서 나온 값이고 검증(2025-04~)에서 한 번 확인함."""
import numpy as np, pandas as pd

def tag_frame(raw):
    c, h, l, v = (raw[k].astype(float) for k in ("close", "high", "low", "volume"))
    ma20 = c.rolling(20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    bw = 4 * sd / ma20
    f = pd.DataFrame({"date": raw["date"]})
    f["disp20"] = c / ma20 - 1
    f["ret5"] = c / c.shift(5) - 1
    f["std20"] = c.pct_change().rolling(20).std()
    f["from_low90"] = c / l.rolling(90, min_periods=40).min() - 1
    f["bw_rel"] = bw / bw.rolling(120, min_periods=60).median()
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv20"] = (obv - obv.shift(20)) / v.rolling(20).sum().replace(0, np.nan)
    f["A"] = (f.disp20 > .2705) & (f.ret5 > .4764)
    f["A1"] = f["A"] & (f.std20 <= .1167)
    f["B"] = (f.disp20 > .2705) & (f.ret5 <= .4764) & (f.from_low90 <= 1.283)
    f["C"] = (f.bw_rel > 2.119) & (f.from_low90 <= 1.734) & (f.obv20 > .5592)
    return f
