#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
매일 디스코드 알림: 1번(discover close) / 2번(discover high) / 3번(Clean V001 8지표 5개 이상) / 겹치는 후보
사전 준비: python backtest_v64.py fetch --days 4000   (upbit_daily 폴더)  /  discover_v2.py 같은 폴더
실행: python overlap_notify.py [--top 15] [--min-count 5] [--dry]
환경변수 DISCORD_WEBHOOK_URL 이 없으면 화면에만 출력합니다.
"""
import argparse, os, shutil, sys, tempfile, traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier

import discover_v2 as d

KST = timezone(timedelta(hours=9))


def send(text, dry):
    print(text + "\n")
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if dry or not url:
        return
    for i in range(0, len(text), 1900):
        requests.post(url, json={"content": text[i:i + 1900]}, timeout=15).raise_for_status()


def closed_copy(src, today):
    """오늘(진행 중) 일봉을 뺀 사본 폴더를 만든다 -> 마지막 줄 = 어제까지 확정된 봉"""
    tmp = tempfile.mkdtemp()
    for fn in os.listdir(src):
        if not fn.endswith(".csv"):
            continue
        r = pd.read_csv(os.path.join(src, fn), parse_dates=["date"])
        r = r[r["date"] < today]
        if len(r):
            r.to_csv(os.path.join(tmp, fn), index=False)
    return tmp


def model_top(df, pct, target, top, args):
    df = d.add_target(df.copy(), pct, target, 0.10, 0.05, 0.0005, 0.001)
    feats = [c for c in df.columns if not c.startswith(("_", "net")) and c not in
             ("market", "date", "y", "has_next", "liquid")]
    ok = df[feats].notna().all(axis=1)
    last = df["date"].max()
    today = df[ok & (df["date"] == last) & df["liquid"]].copy()
    lab = df[ok & df["has_next"] & df["liquid"]]
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                        early_stopping=True, validation_fraction=0.15, random_state=0)
    gb.fit(lab[feats], lab["y"])
    today["p"] = gb.predict_proba(today[feats])[:, 1]
    return today.sort_values("p", ascending=False).head(top)[["market", "p"]], str(last)[:10]


def clean_v001(data_dir):
    rows = []
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith(".csv"):
            continue
        r = pd.read_csv(os.path.join(data_dir, fn), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if len(r) < 60:
            continue
        c, h, l, v, val = r["close"], r["high"], r["low"], r["volume"], r["value"]
        disp = c / c.rolling(20).mean() * 100
        ichi = ((h.rolling(9).max() + l.rolling(9).min()) / 2) > ((h.rolling(26).max() + l.rolling(26).min()) / 2)
        up = h.rolling(26).apply(lambda x: 25 - (len(x) - 1 - np.argmax(x)), raw=True)
        dn = l.rolling(26).apply(lambda x: 25 - (len(x) - 1 - np.argmin(x)), raw=True)
        aroon = (up - dn) * 4
        bear = l - c.ewm(span=13, adjust=False).mean()
        hh, ll = h.rolling(20).max(), l.rolling(20).min()
        pc = (c - ll) / (hh - ll).replace(0, np.nan)
        obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
        sonar = (c - c.shift(25)).ewm(span=9, adjust=False).mean()
        vavg = v.shift(1).rolling(20).mean()
        i = len(r) - 1
        conds = [
            bool(val.iloc[i] >= 5e8 or (vavg.iloc[i] > 0 and v.iloc[i] >= 3.5 * vavg.iloc[i])),
            bool(obv.iloc[i] > obv.rolling(10).mean().iloc[i]),
            bool(100.0 <= disp.iloc[i] <= 118.0),
            bool(pc.iloc[i] >= 0.52),
            bool(ichi.iloc[i]),
            bool(aroon.iloc[i] > 0),
            bool(bear.iloc[i] > 0),
            bool(sonar.iloc[i] > 0),
        ]
        rows.append((fn[:-4], sum(conds), val.iloc[i]))
    return pd.DataFrame(rows, columns=["market", "n", "value"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="upbit_daily")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--min-value", type=float, default=1e8)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    today = pd.Timestamp(datetime.now(KST).date())
    tmp = closed_copy(a.data_dir, today)
    try:
        df = d.load_all(tmp, a.min_value)
        l1, day = model_top(df, 0.05, "open_to_close", a.top, a)
        l2, _ = model_top(df, 0.10, "open_to_high", a.top, a)
        g = clean_v001(tmp)
        l3 = g[g["n"] >= a.min_count].sort_values(["n", "value"], ascending=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    nm = lambda m: m.replace("KRW-", "")
    s1, s2, s3 = set(l1["market"]), set(l2["market"]), set(l3["market"])
    both = s1 & s2 & s3
    L = [f"[업비트 일봉 후보 | 신호봉 {day}]"]
    L.append(f"1번 close(+5%) 상위{a.top}: " + ", ".join(f"{nm(m)} {p:.0%}" for m, p in zip(l1.market, l1.p)))
    L.append(f"2번 high(+10%) 상위{a.top}: " + ", ".join(f"{nm(m)} {p:.0%}" for m, p in zip(l2.market, l2.p)))
    k = 40
    t3 = ", ".join(f"{nm(m)}({n}/8)" for m, n in zip(l3.market[:k], l3.n[:k]))
    L.append(f"3번 8지표 {a.min_count}개 이상 ({len(l3)}개): " + (t3 or "없음") + (" …" if len(l3) > k else ""))
    send("\n".join(L), a.dry)
    ov = sorted(both)
    send("겹치는 후보 >> " + (", ".join(nm(m) for m in ov) if ov else "없음") +
         f"\n(2개 겹침: 1&2 {len(s1 & s2)}개, 1&3 {len(s1 & s3)}개, 2&3 {len(s2 & s3)}개)", a.dry)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        send("overlap_notify 오류\n" + tb[-1500:], False)
        raise
