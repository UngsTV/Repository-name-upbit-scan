#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
screen_alert.py - 매일 아침 두 가지 스크리닝을 디스코드로 보냅니다 (업비트 KRW 일봉, 마감된 봉 기준)

스크리닝 1 (학습·검증에서 플러스였던 규칙; positive_rules.py 결과)
  1-A 급락 반등: 2일 수익률 ≤ -17% 이고 변동폭 ≥ 7% 이고 그날 상승 종목 비율 < 30%  -> 다음날 시가 매수, 그날 종가 매도 (1일)
      (189개 조합 중 전 조건 통과는 이것 하나. 단 하루에 1개만 고르면 학습 -0.69% / 검증 +0.54%라 한 번에 여러 개 살 때만 플러스)
  1-B 제미니 8지표 8/8, 1-C 추세 7조건 (7일 보유: 학습·검증 비관 평균 플러스, 합산 99% 하한은 0 미만 = 시장 상승 효과 포함)
스크리닝 2 (크게 움직이기 전 변동성 신호)
  변동폭 ≥ 10% 이고 ATR14 ≥ 10% 이고 볼린저폭(20, 2σ) ≥ 0.40  (+ 이격도20 ≥ 110% 표시)
  -> 폭등(익일 시가 대비 고가 +10%) 확률을 높이는 신호이지 방향 신호가 아님. 학습·검증에서 비용 후 플러스 규칙은 아니었음.

환경변수 DISCORD_WEBHOOK_URL 이 없으면 화면에만 출력. --data-dir 를 주면 API 대신 폴더의 CSV(date,open,high,low,close,volume,value)를 읽음.
"""
import argparse, os, sys, time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from pattern_tag import tag_frame

KST = timezone(timedelta(hours=9))
BASE = "https://api.upbit.com/v1"
MIN_VALUE = 1e8


def get_json(url, params=None, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1 + i)
    raise RuntimeError(f"요청 실패 {url} {params}")


def fetch_all(count=200):
    info = get_json(f"{BASE}/market/all", {"isDetails": "false"})
    markets = sorted(m["market"] for m in info if m["market"].startswith("KRW-"))
    out = {}
    for m in markets:
        rows = get_json(f"{BASE}/candles/days", {"market": m, "count": count})
        time.sleep(0.12)
        if not rows:
            continue
        df = pd.DataFrame(rows).rename(columns={"opening_price": "open", "high_price": "high", "low_price": "low",
                                                "trade_price": "close", "candle_acc_trade_volume": "volume",
                                                "candle_acc_trade_price": "value"})
        df["date"] = pd.to_datetime(df["candle_date_time_kst"])
        out[m] = df[["date", "open", "high", "low", "close", "volume", "value"]].sort_values("date").reset_index(drop=True)
    return out


def load_dir(d, tail=220):
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".csv"):
            out[fn[:-4]] = pd.read_csv(os.path.join(d, fn), parse_dates=["date"]).sort_values("date").tail(tail).reset_index(drop=True)
    return out


def last_row_features(r):
    """마지막(확정) 봉 기준 지표. 일봉 60개 미만이면 None"""
    if len(r) < 60:
        return None
    c, h, l, v, val = (r[k].astype(float) for k in ("close", "high", "low", "volume", "value"))
    ma20 = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    up = h.rolling(26).apply(lambda x: 25 - (len(x) - 1 - np.argmax(x)), raw=True)
    dn = l.rolling(26).apply(lambda x: 25 - (len(x) - 1 - np.argmin(x)), raw=True)
    hh, ll = h.rolling(20).max(), l.rolling(20).min()
    pc = (c - ll) / (hh - ll).replace(0, np.nan)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    sonar = (c - c.shift(25)).ewm(span=9, adjust=False).mean()
    vavg = v.shift(1).rolling(20).mean()
    i = len(r) - 1
    disp = c.iloc[i] / ma20.iloc[i] * 100
    f = dict(
        close=c.iloc[i], value=val.iloc[i],
        ret1=c.iloc[i] / c.iloc[i - 1] - 1, ret2=c.iloc[i] / c.iloc[i - 2] - 1,
        range=(h.iloc[i] - l.iloc[i]) / c.iloc[i],
        atr14=tr.rolling(14).mean().iloc[i] / c.iloc[i],
        bw=4 * sd.iloc[i] / ma20.iloc[i], disp=disp)
    c1 = bool(val.iloc[i] >= 5e8 or (vavg.iloc[i] > 0 and v.iloc[i] >= 3.5 * vavg.iloc[i]))
    c2 = bool(obv.iloc[i] > obv.rolling(10).mean().iloc[i])
    c3 = bool(100 <= disp <= 118)
    c4 = bool(pc.iloc[i] >= 0.52)
    c5 = bool(tenkan.iloc[i] > kijun.iloc[i])
    c6 = bool((up.iloc[i] - dn.iloc[i]) * 4 > 0)
    c7 = bool((l.iloc[i] - c.ewm(span=13, adjust=False).mean().iloc[i]) > 0)
    c8 = bool(sonar.iloc[i] > 0)
    f["g_n"] = int(sum([c1, c2, c3, c4, c5, c6, c7, c8]))
    f["t7"] = bool(c2 and c3 and (pc.iloc[i] >= 0.5) and c5 and c6 and c7 and c8)
    return f


def send(text, dry):
    print(text + "\n")
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if dry or not url:
        return
    for i in range(0, len(text), 1900):
        requests.post(url, json={"content": text[i:i + 1900]}, timeout=15).raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--log", default="screen_log.csv")
    ap.add_argument("--overlap-log", default="candidates_log.csv", help="overlap_notify.py가 쌓는 후보 기록(교집합 표시용)")
    a = ap.parse_args()
    today = pd.Timestamp(datetime.now(KST).date())
    data = load_dir(a.data_dir) if a.data_dir else fetch_all()
    rows, day = [], None
    for m, r in data.items():
        r = r[r["date"] < today]   # 오늘(진행 중) 봉 제외
        if len(r) < 60:
            continue
        f = last_row_features(r)
        if f is None:
            continue
        f["market"] = m
        f["date"] = r["date"].iloc[-1]
        tf = tag_frame(r)
        for q in ("A", "A1", "B", "C"):
            f["p" + q] = bool(tf[q].iloc[-1])
        f["pC4"] = bool(tf["C"].iloc[-4]) if len(tf) >= 4 else False   # 신호+4일 진입용: 4번째 전 봉
        f["sig4"] = str(tf["date"].iloc[-4])[:10] if len(tf) >= 4 else ""
        f["disp20"] = tf["disp20"].iloc[-1]; f["ret5"] = tf["ret5"].iloc[-1]
        rows.append(f)
    F = pd.DataFrame(rows)
    day = F["date"].max()
    F = F[F["date"] == day].copy()
    breadth = (F["ret1"] > 0).mean()
    L = F[F["value"] >= MIN_VALUE].copy()
    nm = lambda m: m.replace("KRW-", "")
    ds = str(day)[:10]
    ov, ov_known = set(), False
    if os.path.exists(a.overlap_log):
        try:
            ol = pd.read_csv(a.overlap_log, dtype={"signal_date": str})
            ol = ol[ol["signal_date"] == ds]
            ov_known = len(ol) > 0
            ov = set(ol[ol["overlap"] == 1]["market"])
        except Exception:
            pass
    tag = lambda m: " ★교집합" if m in ov else ""

    # 스크리닝 1
    r1 = L[(L["ret2"] <= -0.17) & (L["range"] >= 0.07)].sort_values("value", ascending=False)
    lines = [f"[스크리닝 1 | 신호봉 {ds}] 학습·검증 플러스 규칙  (상승 종목 비율 {breadth:.0%})"]
    if breadth < 0.30:
        txt = ", ".join(f"{nm(x.market)}({x.ret2:+.0%}, 변동폭 {x.range:.0%}){tag(x.market)}" for x in r1.itertuples()) or "없음"
        lines.append(f"1-A 급락 반등 (스트레스일 조건 충족): {txt}\n    > 다음날 시가 매수, 그날 종가 매도 (1일)")
    else:
        lines.append(f"1-A 급락 반등: 상승 종목 비율 {breadth:.0%} ≥ 30% 라 조건 미충족 (참고 후보 {', '.join(nm(m) for m in r1.market) or '없음'})")
    g8 = L[L["g_n"] >= 8].sort_values("value", ascending=False)
    t7 = L[L["t7"]].sort_values("value", ascending=False)
    lines.append(f"1-B 제미니 8/8 ({len(g8)}개): " + (", ".join(nm(m) + tag(m) for m in g8.market[:25]) or "없음") + (" …" if len(g8) > 25 else ""))
    lines.append(f"1-C 추세 7조건 ({len(t7)}개): " + (", ".join(nm(m) + tag(m) for m in t7.market[:25]) or "없음") + (" …" if len(t7) > 25 else ""))
    send("\n".join(lines), a.dry)

    # 스크리닝 3 : 교집합 안 패턴 (일봉 연구에서 +30% 도달 확률이 오른 조건; 단독으로 이기는 규칙은 아님)
    L3 = [f"[스크리닝 3 | 신호봉 {ds}] 교집합 패턴 A·B·C"]
    inov = L[L["market"].isin(ov)] if ov_known else L
    if not ov_known:
        L3.append("(교집합 기록 없음 -> 전체 유동 코인 기준으로 표시)")
    for q, nmq, desc in (("pA1", "A1", "이미 급등 중(변동성 낮음)"), ("pA", "A", "이미 급등 중(이격20>27%, 5일>+48%)"), ("pB", "B", "급등 후 완만한 상승")):
        h = inov[inov[q]]
        L3.append(f"{nmq} {desc}: " + (", ".join(f"{nm(x.market)}(이격 {x.disp20:+.0%}, 5일 {x.ret5:+.0%})" for x in h.itertuples()) or "없음") + "   > 신호 다음날 시가 진입 기준")
    sig4 = L[L["pC4"]]
    if ov_known and os.path.exists(a.overlap_log):
        try:
            olx = pd.read_csv(a.overlap_log, dtype={"signal_date": str})
            sig4 = sig4[[(olx[(olx["signal_date"] == d) & (olx["overlap"] == 1)]["market"] == m).any() for m, d in zip(sig4["market"], sig4["sig4"])]]
        except Exception:
            pass
    L3.append("C 볼린저폭 급확대(신호+4일 시가 진입 후보): " + (", ".join(f"{nm(x.market)}(신호일 {x.sig4})" for x in sig4.itertuples()) or "없음"))
    send("\n".join(L3), a.dry)

    # 스크리닝 2
    v6 = L[(L["range"] >= 0.10) & (L["atr14"] >= 0.10) & (L["bw"] >= 0.40)].sort_values("range", ascending=False)
    L2 = [f"[스크리닝 2 | 신호봉 {ds}] 변동성 신호 (변동폭≥10% & ATR14≥10% & 볼린저폭≥0.40): {len(v6)}개"]
    for x in v6.head(25).itertuples():
        mark = " 이격도≥110" if x.disp >= 110 else ""
        both = " [1-B]" if x.g_n >= 8 else ""
        L2.append(f"  {nm(x.market)}{tag(x.market)}  변동폭 {x.range:.0%} ATR {x.atr14:.0%} 볼폭 {x.bw:.2f} 이격 {x.disp:.0f}{mark}{both}")
    if len(v6) > 25:
        L2.append(f"  … 외 {len(v6) - 25}개")
    if ov_known:
        hit = sorted(nm(m) for m in ov if m in set(v6.market) | set(g8.market) | set(t7.market) | set(r1.market))
        miss = sorted(nm(m) for m in ov if m not in set(v6.market) | set(g8.market) | set(t7.market) | set(r1.market))
        L2.append(f"교집합 {len(ov)}개 중 스크리닝에도 걸림: {', '.join(hit) or '없음'} / 안 걸림: {', '.join(miss) or '없음'}")
    else:
        L2.append(f"(교집합 기록 {a.overlap_log} 에 {ds} 신호일이 없어 ★표시 생략)")
    send("\n".join(L2), a.dry)

    log = pd.concat([
        pd.DataFrame({"signal_date": ds, "screen": "1-A", "market": r1.market if breadth < 0.30 else []}),
        pd.DataFrame({"signal_date": ds, "screen": "1-B", "market": g8.market}),
        pd.DataFrame({"signal_date": ds, "screen": "1-C", "market": t7.market}),
        pd.DataFrame({"signal_date": ds, "screen": "2", "market": v6.market}),
        pd.DataFrame({"signal_date": ds, "screen": "3-A1", "market": inov[inov["pA1"]].market}),
        pd.DataFrame({"signal_date": ds, "screen": "3-A", "market": inov[inov["pA"]].market}),
        pd.DataFrame({"signal_date": ds, "screen": "3-B", "market": inov[inov["pB"]].market}),
        pd.DataFrame({"signal_date": ds, "screen": "3-C", "market": sig4.market}),
    ], ignore_index=True)
    if os.path.exists(a.log):
        old = pd.read_csv(a.log, dtype={"signal_date": str})
        log = pd.concat([old[old["signal_date"] != ds], log], ignore_index=True)
    log.to_csv(a.log, index=False)


if __name__ == "__main__":
    main()
