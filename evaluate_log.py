#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
candidates_log.csv (알림 후보 기록) 의 사후 성적표.
신호일 D -> 진입 = D+1 일봉 시가. 이후 k=0..4일째 성적을 '같은 날짜의 전체 유동 코인 평균(무작위)'과 비교한다.
출력: candidate_report.txt  (오른 코인뿐 아니라 빠진 코인도 모두 포함)
실행: python evaluate_log.py [--data-dir upbit_daily] [--log candidates_log.csv]
"""
import argparse, os
import numpy as np
import pandas as pd

COST = 0.003


def load(data_dir, min_value=1e8):
    out = {}
    for fn in os.listdir(data_dir):
        if fn.endswith(".csv"):
            r = pd.read_csv(os.path.join(data_dir, fn), parse_dates=["date"]).sort_values("date").set_index("date")
            out[fn[:-4]] = r
    return out


def day_stats(r, d0, k):
    """진입일 d0 의 시가 기준, d0+k 일 봉의 지표. 없으면 None"""
    dk = d0 + pd.Timedelta(days=k)
    if d0 not in r.index or dk not in r.index:
        return None
    o0 = r.loc[d0, "open"]
    row = r.loc[dk]
    if not o0 or not row["open"]:
        return None
    return dict(oc=row["close"] / row["open"] - 1, hi=row["high"] / row["open"] - 1, lo=row["low"] / row["open"] - 1,
                cum=row["close"] / o0 - 1)


def rule_10_5(r, d0, H=7, tp=0.10, sl=0.05):
    """진입일 시가 매수, +10/-5, 7일, 같은 날 둘 다면 손절 먼저(보수적). 아직 7일이 안 지났으면 None"""
    days = [d0 + pd.Timedelta(days=i) for i in range(H)]
    if any(d not in r.index for d in days) or d0 not in r.index:
        return None
    e = r.loc[d0, "open"] * 1.001
    for d in days:
        row = r.loc[d]
        if row["low"] <= e * (1 - sl):
            return -sl - COST
        if row["high"] >= e * (1 + tp):
            return tp - COST
    return r.loc[days[-1], "close"] * 0.999 / e - 1 - COST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="upbit_daily")
    ap.add_argument("--log", default="candidates_log.csv")
    ap.add_argument("--out", default="candidate_report.txt")
    ap.add_argument("--notify", action="store_true", help="최근 신호일 개별 현황과 교집합 요약을 디스코드로 전송")
    a = ap.parse_args()
    log = pd.read_csv(a.log, dtype={"signal_date": str})
    data = load(a.data_dir)
    last = max(r.index.max() for r in data.values())
    lines = []
    P = lines.append
    dates = sorted(log["signal_date"].unique())
    P(f"기록된 신호일 {len(dates)}개: {dates[0]} ~ {dates[-1]} | 데이터 마지막 일봉 {last.date()}")
    P("진입 = 신호일 다음 일봉 시가. 일봉 날짜는 09:00 KST 시작. '무작위' = 같은 날짜들의 거래대금 1억 이상 전체 코인 평균.")
    P("진행 중인 오늘 봉은 제외하고, 확정된 일봉만 사용합니다.\n")
    groups = [("3개 교집합", log["overlap"] == 1), ("1번 close", log["l1"] == 1), ("2번 high", log["l2"] == 1),
              ("3번 8지표", log["l3"] == 1), ("후보 전체", log["l1"] >= 0)]
    for name, mask in groups:
        sub = log[mask]
        P(f"################ {name} (신호일-코인 {len(sub)}건) ################")
        P("k일째 | 건수 | 시가→종가 평균 | 고가≥+5% | 저가≤-5% | 진입가 대비 누적(종가) | 무작위 시가→종가 | 무작위 고가≥+5% | 무작위 저가≤-5% | 무작위 누적")
        for k in range(5):
            v, bs = [], []
            for sd in sub["signal_date"].unique():
                d0 = pd.Timestamp(sd) + pd.Timedelta(days=1)
                if d0 + pd.Timedelta(days=k) > last:
                    continue
                for m in sub[sub["signal_date"] == sd]["market"]:
                    if m in data:
                        s = day_stats(data[m], d0, k)
                        if s:
                            v.append(s)
                # 무작위: 같은 진입일의 모든 유동 코인
                for m, r in data.items():
                    dk = d0 + pd.Timedelta(days=k)
                    if d0 in r.index and dk in r.index and r.loc[d0, "value"] >= 1e8:
                        s = day_stats(r, d0, k)
                        if s:
                            bs.append(s)
            if not v:
                P(f"  {k} | 아직 없음")
                continue
            f = lambda xs, key: np.mean([x[key] for x in xs])
            g = lambda xs, key, cmp: np.mean([cmp(x[key]) for x in xs])
            P(f"  {k} | {len(v)} | {f(v,'oc'):+.2%} | {g(v,'hi',lambda z:z>=0.05):.0%} | {g(v,'lo',lambda z:z<=-0.05):.0%} | "
              f"{f(v,'cum'):+.2%} | {f(bs,'oc'):+.2%} | {g(bs,'hi',lambda z:z>=0.05):.0%} | {g(bs,'lo',lambda z:z<=-0.05):.0%} | {f(bs,'cum'):+.2%}")
        res, bres = [], []
        for sd in sub["signal_date"].unique():
            d0 = pd.Timestamp(sd) + pd.Timedelta(days=1)
            for m in sub[sub["signal_date"] == sd]["market"]:
                if m in data:
                    x = rule_10_5(data[m], d0)
                    if x is not None:
                        res.append(x)
        if res:
            P(f"  규칙 +10%/-5%/7일 (비용 포함, 보수적): {len(res)}건 평균 {np.mean(res):+.2%}, 승 {np.mean([x>0 for x in res]):.0%}")
        else:
            P("  규칙 +10%/-5%/7일: 7일 지난 건 아직 없음")
        P("")
    # 개별 현황 (가장 최근 신호일)
    sd = dates[-1]
    d0 = pd.Timestamp(sd) + pd.Timedelta(days=1)
    P(f"################ 최근 신호일 {sd} 개별 현황 (진입가 = {d0.date()} 시가) ################")
    for _, row in log[log["signal_date"] == sd].iterrows():
        r = data.get(row["market"])
        if r is None or d0 not in r.index:
            continue
        sub_r = r[r.index >= d0]
        o0 = r.loc[d0, "open"]
        tag = "교집합" if row["overlap"] else ("/".join(n for n, c in [("1", row["l1"]), ("2", row["l2"]), ("3", row["l3"])] if c))
        P(f"  {row['market'][4:]:8s} [{tag}] 최근 종가 {sub_r['close'].iloc[-1]/o0-1:+.1%} | 최고 {sub_r['high'].max()/o0-1:+.1%} | 최저 {sub_r['low'].min()/o0-1:+.1%}")
    txt = "\n".join(lines)
    open(a.out, "w", encoding="utf-8").write(txt)
    print(txt)
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if a.notify and url:
        import requests
        i = txt.find("################ 최근 신호일")
        msg = "[후보 사후 성적표]\n" + txt[:txt.find("\n\n")].split("\n")[0] + "\n" + txt[i:]
        requests.post(url, json={"content": msg[:1900]}, timeout=15)


if __name__ == "__main__":
    main()
