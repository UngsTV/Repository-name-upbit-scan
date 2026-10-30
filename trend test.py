#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제미니 '8대 지표 동시 충족' 추세추종 규칙 재현 검증 (trend_test)

제미니 주장: 이격도 100~118%, 일목 전환선>기준선, 아론(25) 오실레이터>0, 엘더레이 베어파워>0,
프라이스채널(20) 중상단, OBV>10일평균, 소나>0 을 모두 만족하면 (익절 +15%, 손절 -5%, 최대 7일)
불장에서 승률 37%, 평균 +1.35%, 손익비 1.45.

여기서는 같은 규칙을 다음 기준으로 다시 계산합니다.
  - 진입은 신호 다음날 시가 (신호일 종가를 보고 그 가격에 살 수는 없으므로), 수수료+슬리피지 반영
  - 같은 장세의 '신호 없는 전체 코인'을 기준선으로 비교 (장세 효과 vs 지표 효과 분리)
  - 같은 코인에서 7일 안에 겹치는 신호는 하나로 줄인 버전 (겹치면 표본이 부풀려짐)
  - 학습/검증/테스트, 날짜 단위 부트스트랩 신뢰구간, 연도별 쏠림
  - 장세는 과거 값만으로 계산한 거래대금 순위 (미래 정보 금지)
가정: 소나(Sonar)는 정의가 흔들려서 'EMA9(종가 - 25일 전 종가) > 0' 으로 구현했습니다. 엘더레이는 저가 - EMA13 > 0 입니다.
일봉으로는 손절/익절 중 뭐가 먼저인지 모르므로 손절 먼저(비관) / 익절 먼저(낙관) 두 범위를 냅니다.

실행: python trend_test.py   (결과는 results_trend.txt 에도 저장)
"""
import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H = 7  # 최대 보유일


def build(data_dir, min_value):
    parts = []
    for fn in sorted(x for x in os.listdir(data_dir) if x.endswith(".csv")):
        raw = pd.read_csv(os.path.join(data_dir, fn), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        if len(raw) < 150:
            continue
        c, o, h, l, v, val = (raw[k] for k in ("close", "open", "high", "low", "volume", "value"))
        f = pd.DataFrame({"date": raw["date"], "market": fn[:-4]})
        f["c"], f["value"] = c, val
        ma20 = c.rolling(20).mean()
        f["disp"] = c / ma20 * 100
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
        f["ichi"] = tenkan > kijun
        up = h.rolling(26).apply(lambda x: 25 - (len(x) - 1 - np.argmax(x)), raw=True)
        dn = l.rolling(26).apply(lambda x: 25 - (len(x) - 1 - np.argmin(x)), raw=True)
        f["aroon"] = (up - dn) * 4  # (up-dn)/25*100
        f["bear"] = l - c.ewm(span=13, adjust=False).mean()
        hh20, ll20 = h.rolling(20).max(), l.rolling(20).min()
        f["pc_pos"] = (c - ll20) / (hh20 - ll20).replace(0, np.nan)
        obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
        f["obv_up"] = obv > obv.rolling(10).mean()
        f["sonar"] = (c - c.shift(25)).ewm(span=9, adjust=False).mean()
        f["ret1"] = c.pct_change()
        for k in range(1, H + 1):
            f[f"o{k}"], f[f"h{k}"], f[f"l{k}"], f[f"c{k}"] = o.shift(-k), h.shift(-k), l.shift(-k), c.shift(-k)
        # 7일 안에 하루라도 종가 기준 +30% 이상 오른 날이 있었는지 (제미니식 적중률)
        s = f["ret1"].rolling(H, min_periods=H).max().shift(-H)  # t+1 ~ t+7 의 일일 상승률 중 최대
        f["surge7"] = np.where(s.notna(), s >= 0.30, np.nan)
        parts.append(f)
    df = pd.concat(parts, ignore_index=True)
    df["liquid"] = df["value"] >= min_value
    return df


def regime(df):
    mv = df.groupby("date")["value"].sum().sort_index().rolling(60).mean()
    pct = mv.expanding(250).apply(lambda x: (x <= x[-1]).mean(), raw=True)
    lab = pd.Series(np.where(pct > 2 / 3, "불장", np.where(pct <= 1 / 3, "하락장", "횡보장")), index=mv.index)
    lab[pct.isna()] = "미정"
    return lab.rename("regime").reset_index()


def simulate(d, order, slip, fee=0.0005, stop=0.05, tp=0.15):
    E = d["o1"].values
    sl, tgt = E * (1 - stop), E * (1 + tp)
    ex = np.full(len(d), np.nan)
    done = np.zeros(len(d), bool)
    for k in range(1, H + 1):
        o, h, l, c = (d[f"{x}{k}"].values for x in "ohlc")
        hit_s, hit_t = l <= sl, h >= tgt
        if k == 1:
            s_px, t_px = sl, tgt
        else:
            s_px, t_px = np.minimum(sl, o), np.maximum(tgt, o)
        if order == "pess":
            ev = np.where(hit_s, s_px, np.where(hit_t, t_px, np.nan))
        else:
            ev = np.where(hit_t, t_px, np.where(hit_s, s_px, np.nan))
        if k == H:
            ev = np.where(np.isnan(ev), c, ev)
        new = ~done & ~np.isnan(ev) & ~np.isnan(o)
        ex[new] = ev[new]
        done |= new
    cost = E * (1 + slip) * (1 + fee)
    return ex * (1 - slip) * (1 - fee) / cost - 1


def stat(net):
    if len(net) == 0:
        return "-"
    w, l = net[net > 0], net[net <= 0]
    pf = w.sum() / -l.sum() if len(l) and l.sum() < 0 else np.inf
    return f"{net.mean():+.2%}|승{(net > 0).mean():.0%}|PF{pf:.2f}|n{len(net)}"


def declutter(d):
    keep, last = [], {}
    for idx, m, dt in zip(d.index, d["market"].values, d["date"].values):
        p = last.get(m)
        if p is None or (dt - p) / np.timedelta64(1, "D") >= H:
            keep.append(idx)
            last[m] = dt
    return d.loc[keep]


def run(args):
    from discover_v2 import boot_ci
    df = build(args.data_dir, args.min_value)
    df = df.merge(regime(df), on="date", how="left")
    need = ["disp", "aroon", "bear", "pc_pos", "sonar", "o1", "h1", "l1", "c1"]
    base = df[df["liquid"] & df[need].notna().all(axis=1) & df["c" + str(H)].notna() & (df["regime"] != "미정")].copy()
    dates = np.sort(base["date"].unique())
    c1, c2 = dates[int(len(dates) * 0.6)], dates[int(len(dates) * 0.8)]
    base["period"] = np.where(base["date"] < c1, "학습", np.where(base["date"] < c2, "검증", "테스트"))
    conds = {
        "이격도 100~118": base["disp"].between(100, 118),
        "전환선>기준선": base["ichi"],
        "아론>0": base["aroon"] > 0,
        "베어파워>0": base["bear"] > 0,
        "채널위치>=0.5": base["pc_pos"] >= 0.5,
        "OBV>10일평균": base["obv_up"],
        "소나>0": base["sonar"] > 0,
    }
    print(f"코인-일 {len(base):,}개 / 학습 ~{str(c1)[:10]} 전, 검증 ~{str(c2)[:10]} 전, 테스트 이후\n")
    m = pd.Series(True, index=base.index)
    steps = []
    for k, v in conds.items():
        m &= v
        steps.append(f"{k} {int(m.sum()):,}")
    print("=== [1] 조건을 하나씩 붙일 때 남는 코인-일 === \n" + " -> ".join(steps))
    sig = base[m]
    print(f"\n신호 {len(sig):,}개 (제미니는 19,947건이라고 함). 서로 겹치지 않게 줄이면 {len(declutter(sig)):,}개\n")

    print("=== [2] 신호 vs '신호 없는 전체 코인' 기준선 (익절 +15% / 손절 -5% / 최대 7일, 다음날 시가 진입) ===")
    print("칸: 평균 순수익|승률|손익비|건수. 신호가 기준선보다 확실히 좋아야 지표에 의미가 있습니다.")
    sig_d = declutter(sig)
    for order, nm in (("pess", "손절 먼저(비관)"), ("opt", "익절 먼저(낙관)")):
        print(f"\n[{nm}]")
        rows = []
        net_b = pd.Series(simulate(base, order, args.slip), index=base.index)
        net_s = pd.Series(simulate(sig, order, args.slip), index=sig.index)
        net_d = pd.Series(simulate(sig_d, order, args.slip), index=sig_d.index)
        for rg in ("불장", "횡보장", "하락장", "전체"):
            for lab_, dd, nn in (("기준선(전체 코인)", base, net_b), ("신호(겹침 포함)", sig, net_s),
                                ("신호(겹침 제거)", sig_d, net_d)):
                k = (dd["regime"] == rg) if rg != "전체" else pd.Series(True, index=dd.index)
                r = {"장세": rg, "대상": lab_}
                for per in ("학습", "검증", "테스트"):
                    kk = k & (dd["period"] == per)
                    r[per] = stat(nn[kk].values)
                rows.append(r)
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== [3] 겹침 제거 신호의 95% 신뢰구간 (날짜 단위 부트스트랩, 비관 기준) ===")
    net_d = pd.Series(simulate(sig_d, "pess", args.slip), index=sig_d.index)
    for rg in ("불장", "횡보장", "하락장", "전체"):
        k = (sig_d["regime"] == rg) if rg != "전체" else pd.Series(True, index=sig_d.index)
        x, dt = net_d[k].values, sig_d.loc[k, "date"].values
        if len(x) >= 30:
            lo, hi = boot_ci(dt, x)
            print(f"{rg}: 평균 {x.mean():+.2%}, 95% {lo:+.1%}~{hi:+.1%} (n={len(x)})")
        else:
            print(f"{rg}: 표본 부족 (n={len(x)})")

    print("\n=== [4] 연도별 신호 수와 평균 (한 시기에 쏠렸는지, 겹침 제거·비관) ===")
    yy = sig_d.assign(net=net_d, year=sig_d["date"].dt.year).groupby("year")["net"].agg(["size", "mean"])
    yy["mean"] = yy["mean"].map(lambda v: f"{v:+.2%}")
    print(yy.rename(columns={"size": "신호수", "mean": "평균"}).to_string())

    print("\n=== [5] 제미니식 적중률: 신호 후 7일 안에 하루 +30% 이상 상승(종가 기준)이 나왔나 ===")
    bb = base[base["surge7"].notna()]
    ss = sig[sig["surge7"].notna()]
    br, sr = bb["surge7"].astype(float).mean(), ss["surge7"].astype(float).mean()
    print(f"전체 코인 기준선 {br:.1%} / 신호 {sr:.1%} -> {sr / br:.2f}배 (제미니: 11.05%, '기준선 3~4배' 주장)")
    for rg in ("불장", "횡보장", "하락장"):
        b, s = bb[bb["regime"] == rg]["surge7"].astype(float).mean(), ss[ss["regime"] == rg]["surge7"].astype(float).mean()
        print(f"  {rg}: 기준선 {b:.1%} / 신호 {s:.1%}")

    print("\n=== [6] 오늘 신호 (관찰용) ===")
    last = df["date"].max()
    tm = df[(df["date"] == last) & df["liquid"]].copy()
    tm = tm[tm[["disp", "aroon", "bear", "pc_pos", "sonar"]].notna().all(axis=1)]
    k = (tm["disp"].between(100, 118) & tm["ichi"] & (tm["aroon"] > 0) & (tm["bear"] > 0)
         & (tm["pc_pos"] >= 0.5) & tm["obv_up"] & (tm["sonar"] > 0))
    t = tm[k]
    print(f"기준 신호봉 {str(last)[:10]}, 오늘 장세: {tm['regime'].iloc[0] if len(tm) else '-'}")
    if len(t):
        print(t.assign(**{"거래대금(억)": (t["value"] / 1e8).round(1)})[
            ["market", "c", "disp", "aroon", "pc_pos", "거래대금(억)"]].sort_values("거래대금(억)", ascending=False).head(20).round(2).to_string(index=False))
        print(f"... 총 {len(t)}개")
    else:
        print("오늘 걸리는 코인이 없습니다.")
    print("\n※ 연구용입니다. 거래정지·유의 지정·상장폐지 예정·입금량 경고, 호가 두께, 상장폐지된 코인(생존편향)은 반영되지 않았습니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="upbit_daily")
    ap.add_argument("--min-value", type=float, default=1e8)
    ap.add_argument("--slip", type=float, default=0.001, help="편도 슬리피지(기본 0.001 = 0.1퍼센트)")
    ap.add_argument("--out", default="results_trend.txt")
    args = ap.parse_args()

    class Tee:
        def __init__(self, *fs):
            self.fs = fs

        def write(self, x):
            for f in self.fs:
                f.write(x)

        def flush(self):
            for f in self.fs:
                f.flush()

    with open(args.out, "w", encoding="utf-8-sig") as fh:
        real = sys.stdout
        sys.stdout = Tee(real, fh)
        try:
            run(args)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            print("\n!!! 오류가 났습니다. 이 파일 전체를 그대로 보내주세요")
        finally:
            sys.stdout = real
    print(f"\n화면 내용 전체가 {args.out} 에도 저장됐습니다.")


if __name__ == "__main__":
    main()
