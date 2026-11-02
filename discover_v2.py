#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
업비트 KRW 일봉 데이터에서 '다음날 큰 상승'의 공통점을 데이터로 찾는 연구용 스크립트 (discover_v2)

준비: backtest_v64.py 의 fetch 로 받은 upbit_daily 폴더가 있어야 합니다.
  pip install pandas numpy scikit-learn
실행:
  python discover_v2.py                      # 기본: 다음날 시가 대비 고가 +10% 이상을 '폭등'으로 정의
  python discover_v2.py --pct 0.15           # 기준을 +15%로
  python discover_v2.py --target open_to_close --pct 0.05   # 다음날 시가에 사서 종가가 +5% 이상 (방향 맞히기)
  python discover_v2.py --target close_to_high   # 신호봉 종가 대비 다음날 고가(갭 포함)

결과는 화면과 results_discover.txt 에 같이 저장됩니다. 이 파일 하나만 보내주세요.

구성
  [1] 단일 지표별 공통점: 값 구간별 폭등 확률 (학습/검증 둘 다 같은 방향인 것만 의미 있음)
  [2] 결정트리(깊이 3): 사람이 읽을 수 있는 규칙
  [3] 그래디언트 부스팅: 하루 상위 N개를 골랐을 때 성적 (검증/테스트)
  [4] 오늘의 상위 후보(연구용)

시간 순서로 학습 60% / 검증 20% / 테스트 20%로 나눕니다. 테스트는 처음 돌릴 때 한 번만 의미가 있습니다.
결과를 보고 모델을 고치면 테스트도 더 이상 깨끗하지 않습니다.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text


# ------------------------------------------------------------------
# 1. 지표 계산 (신호봉까지의 정보만 사용: 미래 정보 금지)
# ------------------------------------------------------------------
def make_features(raw):
    c, h, l, o, v, val = (raw[k] for k in ("close", "high", "low", "open", "volume", "value"))
    f = pd.DataFrame({"date": raw["date"]})
    r = c.pct_change()
    f["ret1"] = r
    for k in (2, 3, 5, 10, 20):
        f[f"ret{k}"] = c / c.shift(k) - 1
    f["range"] = (h - l) / c
    f["body"] = (c - o) / o
    f["upper_wick"] = (h - np.maximum(o, c)) / c
    f["lower_wick"] = (np.minimum(o, c) - l) / c
    uwr = ((h - np.maximum(o, c)) / (h - l).replace(0, np.nan)).fillna(0)  # 윗꼬리가 전체 변동폭에서 차지하는 비율
    f["uwr"] = uwr
    f["uwr3"] = uwr.rolling(3).mean()
    f["close_pos"] = (c - l) / (h - l).replace(0, np.nan)
    f["gap"] = o / c.shift() - 1

    vma5, vma20 = v.rolling(5).mean(), v.rolling(20).mean()
    f["vol_r5"] = v / vma5.replace(0, np.nan)
    f["vol_r20"] = v / vma20.replace(0, np.nan)
    f["vma5_20"] = vma5 / vma20.replace(0, np.nan)
    f["logvalue"] = np.log10(val + 1)
    f["trap"] = ((f["vol_r20"] > 2) & (f["uwr"] > 0.5)).astype(float)  # 거래량 폭증 + 긴 윗꼬리(매물 저항 흔적)
    f["val_r20"] = val / val.rolling(20).mean().replace(0, np.nan)

    f["std20"] = r.rolling(20).std()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr14"] = tr.rolling(14).mean() / c

    ma20 = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    up, lo = ma20 + 2 * sd, ma20 - 2 * sd
    f["pctb"] = (c - lo) / (up - lo).replace(0, np.nan)
    f["bw"] = (up - lo) / ma20
    f["low_touch"] = (l - lo) / c
    for k in (5, 10, 20):
        f[f"disp{k}"] = c / c.rolling(k).mean() - 1

    upd = (r > 0).astype(float)
    psy = upd.rolling(10).mean() * 100
    f["psy10"] = psy
    f["psy_min20"] = psy.rolling(20).min()
    f["up10"] = upd.rolling(10).sum()

    rng = (h - l).replace(0, np.nan)
    mfm = (((c - l) - (h - c)) / rng).fillna(0)
    cmf = (mfm * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    f["cmf20"] = cmf
    f["cmf_chg5"] = cmf - cmf.shift(5)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_chg10"] = (obv - obv.shift(10)) / v.rolling(10).sum().replace(0, np.nan)

    dlt = c.diff()
    gain = dlt.clip(lower=0).rolling(14).mean()
    loss = (-dlt.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    f["dd20"] = c / h.rolling(20).max() - 1
    f["from_low60"] = c / l.rolling(60).min() - 1
    f["days_since_high20"] = h.rolling(20).apply(lambda x: 19 - np.argmax(x), raw=True)
    dn = (r < 0).astype(int)
    grp = (dn != dn.shift()).cumsum()
    f["down_streak"] = (dn.groupby(grp).cumcount() + 1) * dn
    f["dow"] = raw["date"].dt.dayofweek

    # 결과(다음날) - 학습 정답과 거래 계산용. 특징으로 쓰지 않음
    f["_c"], f["_o1"], f["_h1"], f["_l1"], f["_c1"] = c, o.shift(-1), h.shift(-1), l.shift(-1), c.shift(-1)
    f["_c2"] = c.shift(-2)
    f["_value"] = val
    return f


FEATURES = None  # load_all 에서 채움


def load_all(data_dir, min_value):
    files = sorted(x for x in os.listdir(data_dir) if x.endswith(".csv"))
    if not files:
        sys.exit("upbit_daily 폴더에 데이터가 없습니다. 먼저 backtest_v64.py fetch 를 실행하세요.")
    parts, btc = [], None
    for fn in files:
        raw = pd.read_csv(os.path.join(data_dir, fn), parse_dates=["date"]).sort_values("date")
        if len(raw) < 100:
            continue
        f = make_features(raw.reset_index(drop=True))
        f["market"] = fn[:-4]
        parts.append(f)
        if fn[:-4] == "KRW-BTC":
            btc = f[["date", "ret1", "ret5"]].rename(columns={"ret1": "btc_ret1", "ret5": "btc_ret5"})
    df = pd.concat(parts, ignore_index=True)
    g = df.groupby("date")
    df["mkt_ret1"] = g["ret1"].transform("median")
    df["breadth"] = g["ret1"].transform(lambda x: (x > 0).mean())
    df["mkt_volr"] = g["vol_r20"].transform("median")
    df["rank_ret1"] = g["ret1"].rank(pct=True)
    df["rank_volr"] = g["vol_r20"].rank(pct=True)
    if btc is not None:
        df = df.merge(btc, on="date", how="left")
    df["liquid"] = df["_value"] >= min_value
    return df


def add_target(df, pct, target, tp, stop, fee, slip):
    if target == "open_to_high":
        gain = df["_h1"] / df["_o1"] - 1
    elif target == "open_to_close":
        gain = df["_c1"] / df["_o1"] - 1
    else:
        gain = df["_h1"] / df["_c"] - 1
    df["y"] = (gain >= pct).astype(int)
    df["has_next"] = df["_c1"].notna()
    o1, h1, l1, c1 = df["_o1"], df["_h1"], df["_l1"], df["_c1"]
    sl, tgt = o1 * (1 - stop), o1 * (1 + tp)
    cost = o1 * (1 + slip) * (1 + fee)

    def net(ex):
        return ex * (1 - slip) * (1 - fee) / cost - 1

    # 일봉만으로는 같은 날 고가/저가 중 뭐가 먼저인지 모름 -> 두 가지 극단으로 범위를 표시
    df["net_pess"] = net(np.where(l1 <= sl, sl, np.where(h1 >= tgt, tgt, c1)))   # 손절 먼저(비관)
    df["net_opt"] = net(np.where(h1 >= tgt, tgt, np.where(l1 <= sl, sl, c1)))    # 익절 먼저(낙관)
    df["net_close"] = net(c1)                                                    # 시가 매수 -> 종가 청산
    # 손절 없이 지정가 익절만: 고가가 목표에 닿으면 목표가 체결, 아니면 종가 청산 (순서 모호성 없음)
    df["net_lim10"] = net(np.where(h1 >= o1 * 1.10, o1 * 1.10, c1))
    df["net_lim5"] = net(np.where(h1 >= o1 * 1.05, o1 * 1.05, c1))
    df["net"] = df["net_pess"]
    return df


def boot_ci(dates, vals, n=1000, seed=0):
    codes, _ = pd.factorize(dates)
    sums = np.bincount(codes, weights=vals)
    cnt = np.bincount(codes)
    rng = np.random.default_rng(seed)
    k = len(sums)
    e = np.empty(n)
    for i in range(n):
        p = rng.integers(0, k, k)
        e[i] = sums[p].sum() / cnt[p].sum()
    return np.percentile(e, [2.5, 97.5])


# ------------------------------------------------------------------
# 2. 분석
# ------------------------------------------------------------------
def univariate(tr, va, feats):
    rows = []
    bt, bv = tr["y"].mean(), va["y"].mean()
    for f in feats:
        x = tr[f].values
        q = np.nanquantile(x, [0.2, 0.4, 0.6, 0.8])
        q = np.unique(q)
        if len(q) < 2:
            continue
        gt = np.digitize(tr[f].values, q)
        gv = np.digitize(va[f].values, q)
        rt = pd.Series(tr["y"].values).groupby(gt).mean()
        rv = pd.Series(va["y"].values).groupby(gv).mean()
        edges = [-np.inf] + list(q) + [np.inf]
        for b in rt.index:
            if b not in rv.index:
                continue
            rows.append(dict(지표=f, 구간=f"{edges[b]:.4g} ~ {edges[b + 1]:.4g}",
                             학습lift=rt[b] / bt, 검증lift=rv[b] / bv))
    t = pd.DataFrame(rows)
    t["낮은쪽"] = t[["학습lift", "검증lift"]].min(axis=1)
    t["높은쪽"] = t[["학습lift", "검증lift"]].max(axis=1)
    return t


def leaf_rules(tree, names):
    t, out = tree.tree_, {}

    def rec(n, conds):
        if t.children_left[n] == -1:
            out[n] = " AND ".join(conds) if conds else "(전체)"
            return
        nm, th = names[t.feature[n]], t.threshold[n]
        rec(t.children_left[n], conds + [f"{nm} <= {th:.4g}"])
        rec(t.children_right[n], conds + [f"{nm} > {th:.4g}"])

    rec(0, [])
    return out


def topn_eval(part, proba, bench=None, ns=(1, 3, 5)):
    """하루 상위 N개 성적. 청산 가정 3가지(비관/낙관/시가→종가)를 같이 표시.
    bench(예: 어제 변동폭)를 주면 '모델 없이 그 지표만으로 고른 경우'도 비교."""
    d = part.assign(p=proba)
    if bench is not None:
        d["b"] = bench
    d = d[d["liquid"]]

    def row(name, x, ci=True):
        lo, hi = boot_ci(x["date"].values, x["net_pess"].values) if ci else (np.nan, np.nan)
        return dict(방식=name, 후보수=len(x), 폭등률=f"{x['y'].mean():.1%}",
                    수익_비관=f"{x['net_pess'].mean():+.2%}", 수익_낙관=f"{x['net_opt'].mean():+.2%}",
                    시가to종가=f"{x['net_close'].mean():+.2%}", 승률종가=f"{(x['net_close'] > 0).mean():.0%}",
                    지정가10만=f"{x['net_lim10'].mean():+.2%}", 지정가5만=f"{x['net_lim5'].mean():+.2%}",
                    비관95구간=f"{lo:+.1%}~{hi:+.1%}" if ci else "")

    rows = [row("전체(무작위 기준선)", d, ci=False)]
    for n in ns:
        sel = d.sort_values(["date", "p"], ascending=[True, False]).groupby("date").head(n)
        rows.append(row(f"모델 상위 {n}개", sel))
    if bench is not None:
        for n in ns:
            sel = d.sort_values(["date", "b"], ascending=[True, False]).groupby("date").head(n)
            rows.append(row(f"어제 변동폭 상위 {n}개(모델 없음)", sel))
    return pd.DataFrame(rows).to_string(index=False)


# ------------------------------------------------------------------
# 3. 실행
# ------------------------------------------------------------------
def run(args):
    df = load_all(args.data_dir, args.min_value)
    df = add_target(df, args.pct, args.target, args.tp, args.stop, args.fee, args.slip)
    feats = [c for c in df.columns if not c.startswith(("_", "net")) and c not in ("market", "date", "y",
                                                                                  "has_next", "liquid")]
    assert not any(f.startswith(("net", "_", "y")) or f in ("has_next",) for f in feats), "미래 정보가 지표에 섞였습니다"
    ok = df[feats].notna().all(axis=1)
    last_date = df["date"].max()
    today = df[ok & (df["date"] == last_date) & df["liquid"]].copy()
    lab = df[ok & df["has_next"] & df["liquid"]].copy()

    dates = np.sort(lab["date"].unique())
    c1, c2 = dates[int(len(dates) * 0.6)], dates[int(len(dates) * 0.8)]
    tr, va, te = lab[lab["date"] < c1], lab[(lab["date"] >= c1) & (lab["date"] < c2)], lab[lab["date"] >= c2]
    tdef = {"open_to_high": "다음날 시가 대비 고가", "open_to_close": "다음날 시가 대비 종가(방향 맞히기)", "close_to_high": "신호봉 종가 대비 다음날 고가"}[args.target]
    print(f"종목 {df['market'].nunique()}개 / 사용 가능한 코인-일 {len(lab):,}개 (거래대금 {args.min_value:,.0f}원 이상)")
    print(f"'폭등' 정의: {tdef} +{args.pct:.0%} 이상")
    print(f"학습 ~{str(c1)[:10]} 전 ({len(tr):,}), 검증 ~{str(c2)[:10]} 전 ({len(va):,}), 테스트 이후 ({len(te):,})")
    print(f"폭등 비율(기준선): 학습 {tr['y'].mean():.2%} / 검증 {va['y'].mean():.2%} / 테스트 {te['y'].mean():.2%}")
    print(f"거래 가정: 다음날 시가 매수, 손절 -{args.stop:.0%}, 익절 +{args.tp:.0%}, 안 닿으면 그날 종가 청산, 수수료·슬리피지 반영")

    # [1] 단일 지표
    print("\n=== [1] 단일 지표별 공통점 (학습/검증 둘 다에서 같은 방향으로 높은 구간) ===")
    t = univariate(tr, va, feats)
    good = t[(t["낮은쪽"] > 1.3)].sort_values("낮은쪽", ascending=False).head(15)
    if good.empty:
        print("학습·검증 모두 lift 1.3 이상인 단일 구간이 없습니다.")
    else:
        print(good[["지표", "구간", "학습lift", "검증lift"]].round(2).to_string(index=False))
    bad = t[(t["높은쪽"] < 0.7)].sort_values("높은쪽").head(8)
    print("\n(반대로 폭등이 덜 나오는 구간 - 피해야 할 조건)")
    print(bad[["지표", "구간", "학습lift", "검증lift"]].round(2).to_string(index=False) if len(bad) else "없음")

    # [2] 결정트리
    print("\n=== [2] 결정트리 (깊이 3) 규칙 (검증일수: 그 규칙이 검증 구간에서 며칠에 걸쳐 나왔는지 - 적으면 하루 장세 효과) ===")
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, random_state=0).fit(tr[feats], tr["y"])
    rules = leaf_rules(tree, feats)
    lt, lv = tree.apply(tr[feats]), tree.apply(va[feats])
    rows = []
    for leaf, rule in rules.items():
        mt, mv = lt == leaf, lv == leaf
        if mt.sum() == 0:
            continue
        vv = va[mv]
        rows.append(dict(학습n=int(mt.sum()), 학습폭등률=tr["y"].values[mt].mean(),
                         검증n=int(mv.sum()), 검증폭등률=va["y"].values[mv].mean() if mv.any() else np.nan,
                         검증일수=int(vv["date"].nunique()),
                         검증시가to종가=f"{vv['net_close'].mean():+.1%}" if len(vv) else "-",
                         검증지정가10=f"{vv['net_lim10'].mean():+.1%}" if len(vv) else "-",
                         규칙=rule))
    tt = pd.DataFrame(rows).sort_values("학습폭등률", ascending=False)
    for c in ("학습폭등률", "검증폭등률"):
        tt[c] = tt[c].map(lambda x: "-" if pd.isna(x) else f"{x:.1%}")
    print(tt.to_string(index=False))

    # [3] 부스팅
    print("\n=== [3] 그래디언트 부스팅: 학습에서 배우고 검증/테스트에서 하루 상위 N개 성적 ===")
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                        early_stopping=True, validation_fraction=0.15, random_state=0)
    gb.fit(tr[feats], tr["y"])
    pv, pt = gb.predict_proba(va[feats])[:, 1], gb.predict_proba(te[feats])[:, 1]
    print(f"AUC(0.5=우연): 검증 {roc_auc_score(va['y'], pv):.3f} / 테스트 {roc_auc_score(te['y'], pt):.3f}")
    print("수익_비관: 같은 날 손절·익절 둘 다 닿으면 손절 먼저 / 수익_낙관: 익절 먼저 / 시가to종가: 시가에 사서 종가에 팖\n지정가10만·5만: 손절 없이 +10%/+5% 지정가 매도만 걸고 안 닿으면 종가 청산 (일봉만으로 정확히 계산 가능)")
    print("→ 실제 결과는 비관과 낙관 사이이고, 일봉만으로는 순서를 알 수 없습니다.")
    print("\n[검증 구간]")
    print(topn_eval(va, pv, bench=va["range"].values))
    print("\n[테스트 구간 - 처음 한 번만 의미 있음]")
    print(topn_eval(te, pt, bench=te["range"].values))

    sub = va.sample(min(15000, len(va)), random_state=0)
    imp = permutation_importance(gb, sub[feats], sub["y"], scoring="roc_auc", n_repeats=3, random_state=0)
    order = np.argsort(-imp.importances_mean)[:12]
    print("\n중요한 지표 (섞었을 때 AUC가 얼마나 떨어지는지):")
    for i in order:
        print(f"  {feats[i]:<18} {imp.importances_mean[i]:.4f}")

    # [4] 오늘
    print("\n=== [4] 오늘의 상위 후보 (연구용. 학습+검증+테스트 전체로 다시 학습) ===")
    gb2 = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.15, random_state=0)
    gb2.fit(lab[feats], lab["y"])
    if len(today):
        today["확률"] = gb2.predict_proba(today[feats])[:, 1]
        show = today.sort_values("확률", ascending=False).head(10)
        show = show[["market", "_c", "확률", "ret1", "vol_r20", "pctb", "psy10", "cmf20"]].copy()
        show["확률"] = show["확률"].map("{:.1%}".format)
        print(f"기준 신호봉: {str(last_date)[:10]} (평균 폭등 확률 {lab['y'].mean():.1%})")
        print(show.round(3).to_string(index=False))
    else:
        print("오늘 후보 데이터가 없습니다.")
    print("\n※ 연구용입니다. 확률이 높아도 매수 신호가 아닙니다. X1(글로벌 괴리)·거래정지/유의·입금량 경고는 직접 확인하세요.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="upbit_daily")
    ap.add_argument("--pct", type=float, default=0.10, help="폭등 기준 상승률")
    ap.add_argument("--target", choices=["open_to_high", "open_to_close", "close_to_high"], default="open_to_high")
    ap.add_argument("--tp", type=float, default=0.10, help="거래 가정: 익절폭")
    ap.add_argument("--stop", type=float, default=0.05, help="거래 가정: 손절폭")
    ap.add_argument("--min-value", type=float, default=1e8, help="신호봉 거래대금 하한(원)")
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.001)
    ap.add_argument("--out", default="results_discover.txt")
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
        finally:
            sys.stdout = real
    print(f"\n화면 내용 전체가 {args.out} 에도 저장됐습니다.")


if __name__ == "__main__":
    main()
