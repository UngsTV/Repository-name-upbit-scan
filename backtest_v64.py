#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
폭등 직전 후보 공식 v6.4 - 업비트 KRW 마켓 일봉 백테스트

사용법
  pip install pandas numpy requests
  python backtest_v64.py fetch            # 업비트 KRW 전 종목 일봉 내려받기 (최초 1회, 몇 분 걸림)
  python backtest_v64.py fetch --days 4000 --refresh   # 상장 이후 전체 기간으로 다시 받기(오래 걸림)
  python backtest_v64.py fetch --update   # 매일: 최근 며칠만 이어붙이기(빠름)
  python backtest_v64.py run              # 백테스트 (기본값 + 하나씩 바꿔보기 + 랜덤 탐색)
  python backtest_v64.py scan             # 가장 최근 마감 일봉 기준 오늘의 후보 목록(v6.5) + paper_log.csv 기록
  python backtest_v64.py check            # 기록해 둔 후보가 며칠 뒤 실제로 어땠는지 확인(미래 검증)

주의: 연구/검증용입니다. 자동 매수 기능은 없고, 과거 성과는 미래를 보장하지 않습니다.
"""
import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

BASE = "https://api.upbit.com/v1"
KST = timezone(timedelta(hours=9))

# ------------------------------------------------------------------
# 공식 v6.4 파라미터 (전부 '임시값'. 아래 GRID가 탐색 범위)
# ------------------------------------------------------------------
DEFAULT = dict(
    cut=70,          # 후보 컷 점수
    x2=0.25,         # X2 신호봉 당일 상승률 제외
    x3=115,          # X3 이격도(MA5) 제외 (초과)
    x6=0.30,         # X6 최근 5봉 누적 상승률 제외
    x7=0.85,         # X7 %B 제외
    s1_win=10,       # S1 볼린저 하단 터치 확인 구간(봉)
    s2_thr=30,       # S2 투자심리도(10) 20봉 최저 기준
    r2_thr=99,       # R2 이격도(MA5) 하한
    v2_thr=0.10,     # V2 CMF20 5봉 변화 기준
    surge=0.30,      # 폭등 정의: 다음 1~3일 고가 >= 신호봉 종가*(1+surge)
    use_v1=True,     # V1(거래량) 사용 여부 (제거 후보 1순위)
    tp=0.05,         # [내 추가] 시험 익절폭 (진입가 대비). 목표 +5%
    hold=3,          # 최대 보유일(3 또는 5). 이후 종가 청산
    stop=0.0,        # 0 = 신호봉 저가 이탈 손절, 0.05 = 진입가 -5% 손절
)
V65 = dict(DEFAULT, cut=90, r2_thr=103, s2_thr=30, x2=9.0, x3=999, x6=9.0, x7=0.95, tp=0.10)  # v6.5 시험안(2차): 점수 컷 높이고 효과 없는 필터는 끔

GRID = dict(
    cut=[70, 75, 80, 85, 90],
    x2=[0.25, 9.0],          # 9.0 = 필터 끄기
    x3=[115, 999],           # 999 = 필터 끄기
    x6=[0.30, 9.0],          # 9.0 = 필터 끄기
    x7=[0.85, 0.95, 1.01],   # 1.01 = 필터 끄기
    s1_win=[5, 10],
    s2_thr=[20, 30, 40],     # 투자심리도는 10 단위라 25/35는 20/30과 같음
    r2_thr=[99, 101, 103],
    v2_thr=[0.05, 0.10, 0.15],
    surge=[0.20, 0.30, 0.40],
    use_v1=[True, False],
    tp=[0.05, 0.10, 0.15],
)


# ------------------------------------------------------------------
# 1. 데이터 수집
# ------------------------------------------------------------------
def get_json(url, params=None, tries=8):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=10)
        except requests.RequestException:
            time.sleep(1 + i)
            continue
        if r.status_code == 200:
            return r.json()
        time.sleep(1 + i)  # 429(요청 제한) 등은 잠깐 쉬고 재시도
    raise RuntimeError(f"요청 실패: {url} {params}")


def fetch_market(market, days):
    rows, to = [], None
    while len(rows) < days:
        params = {"market": market, "count": 200}
        if to:
            params["to"] = to
        batch = get_json(f"{BASE}/candles/days", params)
        if not batch:
            break
        rows += batch
        to = batch[-1]["candle_date_time_utc"] + "Z"  # 이 시각 '이전' 캔들을 다음에 받음
        time.sleep(0.12)  # 업비트 캔들 API 초당 10회 제한 대응
        if len(batch) < 200:
            break
    if not rows:
        return None
    df = pd.DataFrame(rows).rename(columns={
        "opening_price": "open", "high_price": "high", "low_price": "low",
        "trade_price": "close", "candle_acc_trade_volume": "volume",
        "candle_acc_trade_price": "value"})
    df["date"] = pd.to_datetime(df["candle_date_time_kst"])  # 일봉 시작 시각(KST 09:00)
    df = df[["date", "open", "high", "low", "close", "volume", "value"]]
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    # 아직 마감 안 된 오늘 캔들 제거 (마감된 일봉만 사용)
    now_kst = datetime.now(KST).replace(tzinfo=None)
    df = df[df["date"] + pd.Timedelta(days=1) <= now_kst]
    return df.reset_index(drop=True)


def cmd_fetch(args):
    os.makedirs(args.data_dir, exist_ok=True)
    info = get_json(f"{BASE}/market/all", {"isDetails": "false"})
    markets = sorted(m["market"] for m in info if m["market"].startswith("KRW-"))
    mode = "최근 데이터만 이어붙이기(--update)" if args.update else f"최근 {args.days}일"
    print(f"KRW 마켓 {len(markets)}개, {mode} 내려받는 중...")
    for i, m in enumerate(markets, 1):
        path = os.path.join(args.data_dir, f"{m}.csv")
        have = os.path.exists(path)
        if args.update and have:
            new = fetch_market(m, 15)  # 최근 며칠만 받아서 기존 파일에 합침(같은 날짜는 최신값으로 덮어씀)
            if new is not None and len(new) > 0:
                old = pd.read_csv(path, parse_dates=["date"])
                comb = pd.concat([old, new]).drop_duplicates("date", keep="last").sort_values("date")
                comb.to_csv(path, index=False)
        elif have and not args.refresh and not args.update:
            continue
        else:
            df = fetch_market(m, args.days)  # 새 상장 종목이거나 전체 다시 받기
            if df is not None and len(df) > 0:
                df.to_csv(path, index=False)
        if i % 20 == 0:
            print(f"  {i}/{len(markets)}")
    print("완료. 이어서: python backtest_v64.py run  (또는 run_daily.bat)")


# ------------------------------------------------------------------
# 2. 지표 계산
# ------------------------------------------------------------------
def build_features(raw, name, mid):
    c, h, l, o, v = raw["close"], raw["high"], raw["low"], raw["open"], raw["volume"]
    f = pd.DataFrame({"date": raw["date"]})
    f["market"], f["mid"], f["pos"] = name, mid, np.arange(len(raw))
    f["sig_close"], f["sig_low"], f["value"] = c, l, raw["value"]

    # 볼린저(20, 2), %B, 하단 터치
    ma20 = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    up, lo = ma20 + 2 * sd, ma20 - 2 * sd
    f["pctb"] = (c - lo) / (up - lo).replace(0, np.nan)
    gap = l - lo
    for k in (5, 10, 15):
        f[f"touch{k}"] = gap.rolling(k).min() <= 0

    # 투자심리도(10) 20봉 최저
    upday = (c.diff() > 0).astype(float)
    psy = upday.rolling(10).sum() / 10 * 100
    f["psy_min20"] = psy.rolling(20).min()

    # 이격도(MA5), 당일 상승률, 5봉 누적, 양봉
    f["disp"] = c / c.rolling(5).mean() * 100
    f["ret1"] = c.pct_change()
    f["vol20"] = c.pct_change().rolling(20).std()
    f["cum5"] = c / c.shift(5) - 1
    f["bull"] = c > o

    # OBV >= 10봉 전
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["obv_ok"] = obv >= obv.shift(10)

    # V1: 거래량 MA5 <= MA20의 50%  또는  신호봉 거래량 >= MA20*2
    vma5, vma20 = v.rolling(5).mean(), v.rolling(20).mean()
    f["v1"] = (vma5 <= 0.5 * vma20) | (v >= 2 * vma20)

    # V2: CMF20 의 5봉 변화
    rng = (h - l).replace(0, np.nan)
    mfm = (((c - l) - (h - c)) / rng).fillna(0)
    cmf = (mfm * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    f["cmf_chg"] = cmf - cmf.shift(5)

    # V3: MDI(14, Wilder) 가 3봉 전보다 하락
    upm, dnm = h.diff(), -l.diff()
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0)  # 참고용(점수에는 안 씀)
    mdm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=raw.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    f["mdi_down"] = mdi < mdi.shift(3)

    # 미래값(검증용): 다음 1~3봉
    for d in (1, 2, 3, 4, 5):
        f[f"h{d}"], f[f"l{d}"] = h.shift(-d), l.shift(-d)
    f["o1"], f["c3"], f["c5"] = o.shift(-1), c.shift(-3), c.shift(-5)
    f["maxh3"] = f[["h1", "h2", "h3"]].max(axis=1, skipna=False)
    return f[f["pos"] >= 40]  # 지표 워밍업 구간 제외


def load_all(data_dir, min_value):
    files = sorted(x for x in os.listdir(data_dir) if x.endswith(".csv"))
    if not files:
        sys.exit("데이터가 없습니다. 먼저 `python backtest_v64.py fetch` 를 실행하세요.")
    parts = []
    for mid, fn in enumerate(files):
        raw = pd.read_csv(os.path.join(data_dir, fn), parse_dates=["date"])
        if len(raw) < 80:
            continue
        parts.append(build_features(raw, fn[:-4], mid))
    df = pd.concat(parts, ignore_index=True)
    df["fwd_ok"] = df["c5"].notna()  # 최대 5일 보유까지 비교하려고 5봉 뒤 데이터가 있는 신호봉만 사용
    df["liquid"] = df["value"] >= min_value
    dates = np.sort(df.loc[df["fwd_ok"], "date"].unique())
    cut_date = dates[int(len(dates) * 0.7)]  # 시간순 앞 70% 학습 / 뒤 30% 검증
    df["train"] = df["fwd_ok"] & (df["date"] < cut_date)
    df["valid"] = df["fwd_ok"] & (df["date"] >= cut_date)
    print(f"종목 {len(parts)}개, 신호봉 {len(df):,}개, 학습/검증 경계일 {str(cut_date)[:10]}")
    return df


# ------------------------------------------------------------------
# 3. 공식 적용 (유형 A)
# ------------------------------------------------------------------
def candidate_mask(df, p):
    s1 = df[f"touch{p['s1_win']}"].values
    s2 = (df["psy_min20"] <= p["s2_thr"]).values
    gate = s1 | s2
    score = (20 * s1 + 15 * s2
             + 10 * df["bull"].values
             + 15 * (df["disp"] >= p["r2_thr"]).values
             + 10 * df["obv_ok"].values
             + 10 * (df["cmf_chg"] >= p["v2_thr"]).values
             + 5 * df["mdi_down"].values)
    cut = p["cut"]
    if p["use_v1"]:
        score = score + 15 * df["v1"].values
    else:
        cut = cut * 0.85  # V1(15점) 제거 시 만점이 85점이므로 컷도 비례 조정
    excluded = ((df["ret1"] >= p["x2"]).values            # X2
                | (df["disp"] > p["x3"]).values           # X3
                | (df["cum5"] >= p["x6"]).values          # X6
                | (df["pctb"] >= p["x7"]).values)         # X7
    # X1(글로벌 시세 괴리), X4(거래정지/유의), X5(입금량 급증)는 과거 데이터가 없어 백테스트 불가
    mask = gate & (score >= cut) & ~excluded & df["liquid"].values
    return mask, score


def dedupe(df, mask):
    """같은 종목에서 3봉 안에 연속 등록된 후보는 첫 번째만 인정(중복 포지션 방지)"""
    idx = np.flatnonzero(mask)
    mk, ps = df["mid"].values[idx], df["pos"].values[idx]
    keep = np.zeros(len(idx), bool)
    lm, lp = -1, -99
    for i in range(len(idx)):
        if mk[i] == lm and ps[i] - lp <= 3:
            continue
        keep[i] = True
        lm, lp = mk[i], ps[i]
    out = np.zeros(len(mask), bool)
    out[idx[keep]] = True
    return out


def trade_returns(df, idx, p, cfg):
    """다음 캔들 시가 진입 / 손절(신호봉 저가 또는 진입가-stop%) / 익절=진입가*(1+tp) /
    hold일차 종가 청산. 같은 날 손절·익절이 겹치면 손절 먼저(보수적). 수수료·슬리피지 반영."""
    hold = p.get("hold", 3)
    entry = df["o1"].values[idx]
    stop = df["sig_low"].values[idx] if p.get("stop", 0.0) == 0 else entry * (1 - p["stop"])
    target = entry * (1 + p["tp"])
    exit_ = np.full(len(idx), np.nan)
    alive = np.ones(len(idx), bool)
    for d in range(1, hold + 1):
        hi, lo = df[f"h{d}"].values[idx], df[f"l{d}"].values[idx]
        stop_hit = alive & (lo <= stop)
        px = np.minimum(stop, entry) if d == 1 else stop
        exit_[stop_hit] = px[stop_hit]
        alive &= ~stop_hit
        tp_hit = alive & (hi >= target)
        exit_[tp_hit] = target[tp_hit]
        alive &= ~tp_hit
    exit_[alive] = df[f"c{hold}"].values[idx][alive]
    cost = entry * (1 + cfg["slip"]) * (1 + cfg["fee"])
    proceeds = exit_ * (1 - cfg["slip"]) * (1 - cfg["fee"])
    return proceeds / cost - 1


def summarize(df, p, cfg, cand, split):
    m = df[split].values
    hit_all = df["maxh3"].values >= df["sig_close"].values * (1 + p["surge"])
    bm = m & df["liquid"].values
    base = hit_all[bm].mean() if bm.any() else np.nan
    idx = np.flatnonzero(cand & m)
    if len(idx) == 0:
        return dict(n=0, precision=np.nan, base=base, lift=np.nan, ev=np.nan, win=np.nan)
    prec = hit_all[idx].mean()
    net = trade_returns(df, idx, p, cfg)
    return dict(n=len(idx), precision=prec, base=base,
                lift=prec / base if base and base > 0 else np.nan,
                ev=net.mean(), win=(net > 0).mean())


def run_params(df, p, cfg):
    mask, score = candidate_mask(df, p)
    cand = dedupe(df, mask & df["fwd_ok"].values)
    return {s: summarize(df, p, cfg, cand, s) for s in ("train", "valid")}, cand, score


def flat(p, res):
    r = dict(p)
    for s in ("train", "valid"):
        for k, v in res[s].items():
            r[f"{s}_{k}"] = v
    return r


def fmt(tbl):
    t = tbl.copy()
    for c in t.columns:
        if t[c].dtype == float:
            t[c] = t[c].map(lambda x: "-" if pd.isna(x) else f"{x:.3f}")
    return t.to_string(index=False)


def vol_adj(df, p, cand, split):
    """변동성 보정 lift: 후보와 '변동성이 비슷한' 코인-일들의 폭등 확률을 기준선으로 비교.
    높은 점수가 단순히 '원래 많이 움직이는 코인'을 고르는 것인지 확인용."""
    ok = df[split].values & df["liquid"].values & df["vol20"].notna().values
    hit = df["maxh3"].values >= df["sig_close"].values * (1 + p["surge"])
    vol = df["vol20"].values
    edges = np.nanquantile(vol[ok], np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(edges[1:-1], vol, side="right"), 0, 9)
    base = np.array([hit[ok & (dec == d)].mean() if (ok & (dec == d)).any() else np.nan
                     for d in range(10)])
    idx = np.flatnonzero(cand & df[split].values & df["vol20"].notna().values)
    if len(idx) == 0:
        return np.nan, np.nan, 0
    exp = np.nanmean(base[dec[idx]])
    return hit[idx].mean() / exp, exp, len(idx)


def boot_ci(df, idx, net, n=2000, seed=0):
    """같은 날 후보는 같이 움직이므로 '날짜 단위'로 묶어서 재표집한 평균 수익률 95% 구간"""
    codes, _ = pd.factorize(df["date"].values[idx])
    sums = np.bincount(codes, weights=net)
    cnt = np.bincount(codes)
    rng = np.random.default_rng(seed)
    k = len(sums)
    evs = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, k, k)
        evs[i] = sums[pick].sum() / cnt[pick].sum()
    return np.percentile(evs, [2.5, 97.5])


def trade_report(df, cand, p, cfg, title, split=None):
    """거래 단위 성적표: 승률, 평균 손익, 연속 승/패, 건당 10만원 고정 투자 시 누적 손익"""
    m = cand & df["fwd_ok"].values
    if split is not None:
        m = m & df[split].values
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        print(f"[{title}] 거래 없음")
        return
    idx = idx[np.argsort(df["date"].values[idx], kind="stable")]  # 날짜순
    net = trade_returns(df, idx, p, cfg)
    win = net > 0
    wr = win.mean()
    aw = net[win].mean() if win.any() else 0.0
    al = net[~win].mean() if (~win).any() else 0.0
    be = -al / (aw - al) if (aw - al) != 0 and al < 0 else float("nan")
    mw = ml = cw = cl = 0
    for w in win:
        if w:
            cw, cl = cw + 1, 0
        else:
            cl, cw = cl + 1, 0
        mw, ml = max(mw, cw), max(ml, cl)
    pnl = 100000 * np.cumsum(net)
    curve = np.r_[0.0, pnl]
    dd = (curve - np.maximum.accumulate(curve)).min()
    print(f"[{title}] 거래 {len(idx)}건 | 승률 {wr:.1%} | 평균 이익 {aw:+.2%} / 평균 손실 {al:+.2%}"
          f" | 본전 승률 {be:.1%}")
    print(f"    최대 연속 승 {mw}회 / 최대 연속 패 {ml}회 | 건당 10만원 고정 투자 시 누적 {pnl[-1]:+,.0f}원,"
          f" 최대 낙폭 {dd:,.0f}원")
    lo_, hi_ = boot_ci(df, idx, net)
    print(f"    거래당 평균 수익률 {net.mean():+.2%} (날짜단위 부트스트랩 95% 구간 {lo_:+.2%} ~ {hi_:+.2%})"
          f"{'  ← 0보다 큼' if lo_ > 0 else ('  ← 전부 마이너스' if hi_ < 0 else '  ← 0을 포함: 우위 입증 안 됨')}")
    print(f"    이 승률로 300연승할 확률 = {wr ** 300:.2e}")


# ------------------------------------------------------------------
# 4. 실행 명령
# ------------------------------------------------------------------
def cmd_run(args):
    cfg = dict(fee=args.fee, slip=args.slip)
    df = load_all(args.data_dir, args.min_value)

    # (1) 기본값 + 합격 기준 점검
    res, cand, score = run_params(df, DEFAULT, cfg)
    print("\n=== [1] 기본값(v6.4 임시값) 결과 ===")
    print(fmt(pd.DataFrame([dict(구간=s, **v) for s, v in res.items()])))
    n_all = res["train"]["n"] + res["valid"]["n"]
    ok = (n_all >= 30
          and res["train"]["lift"] >= 2 and res["valid"]["lift"] >= 2
          and res["valid"]["ev"] > 0)
    print(f"합격 기준(후보 30개↑, 학습·검증 둘 다 lift 2배↑, 검증 기대값>0): "
          f"{'통과' if ok else '미달'}  (후보 {n_all}개)")
    out = df.loc[cand, ["market", "date", "sig_close", "maxh3", "h1", "h2", "h3", "c3"]].copy()
    out["score"] = score[cand]
    out["hit"] = out["maxh3"] >= out["sig_close"] * (1 + DEFAULT["surge"])
    out.to_csv("candidates_default.csv", index=False, encoding="utf-8-sig")
    print("후보 목록 저장: candidates_default.csv (차트 눈으로 확인용)")
    ndays = df.loc[df["fwd_ok"], "date"].nunique()
    print(f"하루 평균 후보 수(v6.4 기본값): {cand.sum() / max(ndays, 1):.1f}개 / 전체 {int(cand.sum())}개")

    res65, cand65, score65 = run_params(df, V65, cfg)
    print("\n=== [1b] v6.5 시험안 (cut90, R2 103, X2/X3/X6 끔, X7 0.95, tp10%) ===")
    print(fmt(pd.DataFrame([dict(구간=s_, **v) for s_, v in res65.items()])))
    print(f"하루 평균 후보 수: {cand65.sum() / max(ndays, 1):.1f}개")
    trade_report(df, cand65, V65, cfg, "v6.5 전체")
    trade_report(df, cand65, V65, cfg, "v6.5 검증 구간만", split="valid")
    print("\n--- 변동성 보정 lift (1.0이면 '원래 잘 움직이는 코인'을 고른 효과 외엔 없음) ---")
    for nm, pp, cc in (("v6.4 기본값", DEFAULT, cand), ("v6.5 시험안", V65, cand65)):
        for sp in ("train", "valid"):
            la, ex, nn = vol_adj(df, pp, cc, sp)
            print(f"  {nm} {sp}: 후보 {nn}개, 보정 lift {la:.2f} (변동성 비슷한 코인의 폭등 확률 {ex:.1%})")
    print("\n=== [4] v6.5 후보에 청산 규칙만 바꿔보기 (신호는 그대로) ===")
    rows = []
    for hold in (3, 5):
        for stp in (0.0, 0.05, 0.08):
            for tpv in (0.05, 0.10, 0.20, 0.30, 0.50):
                pp = dict(V65, hold=hold, stop=stp, tp=tpv)
                r = dict(보유일=hold, 손절="신호봉저가" if stp == 0 else f"-{stp:.0%}", 익절=f"+{tpv:.0%}")
                for sp, nm in (("train", "학습"), ("valid", "검증")):
                    ii = np.flatnonzero(cand65 & df[sp].values)
                    nt = trade_returns(df, ii, pp, cfg)
                    r[f"{nm}_승률%"] = round(100 * (nt > 0).mean(), 1)
                    r[f"{nm}_ev%"] = round(100 * nt.mean(), 2)
                    if sp == "valid":
                        lo_, hi_ = boot_ci(df, ii, nt, n=500)
                        r["검증_95%구간"] = f"{100 * lo_:+.1f}~{100 * hi_:+.1f}"
                rows.append(r)
    t4 = pd.DataFrame(rows)
    t4["둘중낮은ev%"] = t4[["학습_ev%", "검증_ev%"]].min(axis=1)
    t4 = t4.sort_values("둘중낮은ev%", ascending=False)
    t4.to_csv("results_exit_rules.csv", index=False, encoding="utf-8-sig")
    print(t4.head(12).to_string(index=False))
    print("※ 30개 조합 중 고른 것이라 우연일 수 있습니다. 학습·검증 둘 다 +이고 95% 구간 아래쪽이 0 근처 이상이어야 의미 있습니다.")

    print("\n--- v6.5 익절폭별 검증 구간 성적 ---")
    for tpv in (0.05, 0.10, 0.15):
        pp = dict(V65, tp=tpv)
        trade_report(df, cand65, pp, cfg, f"v6.5 tp {tpv:.0%}", split="valid")
    print(f"\n--- 거래 성적표 (익절 +{DEFAULT['tp']:.0%}, 손절 신호봉 저가 이탈, 3일 뒤 종가 청산) ---")
    trade_report(df, cand, DEFAULT, cfg, "전체")
    trade_report(df, cand, DEFAULT, cfg, "검증 구간만", split="valid")

    # (2) 하나씩만 바꿔보기
    print("\n=== [2] 파라미터 하나씩 바꿔보기 (나머지는 기본값) ===")
    rows = []
    for k, vals in GRID.items():
        for v in vals:
            p = dict(DEFAULT, **{k: v})
            r, _, _ = run_params(df, p, cfg)
            rows.append(dict(param=k, value=v, n_train=r["train"]["n"],
                             lift_train=r["train"]["lift"], n_valid=r["valid"]["n"],
                             lift_valid=r["valid"]["lift"], prec_valid=r["valid"]["precision"],
                             ev_valid=r["valid"]["ev"]))
    t2 = pd.DataFrame(rows)
    t2.to_csv("results_one_at_a_time.csv", index=False, encoding="utf-8-sig")
    print(fmt(t2))

    # (3) 랜덤 탐색 (학습 구간 기준으로만 고르고, 검증 구간 성적을 같이 본다)
    if args.random > 0:
        print(f"\n=== [3] 랜덤 탐색 {args.random}회 (학습·검증 중 낮은 쪽 lift 순 상위 10) ===")
        random.seed(1)
        rows = []
        for _ in range(args.random):
            p = {k: random.choice(v) for k, v in GRID.items()}
            r, _, _ = run_params(df, p, cfg)
            if r["train"]["n"] >= 30 and r["valid"]["n"] >= 30:
                rows.append(flat(p, r))
        t3 = pd.DataFrame(rows)
        if len(t3):
            t3["min_lift"] = t3[["train_lift", "valid_lift"]].min(axis=1)
            t3 = t3.sort_values("min_lift", ascending=False)
            t3.to_csv("results_random.csv", index=False, encoding="utf-8-sig")
            cols = ["cut", "x2", "x3", "x6", "x7", "s1_win", "s2_thr", "r2_thr", "v2_thr",
                    "surge", "use_v1", "tp", "train_n", "train_lift", "valid_n", "valid_lift",
                    "valid_ev"]
            print(fmt(t3[cols].head(10)))
            print("\n--- 상위 5개 조합의 변동성 보정 lift ---")
            for _, row in t3.head(5).iterrows():
                pp = dict(DEFAULT, **{k: row[k].item() if hasattr(row[k], "item") else row[k] for k in GRID})
                _, cc, _ = run_params(df, pp, cfg)
                a1 = vol_adj(df, pp, cc, "train")[0]
                a2 = vol_adj(df, pp, cc, "valid")[0]
                print(f"  cut {pp['cut']} r2 {pp['r2_thr']} s2 {pp['s2_thr']} v1 {pp['use_v1']}: "
                      f"보정 lift 학습 {a1:.2f} / 검증 {a2:.2f}")
            print("※ 300개 중 최상위만 보면 우연히 좋아 보이는 조합이 섞입니다. 보정 lift와 신뢰구간으로 다시 거르세요.")
            print("※ 학습에서 좋았던 조합이 검증에서도 유지되는지가 핵심입니다. 검증에서 무너지면 과최적화입니다.")
        else:
            print("학습 구간 후보 30개 이상인 조합이 없습니다.")

    print("\n=== [5] 오늘의 후보 (가장 최근 마감 일봉 기준, v6.5 시험안) ===")
    cmd_scan(args, df)


def cmd_scan(args, df=None):
    if df is None:
        df = load_all(args.data_dir, args.min_value)
    last = df["date"].max()
    mask, score = candidate_mask(df, V65)
    m = mask & (df["date"] == last).values
    print(f"\n기준 신호봉: {str(last)[:10]} 09:00 시작 일봉 (마감 완료) / v6.5 시험안 기준 / 다음 캔들에서 진입 여부 판단")
    out = df.loc[m, ["market", "sig_close", "ret1", "disp", "pctb", "sig_low"]].copy()
    out["score"] = score[m]
    out = out.sort_values("score", ascending=False)
    if out.empty:
        print("후보 없음 (v6.5는 하루 평균 약 2개, 0개인 날도 많습니다)")
        near = df[(df["date"] == last) & df["liquid"]].copy()
        near["score"] = score[near.index.values]
        near = near.sort_values("score", ascending=False).head(5)
        print("참고: 컷 미달이지만 점수 높은 상위 5개 (매수 후보 아님)")
        print(fmt(near[["market", "sig_close", "score", "disp", "pctb"]]))
    else:
        print(fmt(out.rename(columns={"sig_low": "손절선(신호봉저가)"})))
        log = df.loc[m, ["market", "date", "sig_close", "sig_low"]].copy()
        log["score"] = score[m]
        path = "paper_log.csv"
        if os.path.exists(path):
            old = pd.read_csv(path, parse_dates=["date"])
            log = pd.concat([old, log]).drop_duplicates(["market", "date"], keep="last")
        log.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"후보를 {path}에 기록했습니다(모의 검증용). 며칠 뒤 `python backtest_v64.py check`로 결과를 확인하세요.")
    print("※ 후보 등록일 뿐 매수 신호가 아닙니다. X1(글로벌 시세 괴리)·X4(거래정지/유의)·X5(입금량)는 직접 확인하세요.")


def cmd_check(args):
    """scan으로 기록해 둔 후보(paper_log.csv)가 이후 며칠간 실제로 어떻게 됐는지 확인 (미래 검증)"""
    if not os.path.exists("paper_log.csv"):
        sys.exit("paper_log.csv가 없습니다. 먼저 fetch --refresh 후 scan을 며칠 실행하세요.")
    log = pd.read_csv("paper_log.csv", parse_dates=["date"])
    df = load_all(args.data_dir, args.min_value)
    df["fwd_ok"] = df["c3"].notna()
    key = df[["market", "date"]].reset_index().merge(log[["market", "date"]], on=["market", "date"])
    sel = np.zeros(len(df), bool)
    sel[key["index"].values] = True
    done = sel & df["fwd_ok"].values
    print(f"\n기록된 후보 {len(log)}개 중 3일 결과가 나온 것 {int(done.sum())}개")
    if done.sum() == 0:
        print("아직 결과가 나온 후보가 없습니다. fetch --refresh 로 데이터를 새로 받은 뒤 다시 실행하세요.")
        return
    hit = df["maxh3"].values >= df["sig_close"].values * (1 + V65["surge"])
    print(f"폭등(고가 +{V65['surge']:.0%}) 비율: {hit[done].mean():.1%}  (참고: 백테스트 검증 구간 약 7%, 아무 코인이나 샀을 때 약 3%)")
    trade_report(df, done, V65, dict(fee=args.fee, slip=args.slip), "모의 검증 결과")
    ii = np.flatnonzero(done)
    print(df.loc[ii, ["market", "date", "sig_close", "maxh3", "c3"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run", "scan", "check"])
    ap.add_argument("--data-dir", default="upbit_daily")
    ap.add_argument("--days", type=int, default=800, help="종목당 내려받을 일봉 수")
    ap.add_argument("--refresh", action="store_true", help="이미 받은 파일도 처음부터 다시 받기")
    ap.add_argument("--update", action="store_true", help="이미 받은 파일에 최근 며칠만 이어붙이기(매일 사용, 1분 이내)")
    ap.add_argument("--random", type=int, default=300, help="랜덤 탐색 횟수 (0이면 생략)")
    ap.add_argument("--min-value", type=float, default=1e8, help="신호봉 거래대금 하한(원)")
    ap.add_argument("--fee", type=float, default=0.0005, help="편도 수수료")
    ap.add_argument("--slip", type=float, default=0.001, help="편도 슬리피지")
    args = ap.parse_args()
    if args.cmd in ("run", "scan", "check"):
        # 화면에 나오는 내용을 파일로도 저장 -> 이 파일 하나만 보내면 됨
        class Tee:
            def __init__(self, *fs):
                self.fs = fs
            def write(self, x):
                for f in self.fs:
                    f.write(x)
            def flush(self):
                for f in self.fs:
                    f.flush()
        name = f"results_summary_{args.cmd}.txt"
        with open(name, "w", encoding="utf-8-sig") as fh:
            real = sys.stdout
            sys.stdout = Tee(real, fh)
            try:
                {"run": cmd_run, "scan": cmd_scan, "check": cmd_check}[args.cmd](args)
            finally:
                sys.stdout = real
        print(f"\n화면 내용 전체가 {name} 파일에도 저장됐습니다. 이 파일 하나만 보내주세요.")
    else:
        cmd_fetch(args)


if __name__ == "__main__":
    main()
