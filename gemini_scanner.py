#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
업비트 일봉 스캐너 (제미나이 규칙) -> 디스코드 알림

직전 마감 일봉 기준으로 아래 8개 조건을 모두 만족하는 종목을 알려줍니다.
  1) 양봉(종가 > 시가) + 거래량이 그 전 20일 평균의 3배 초과
  2) 20일 이격도 100~118%
  3) 일목균형표 전환선(9) > 기준선(26)
  4) 아론 오실레이터(25) > 0
  5) 엘더레이 베어 파워(EMA13) > 0   (= 저가 - EMA13)
  6) 돈치안 채널(20) 위치 pc_pos >= 0.52   (= (종가-20일저가)/(20일고가-20일저가))
  7) OBV가 OBV의 10일 이동평균 위
  8) 소나(25) > 0
60일 미만 데이터 종목은 제외합니다.

[가정한 값] 아래 상수는 제미나이가 정확한 수식을 알려주지 않아 일반적인 값으로 정했습니다.
  PC_POS_MIN, SONAR_SMOOTH, 일목 9/26, 아론 25, 엘더 EMA13, OBV 이평 10
기존 v6.5 알림(notify.py)과 헷갈리지 않도록 모든 메시지 맨 위에 TITLE 을 붙입니다.
필요한 것: pip install pandas numpy requests, 환경변수 DISCORD_WEBHOOK_URL
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

TITLE = "🔔 [제미나이 규칙] 거래량 3배·양봉 + 추세지표 7종"

MARKET_API = "https://api.upbit.com/v1/market/all?isDetails=false"
CANDLE_API = "https://api.upbit.com/v1/candles/days"
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

CANDLE_COUNT = 100
MIN_CANDLES = 60      # 60일 미만 종목 제외
VOLUME_WINDOW = 20    # 직전 20일 평균 거래량
VOLUME_MULT = 3.0     # 3배 초과
DISP_MIN, DISP_MAX = 100.0, 118.0   # 20일 이격도(%)
TENKAN, KIJUN = 9, 26               # 일목균형표
AROON_N = 25
ELDER_EMA = 13
DONCHIAN_N = 20
PC_POS_MIN = 0.52     # 가정: "평균 0.52 이상"을 '0.52 이상'으로 해석
OBV_MA = 10
SONAR_N = 25
SONAR_SMOOTH = 9      # 가정: 제미나이가 소나의 EMA 평활 기간을 알려주지 않음
MSG_LIMIT = 1900      # 디스코드 메시지 한도 2000자
KST = timezone(timedelta(hours=9))
FUNNEL_LABELS = ["거래량·양봉", "이격도", "일목", "아론", "엘더", "채널", "OBV", "소나"]


def now_kst():
    return datetime.now(KST).replace(tzinfo=None)


def get_json(url, params=None, retries=4):
    """업비트 API 호출. 일시 오류나 429(요청 과다)는 잠깐 쉬고 다시 시도."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(1.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("API 요청 횟수 초과(429)")


def closed_only(df, now):
    """아직 마감되지 않은 일봉(진행 중인 오늘 봉)은 제거. 업비트 일봉은 09:00 시작, 다음날 09:00 마감."""
    start = pd.to_datetime(df["candle_date_time_kst"])
    return df[start + pd.Timedelta(days=1) <= pd.Timestamp(now)].reset_index(drop=True)


def fetch_closed_candles(market, now):
    data = get_json(CANDLE_API, {"market": market, "count": CANDLE_COUNT})
    if not data:
        return None
    df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)  # 업비트는 최신순 -> 시간순으로 뒤집기
    return closed_only(df, now)


def indicators(df):
    """마지막(신호) 봉 기준 지표값. 모두 신호 봉까지의 데이터만 사용."""
    c = df["trade_price"].astype(float)
    h = df["high_price"].astype(float)
    l = df["low_price"].astype(float)
    v = df["candle_acc_trade_volume"].astype(float)
    ind = {"disp20": c.iloc[-1] / c.iloc[-20:].mean() * 100.0}
    ind["tenkan"] = (h.iloc[-TENKAN:].max() + l.iloc[-TENKAN:].min()) / 2.0
    ind["kijun"] = (h.iloc[-KIJUN:].max() + l.iloc[-KIJUN:].min()) / 2.0
    hw = h.iloc[-(AROON_N + 1):].to_numpy()
    lw = l.iloc[-(AROON_N + 1):].to_numpy()
    since_high = int(np.argmax(hw[::-1]))   # 최근 고점이 몇 봉 전인지 (0 = 오늘)
    since_low = int(np.argmin(lw[::-1]))
    ind["aroon"] = 100.0 * (AROON_N - since_high) / AROON_N - 100.0 * (AROON_N - since_low) / AROON_N
    ema = c.ewm(span=ELDER_EMA, adjust=False).mean()
    ind["bear_power"] = l.iloc[-1] - ema.iloc[-1]
    hh, ll = h.iloc[-DONCHIAN_N:].max(), l.iloc[-DONCHIAN_N:].min()
    ind["pc_pos"] = (c.iloc[-1] - ll) / (hh - ll) if hh > ll else float("nan")
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    ind["obv"] = obv.iloc[-1]
    ind["obv_ma"] = obv.iloc[-OBV_MA:].mean()
    ind["sonar"] = (c - c.shift(SONAR_N)).dropna().ewm(span=SONAR_SMOOTH, adjust=False).mean().iloc[-1]
    return ind


def evaluate(df, market):
    """(걸리면 정보 dict 아니면 None, 앞에서부터 연속으로 통과한 조건 수). 데이터 부족이면 (None, -1)"""
    if df is None or len(df) < MIN_CANDLES:
        return None, -1
    last = df.iloc[-1]
    base = df["candle_acc_trade_volume"].iloc[-1 - VOLUME_WINDOW:-1].mean()  # 신호봉 직전 20일
    if not base or base <= 0:
        return None, 0
    ratio = last["candle_acc_trade_volume"] / base
    ind = indicators(df)
    checks = [
        last["trade_price"] > last["opening_price"] and ratio > VOLUME_MULT,
        DISP_MIN <= ind["disp20"] <= DISP_MAX,
        ind["tenkan"] > ind["kijun"],
        ind["aroon"] > 0,
        ind["bear_power"] > 0,
        ind["pc_pos"] >= PC_POS_MIN,   # 값이 NaN이면 False
        ind["obv"] > ind["obv_ma"],
        ind["sonar"] > 0,
    ]
    stage = 0
    for ok in checks:
        if not ok:
            break
        stage += 1
    if stage < len(checks):
        return None, stage
    return {
        "market": market,
        "price": float(last["trade_price"]),
        "ratio": float(ratio),
        "body": float(last["trade_price"] / last["opening_price"] - 1),
        "value": float(last["candle_acc_trade_price"]),
        "disp": float(ind["disp20"]),
        "pc_pos": float(ind["pc_pos"]),
        "start": str(last["candle_date_time_kst"]),
    }, stage


def price_text(x):
    if x >= 100:
        return f"{x:,.0f}"
    if x >= 1:
        return f"{x:,.2f}"
    return f"{x:.4f}"


def funnel_text(stages):
    """조건을 앞에서부터 차례로 모두 통과한 종목 수 (어느 조건에서 많이 걸러지는지 보기 위함)"""
    ev = [s for s in stages if s >= 0]
    parts = [f"대상 {len(ev)}개"]
    for k, name in enumerate(FUNNEL_LABELS, 1):
        parts.append(f"{name} {sum(1 for s in ev if s >= k)}")
    return "조건별 통과 수(누적): " + " → ".join(parts)


def build_messages(signals, latest_start, failed, funnel):
    day = pd.Timestamp(latest_start) if latest_start else None
    head = [TITLE]
    if day is not None:
        head.append(f"기준 일봉: {day:%m-%d} 09:00 ~ {day + pd.Timedelta(days=1):%m-%d} 09:00")
    if failed:
        head.append(f"(일부 종목 {len(failed)}개는 조회 실패)")
    if not signals:
        return ["\n".join(head + ["", "📭 오늘 조건에 맞는 종목 없음", funnel])]

    signals = sorted(signals, key=lambda s: s["ratio"], reverse=True)
    lines = [f"{i}. **{s['market']}**  종가 {price_text(s['price'])}원 / 거래량 {s['ratio']:.1f}배 / "
             f"일봉 {s['body']:+.1%} / 이격도 {s['disp']:.0f} / 채널위치 {s['pc_pos']:.2f} / "
             f"거래대금 {s['value'] / 1e8:,.1f}억"
             for i, s in enumerate(signals, 1)]
    foot = funnel + "\n※ 매수 신호가 아니라 조건에 걸린 목록입니다. 거래정지·유의, 글로벌 시세 괴리, 입금량 경고는 직접 확인하세요."

    msgs, cur = [], "\n".join(head + ["", f"✅ 포착 {len(signals)}개"])
    for ln in lines:
        if len(cur) + len(ln) + 1 > MSG_LIMIT:
            msgs.append(cur)
            cur = f"{TITLE} (이어서)\n{ln}"
        else:
            cur += "\n" + ln
    if len(cur) + len(foot) + 2 > MSG_LIMIT:
        msgs.append(cur)
        cur = f"{TITLE} (이어서)\n{foot}"
    else:
        cur += "\n\n" + foot
    msgs.append(cur)
    return msgs


def send(text):
    print(text)
    if not WEBHOOK:
        print("DISCORD_WEBHOOK_URL 이 설정되지 않아 디스코드 전송을 건너뜁니다.")
        return
    r = requests.post(WEBHOOK, json={"content": text}, timeout=15)
    print("디스코드 전송 상태:", r.status_code)
    r.raise_for_status()


def main():
    now = now_kst()
    markets = [m["market"] for m in get_json(MARKET_API) if m["market"].startswith("KRW-")]
    print(f"대상 KRW 마켓 수: {len(markets)}개")

    signals, failed, stages, latest_start = [], [], [], None
    for mk in markets:
        try:
            df = fetch_closed_candles(mk, now)
        except Exception as e:
            failed.append(mk)
            print(f"[{mk}] 캔들 조회 실패: {e}")
            time.sleep(0.12)
            continue
        if df is not None and len(df):
            s = str(df.iloc[-1]["candle_date_time_kst"])
            latest_start = s if latest_start is None or s > latest_start else latest_start
        hit, stage = evaluate(df, mk)
        stages.append(stage)
        if hit:
            signals.append(hit)
        time.sleep(0.12)  # 업비트 요청 제한 여유

    if len(failed) > max(5, len(markets) * 0.1):
        send(f"{TITLE}\n⚠️ 스캔 실패: 종목 {len(failed)}/{len(markets)}개 조회 실패 (업비트 접속 문제일 수 있음)")
        sys.exit(1)

    print(f"포착된 종목 수: {len(signals)}개")
    for m in build_messages(signals, latest_start, failed, funnel_text(stages)):
        send(m)


if __name__ == "__main__":
    main()
