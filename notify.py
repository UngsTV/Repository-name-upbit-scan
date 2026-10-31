#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""업비트 일봉으로 v6.5 후보를 찾아 디스코드로 알려줍니다.

후보가 없는 날도, 점수가 높은데 빠진 코인이 왜 빠졌는지(제외 규칙/점수 미달)를 같이 보여줍니다.
"""
import contextlib
import io
import os
import sys
import traceback

import numpy as np
import pandas as pd
import requests

import backtest_v64 as b

TITLE = "📊 [v6.5 후보 스캔]"
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
LIMIT = 1900  # 디스코드 메시지 한도 2000자

# 제외 규칙 이름표 (backtest_v64.candidate_mask 의 X2/X3/X6/X7)
X_LABEL = {
    "x2": "X2 당일 상승률 과다",
    "x3": "X3 이격도 과열",
    "x6": "X6 5일 누적 상승 과다",
    "x7": "X7 볼린저 %b 상단(0.95 이상)",
}


def send(text):
    print(text)
    if not WEBHOOK:
        print("DISCORD_WEBHOOK_URL 이 설정되지 않아 디스코드 전송을 건너뜁니다.")
        return
    if len(text) > LIMIT:
        text = text[:LIMIT] + "\n...(이하 생략)"
    r = requests.post(WEBHOOK, json={"content": text}, timeout=15)
    print("디스코드 전송 상태:", r.status_code)
    r.raise_for_status()


def price(x):
    if x >= 100:
        return f"{x:,.0f}"
    if x >= 1:
        return f"{x:,.2f}"
    return f"{x:.4f}"


def why_not(df, p, mask, score, rows):
    """후보가 아닌 코인들이 왜 빠졌는지. 제외 규칙을 하나씩 꺼 보고, 꺼서 후보가 되면 그 규칙 때문."""
    cut = p["cut"] if p["use_v1"] else p["cut"] * 0.85
    off = {k: b.candidate_mask(df, dict(p, **{k: 1e18}))[0] for k in X_LABEL}
    off_all = b.candidate_mask(df, dict(p, **{k: 1e18 for k in X_LABEL}))[0]
    out = {}
    for i in rows:
        hit = [X_LABEL[k] for k, mk in off.items() if mk[i]]
        if hit:
            out[i] = "제외: " + ", ".join(hit)
        elif off_all[i]:
            out[i] = "제외 규칙 여러 개에 걸림"
        elif score[i] < cut:
            out[i] = f"점수 미달 (컷 {cut:g}점, {cut - score[i]:g}점 부족)"
        else:
            out[i] = "게이트 미충족 (볼린저 하단 터치·투자심리 과매도 둘 다 아님)"
    return out


def build_message(data_dir="upbit_daily"):
    with contextlib.redirect_stdout(io.StringIO()):  # load_all 의 안내 문구는 숨김
        df = b.load_all(data_dir, 1e8)
    df = df.reset_index(drop=True)
    p = b.V65
    cut = p["cut"] if p["use_v1"] else p["cut"] * 0.85
    last = df["date"].max()
    nxt = last + pd.Timedelta(days=1)
    head = f"{TITLE}\n기준 일봉: {last:%m-%d} 09:00 ~ {nxt:%m-%d} 09:00"

    mask, score = b.candidate_mask(df, p)
    today = (df["date"] == last).values
    m = mask & today
    others = np.flatnonzero(today & df["liquid"].values & ~mask)  # 오늘 거래대금 조건은 되지만 후보가 아닌 코인
    others = others[np.argsort(-score[others], kind="stable")]

    def ref_line(i, reason):
        r = df.iloc[i]
        return (f"- {r['market']}  점수 {score[i]:.0f} / 이격도 {r['disp']:.0f} / %b {r['pctb']:.2f}\n"
                f"   → {reason}")

    if m.any():
        out = df.loc[m, ["market", "sig_close", "ret1", "sig_low"]].copy()
        out["score"] = score[m]
        out = out.sort_values("score", ascending=False)
        lines = [f"✅ **오늘 후보 {len(out)}개**", head.split("\n", 1)[1], ""]
        for i, r in enumerate(out.itertuples(), 1):
            lines.append(f"{i}. **{r.market}**  종가 {price(r.sig_close)}원 / 점수 {int(r.score)} / "
                         f"당일 {r.ret1:+.1%} / 손절선 {price(r.sig_low)}원")
        lines.insert(0, TITLE)
        hi = [i for i in others if score[i] >= cut][:5]
        if hi:
            why = why_not(df, p, mask, score, hi)
            lines += ["", "참고: 점수는 컷 이상인데 제외된 코인 (후보 아님)"]
            lines += [ref_line(i, why[i]) for i in hi]
        lines += ["", "※ 매수 신호가 아니라 후보 목록입니다. 글로벌 시세 괴리, 거래정지·유의, 입금량 경고는 직접 확인하세요."]
        return "\n".join(lines)

    top = list(others[:5])
    lines = [head, "", "📭 **오늘 후보 없음** (후보가 0개인 날도 많아요)"]
    if top:
        why = why_not(df, p, mask, score, top)
        lines += ["", "참고: 후보는 아니지만 점수 높은 상위 5개"]
        lines += [ref_line(i, why[i]) for i in top]
    lines += ["", "※ 매수 신호가 아닙니다. 위 코인은 후보가 아니라 참고용입니다."]
    return "\n".join(lines)


def main():
    try:
        msg = build_message()
    except Exception:
        err = traceback.format_exc()[-1500:]
        send(f"{TITLE}\n⚠️ **스캔 실패**\n```\n{err}\n```")
        sys.exit(1)
    send(msg)


if __name__ == "__main__":
    main()
