#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""업비트 일봉으로 v6.5 후보를 찾아 디스코드로 짧게 알려줍니다."""
import contextlib
import io
import os
import sys
import traceback

import pandas as pd
import requests

import backtest_v64 as b

WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
LIMIT = 1900  # 디스코드 메시지 한도 2000자


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


def build_message(data_dir="upbit_daily"):
    with contextlib.redirect_stdout(io.StringIO()):  # load_all 의 안내 문구는 숨김
        df = b.load_all(data_dir, 1e8)
    last = df["date"].max()
    nxt = last + pd.Timedelta(days=1)
    head_day = f"기준 일봉: {last:%m-%d} 09:00 ~ {nxt:%m-%d} 09:00"

    mask, score = b.candidate_mask(df, b.V65)
    today = (df["date"] == last).values
    m = mask & today

    if not m.any():
        liquid = today & df["liquid"].values
        best = f"{int(score[liquid].max())}점" if liquid.any() else "-"
        return (f"📭 **오늘 후보 없음**\n{head_day}\n"
                f"(점수 최고 {best} / 후보 컷 {b.V65['cut']}점. 후보가 0개인 날도 많아요)")

    out = df.loc[m, ["market", "sig_close", "ret1", "sig_low"]].copy()
    out["score"] = score[m]
    out = out.sort_values("score", ascending=False)
    lines = [f"✅ **오늘 후보 {len(out)}개**", head_day, ""]
    for i, r in enumerate(out.itertuples(), 1):
        lines.append(f"{i}. **{r.market}**  종가 {price(r.sig_close)}원 / 점수 {int(r.score)} / "
                     f"당일 {r.ret1:+.1%} / 손절선 {price(r.sig_low)}원")
    lines += ["", "※ 매수 신호가 아니라 후보 목록입니다. 글로벌 시세 괴리, 거래정지·유의, "
                  "입금량 경고는 직접 확인하세요."]
    return "\n".join(lines)


def main():
    try:
        msg = build_message()
    except Exception:
        err = traceback.format_exc()[-1500:]
        send(f"⚠️ **스캔 실패**\n```\n{err}\n```")
        sys.exit(1)
    send(msg)


if __name__ == "__main__":
    main()
