#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""backtest_v64.py scan 을 실행하고 결과를 디스코드 웹훅으로 보냅니다."""
import os
import subprocess
import sys

import requests

WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
LIMIT = 1900  # 디스코드 메시지 한도 2000자 (코드블록 표시 여유분 포함)


def send(text):
    if not WEBHOOK:
        print("DISCORD_WEBHOOK_URL 이 설정되지 않아 디스코드 전송을 건너뜁니다.")
        return
    if len(text) > LIMIT:
        text = text[:LIMIT] + "\n...(이하 생략)"
    r = requests.post(WEBHOOK, json={"content": f"```\n{text}\n```"}, timeout=15)
    print("디스코드 전송 상태:", r.status_code)
    r.raise_for_status()


def main():
    p = subprocess.run(
        [sys.executable, "backtest_v64.py", "scan"],
        capture_output=True, text=True, encoding="utf-8",
    )
    out = (p.stdout or "").strip()
    print(out)
    if p.returncode != 0:
        err = (p.stderr or "").strip()[-1500:]
        send(f"[오류] scan 실행 실패\n{err}")
        sys.exit(p.returncode)
    send(out)


if __name__ == "__main__":
    main()
