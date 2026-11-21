"""매일 아침 거시경제 + 코인 뉴스 요약 -> 디스코드
- 뉴스 수집: RSS / 네이버 검색 API (웹검색 X, 그냥 HTTP 요청이라 한도 없음)
- 요약: Claude API 호출 딱 2번 (거시 1회, 코인 1회)
- 피드 하나가 죽어도 나머지로 계속 진행
"""
import os, re, html, time, datetime as dt
import calendar
import requests, feedparser
try:
    import anthropic
except ImportError:
    anthropic = None

MODEL = "claude-haiku-5-5"          # 싸고 빠름. 품질 더 원하면 claude-sonnet-5-5
HOURS = 26                            # 최근 N시간 기사만
PER_FEED = 12                         # 피드당 최대 기사 수
KST = dt.timezone(dt.timedelta(hours=9))

# ── 뉴스 소스 (이름, URL) ─────────────────────────────
MACRO_FEEDS = {
    "해외": [
        ("CNN 최신", "http://rss.cnn.com/rss/edition.rss"),
        ("CNN 월드", "http://rss.cnn.com/rss/edition_world.rss"),
        ("CNN 비즈니스", "http://rss.cnn.com/rss/money_latest.rss"),
    ],
    "국내": [
        ("연합뉴스 속보", "https://www.yna.co.kr/rss/news.xml"),
        ("연합뉴스 경제", "https://www.yna.co.kr/rss/economy.xml"),
        ("연합뉴스 마켓+", "https://www.yna.co.kr/rss/market.xml"),
    ],
}
NAVER_QUERIES = ["경제 속보", "금리 환율", "코스피"]   # 네이버 검색 API용 (키 있을 때만)

CRYPTO_FEEDS = [
    ("블록미디어", "https://www.blockmedia.co.kr/feed"),
    ("토큰포스트", "https://www.tokenpost.kr/rss"),
    ("코인데스크", "https://www.coindesk.com/arc/outboundfeeds/rss"),
    ("코인텔레그래프", "https://cointelegraph.com/rss"),
    ("코인니스", os.getenv("COINNESS_RSS", "")),   # 공식 RSS 확인되면 URL 넣기 (없으면 자동 건너뜀)
]

UA = {"User-Agent": "Mozilla/5.0 (news-digest-bot)"}
clean = lambda s: re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def fetch_feed(name, url):
    if not url:
        return []
    try:
        r = requests.get(url, headers=UA, timeout=20)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
    except Exception as e:
        print(f"[skip] {name}: {e}")
        return []
    cutoff = time.time() - HOURS * 3600
    out = []
    for e in feed.entries:
        ts = e.get("published_parsed") or e.get("updated_parsed")
        t = calendar.timegm(ts) if ts else 0
        if t and t < cutoff:
            continue
        out.append({
            "src": name,
            "title": clean(e.get("title")),
            "desc": clean(e.get("summary"))[:200],
            "ts": t,
            "link": e.get("link", ""),
        })
    out.sort(key=lambda x: x["ts"], reverse=True)
    print(f"[ok] {name}: {len(out)}건")
    return out[:PER_FEED]


def fetch_naver():
    cid, sec = os.getenv("NAVER_CLIENT_ID"), os.getenv("NAVER_CLIENT_SECRET")
    if not (cid and sec):
        print("[skip] 네이버: API 키 없음")
        return []
    out = []
    for q in NAVER_QUERIES:
        try:
            r = requests.get(
                "https://openapi.naver.com/v1/search/news.json",
                headers={"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": sec},
                params={"query": q, "display": 8, "sort": "date"}, timeout=20)
            r.raise_for_status()
            for it in r.json().get("items", []):
                out.append({"src": "네이버", "title": clean(it["title"]),
                            "desc": clean(it["description"])[:200], "ts": 0,
                            "link": it.get("originallink") or it.get("link", "")})
        except Exception as e:
            print(f"[skip] 네이버 {q}: {e}")
    print(f"[ok] 네이버: {len(out)}건")
    return out


def dedupe(items):
    seen, res = set(), []
    for i in items:
        k = re.sub(r"\W", "", i["title"])[:25]
        if k and k not in seen:
            seen.add(k); res.append(i)
    return res


def to_text(items):
    return "\n".join(f"- [{i['src']}] {i['title']}" + (f" :: {i['desc']}" if i["desc"] else "") for i in items)


def summarize(system, content):
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수 사용
    msg = client.messages.create(
        model=MODEL, max_tokens=1500, system=system,
        messages=[{"role": "user", "content": content}])
    return "".join(b.text for b in msg.content if b.type == "text").strip()


MACRO_SYS = """너는 거시경제 브리핑 담당이다. 아래 기사 제목/요약만 근거로 한국어로 작성한다.
규칙: 주제별(예: 미국·글로벌, 금리·환율, 국내 증시·경제, 지정학·원자재 등 기사에 있는 주제만)로 묶고,
각 주제는 최대 5줄 이내 불릿으로 요약. 숫자·사건 위주, 추측·투자권유 금지.
마지막에 '오늘 체크포인트' 한 줄. 기사에 없는 내용은 지어내지 말 것."""

CRYPTO_SYS = """너는 코인 시장 뉴스 브리핑 담당이다. 아래 기사 제목/요약만 근거로 한국어로 정확히 5줄 요약한다.
규칙: 가장 중요한 이슈 순서, 각 줄은 한 문장, 추측·투자권유 금지, 기사에 없는 내용 지어내지 말 것."""



TOPICS = {
    "💵 금리·환율": ["금리", "환율", "달러", "연준", "Fed", "FOMC", "rate", "Treasury", "dollar", "yield", "인플레", "inflation", "CPI"],
    "📈 증시": ["코스피", "코스닥", "증시", "주가", "나스닥", "S&P", "Dow", "stock", "market", "Nasdaq", "wall street"],
    "🛢 원자재·에너지": ["유가", "원유", "금값", "oil", "gold", "OPEC", "gas"],
    "🌍 지정학·무역": ["전쟁", "관세", "트럼프", "중국", "러시아", "이란", "tariff", "Trump", "China", "Russia", "war", "Israel", "Ukraine"],
    "🏠 국내 경제": ["부동산", "수출", "물가", "고용", "예산", "한국", "금융위", "한은", "기재부"],
}


def headline_digest(items, per_topic=4, others=5, with_src=True):
    """API 없이 제목만 주제(키워드)별로 묶어서 전송"""
    def fmt(i):
        link = f" <{i['link']}>" if i.get("link") else ""
        return f"• {i['title']}" + (f" ({i['src']})" if with_src else "") + link
    used, lines = set(), []
    for topic, kws in TOPICS.items():
        hit = [i for n, i in enumerate(items) if n not in used and any(k.lower() in i["title"].lower() for k in kws)][:per_topic]
        for i in hit:
            used.add(items.index(i))
        if hit:
            lines.append(f"**{topic}**\n" + "\n".join(fmt(i) for i in hit))
    rest = [i for n, i in enumerate(items) if n not in used][:others]
    if rest:
        lines.append("**📰 기타**\n" + "\n".join(fmt(i) for i in rest))
    return "\n\n".join(lines)


def discord(text):
    url = (os.getenv("DISCORD_WEBHOOK_URL") or os.getenv("NEWS_DISCORD_WEBHOOK_URL") or "").strip()
    if not url.startswith("https://"):
        raise SystemExit("❌ 웹훅 URL이 비어 있음: news-digest.yml의 env 줄이 secrets.DISCORD_WEBHOOK_URL 로 저장(커밋)됐는지 확인")
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 1900:
            chunks.append(cur); cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    for c in chunks:
        requests.post(url, json={"content": c}, timeout=20).raise_for_status()
        time.sleep(0.5)


def main():
    today = dt.datetime.now(KST).strftime("%m/%d(%a)")

    macro = []
    for group in MACRO_FEEDS.values():
        for name, url in group:
            macro += fetch_feed(name, url)
    macro += fetch_naver()
    macro = dedupe(macro)

    crypto = []
    for name, url in CRYPTO_FEEDS:
        crypto += fetch_feed(name, url)
    crypto = dedupe(crypto)

    use_ai = bool(os.getenv("ANTHROPIC_API_KEY"))   # 키 없으면 제목만 모드
    msgs = []
    if macro:
        body = summarize(MACRO_SYS, to_text(macro)) if use_ai else headline_digest(macro)
        msgs.append(f"📊 **[거시경제 브리핑] {today}**\n" + body)
    else:
        msgs.append("📊 거시경제: 수집된 기사가 없습니다 (피드 점검 필요)")
    if crypto:
        body = summarize(CRYPTO_SYS, to_text(crypto)) if use_ai else "\n".join(
            f"• {i['title']} ({i['src']})" + (f" <{i['link']}>" if i.get("link") else "") for i in crypto[:8])
        msgs.append(f"🪙 **[코인 뉴스] {today}**\n" + body)
    else:
        msgs.append("🪙 코인: 수집된 기사가 없습니다 (피드 점검 필요)")

    discord("\n\n".join(msgs))
    print("done")


if __name__ == "__main__":
    main()
