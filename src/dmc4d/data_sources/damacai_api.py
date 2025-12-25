from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup


# Typical MY 4D draw days: Wed, Sat, Sun (Python weekday Mon=0)
DRAWDAYS = {2, 5, 6}


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    txt = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", txt)


def first_match(patterns: list[str], text: str, flags: int = re.I) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return m.group(1)
    return None


def parse_draw_block(txt: str, fallback_date: date) -> Optional[dict]:
    """
    Extract Big (4-digit) results:
      top3 (4-digit), starter(10), consolation(10)
    Many pages also include Small (3-digit). We ignore those.
    """
    # If date isn't present in the response, use the requested date.
    dmy = first_match(
        [
            r"Draw\s*Date\s*:\s*(\d{2}/\d{2}/\d{4})",
            r"开彩日期[:：]?\s*(\d{2}/\d{2}/\d{4})",
        ],
        txt,
    )
    if dmy:
        draw_date = datetime.strptime(dmy, "%d/%m/%Y").date()
    else:
        draw_date = fallback_date

    draw_no = first_match(
        [
            r"Draw\s*No\.?\s*:\s*([0-9]+/[0-9]+)",
            r"开彩期数\.?[:：]?\s*([0-9]+/[0-9]+)",
        ],
        txt,
    ) or fallback_date.strftime("%y%m%d")

    # Top 3 (Big) prizes
    p1 = first_match([r"1st\s*Prize\s*([0-9]{4})", r"首奖\s*([0-9]{4})"], txt)
    p2 = first_match([r"2nd\s*Prize\s*([0-9]{4})", r"二奖\s*([0-9]{4})"], txt)
    p3 = first_match([r"3rd\s*Prize\s*([0-9]{4})", r"三奖\s*([0-9]{4})"], txt)
    if not (p1 and p2 and p3):
        return None

    # Starter + Consolation blocks
    lower = txt.lower()

    def take_after(label_patterns: list[str], n: int) -> Optional[list[str]]:
        for lp in label_patterns:
            m = re.search(lp, lower)
            if not m:
                continue
            chunk = txt[m.end() : m.end() + 1500]
            nums = re.findall(r"\b\d{4}\b", chunk)
            if len(nums) >= n:
                return nums[:n]
        return None

    starter = take_after([r"starter prizes", r"starter prize", r"奖头", r"入围奖"], 10)
    consolation = take_after([r"consolation prizes", r"consolation prize", r"安慰奖"], 10)

    if not starter or not consolation:
        return None

    return {
        "date": draw_date.isoformat(),
        "draw_no": draw_no,
        "operator": "DMC",
        "top3": [p1, p2, p3],
        "starter": starter,
        "consolation": consolation,
    }


@dataclass
class DamacaiAPIResults:
    """
    Fetch via:
      https://www.damacai.com.my/callpassresult?pastdate=YYMMDD
    """

    rate_per_sec: float = 1.0
    timeout_s: int = 30

    def fetch_one(self, d: date) -> Optional[dict]:
        if d.weekday() not in DRAWDAYS:
            return None

        yymmdd = d.strftime("%y%m%d")
        url = f"https://www.damacai.com.my/callpassresult?pastdate={yymmdd}"

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; dmc4d-lab/0.1)",
            "Referer": "https://www.damacai.com.my/past-draw-result/",
            "Accept": "*/*",
        }

        r = requests.get(url, headers=headers, timeout=self.timeout_s)
        if r.status_code != 200 or not r.text:
            return None

        txt = clean_text(r.text)
        row = parse_draw_block(txt, fallback_date=d)
        return row

    def fetch_range(self, start: date, end: date) -> list[dict]:
        rows: list[dict] = []
        sleep_s = 1.0 / max(self.rate_per_sec, 0.1)

        for d in daterange(start, end):
            row = self.fetch_one(d)
            if row:
                rows.append(row)
            time.sleep(sleep_s)

        return rows
