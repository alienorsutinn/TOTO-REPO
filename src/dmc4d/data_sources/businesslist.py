from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_HISTORY_URL = "https://www.businesslist.my/damacai/results/history"

HDRS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": BASE_HISTORY_URL,
}


_DRAW_URL_RE = re.compile(r"/damacai/draw/4d-damacai-(\d{4}-\d{2}-\d{2})-(\d+)")
_DRAWNO_RE = re.compile(r"\b(\d{4}/\d{2})\b")


def _month_iter(start_d: date, end_d: date) -> Iterable[str]:
    y, m = start_d.year, start_d.month
    while (y < end_d.year) or (y == end_d.year and m <= end_d.month):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            m = 1
            y += 1


@dataclass
class BusinessListResults:
    rate_per_sec: float = 1.0
    timeout_sec: int = 30

    def __post_init__(self) -> None:
        self._sleep = 0.0 if self.rate_per_sec <= 0 else 1.0 / float(self.rate_per_sec)
        self._sess = requests.Session()

    def _get(self, url: str) -> str:
        if self._sleep > 0:
            time.sleep(self._sleep)
        r = self._sess.get(url, headers=HDRS, timeout=self.timeout_sec)
        r.raise_for_status()
        return r.text or ""

    def _month_url(self, ym: str) -> str:
        # IMPORTANT: businesslist expects this exact query key formatting.
        return f"{BASE_HISTORY_URL}?data[Lottery][date]={ym}"

    def _discover_draw_urls_for_month(self, ym: str) -> list[str]:
        html = self._get(self._month_url(ym))
        soup = BeautifulSoup(html, "html.parser")

        urls: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if _DRAW_URL_RE.search(href):
                urls.add(urljoin(BASE_HISTORY_URL, href))

        # Deterministic order
        return sorted(urls)

    def _extract_numbers_ordered(self, soup: BeautifulSoup) -> list[str]:
        """
        Robust extraction:
        - prefer numbers that appear in td/li/span/div elements where the *entire* text is 4 digits
        - keep order of appearance
        """
        nums: list[str] = []

        # In practice, results often appear in tables.
        candidates = soup.find_all(["td", "li", "span", "div"])
        for el in candidates:
            t = el.get_text(" ", strip=True)
            if re.fullmatch(r"\d{4}", t):
                nums.append(t)

        # If too many (rare), still keep in order; we'll slice later after sanity checks.
        return nums

    def _parse_draw_page(self, url: str) -> dict | None:
        html = self._get(url)
        soup = BeautifulSoup(html, "html.parser")

        # date from URL
        m = _DRAW_URL_RE.search(url)
        if not m:
            return None
        date_str = m.group(1)

        # draw_no from page text (e.g., 5358/22)
        txt = soup.get_text(" ", strip=True)
        draw_no_m = _DRAWNO_RE.search(txt)
        draw_no = draw_no_m.group(1) if draw_no_m else ""

        # operator is always DMC for this scraper
        operator = "DMC"

        nums = self._extract_numbers_ordered(soup)

        # Expect exactly 23 numbers in order: 1st/2nd/3rd + 10 starter + 10 consolation.
        # If we got more than 23, try to take the *first* 23 (tables usually are ordered correctly).
        if len(nums) < 23:
            return None
        if len(nums) > 23:
            nums = nums[:23]

        top3 = nums[0:3]
        starter = nums[3:13]
        consolation = nums[13:23]

        # sanity: exact sizes
        if not (len(top3) == 3 and len(starter) == 10 and len(consolation) == 10):
            return None

        return {
            "date": date_str,
            "draw_no": draw_no,
            "operator": operator,
            "top3": top3,
            "starter": starter,
            "consolation": consolation,
        }

    def fetch_range(self, start_d: date, end_d: date) -> list[dict]:
        rows: list[dict] = []
        seen: set[str] = set()

        print(f" businesslist {start_d} -> {end_d} (rate_per_sec={self.rate_per_sec})", flush=True)
        months = list(_month_iter(start_d, end_d))
        print(
            f"[businesslist] months to fetch: {len(months)} ({months[0]} -> {months[-1]})",
            flush=True,
        )

        for ym in months:
            urls = self._discover_draw_urls_for_month(ym)
            print(f"[businesslist] month {ym}: found {len(urls)} 4D draw pages", flush=True)

            for u in urls:
                if u in seen:
                    continue
                seen.add(u)

                m = _DRAW_URL_RE.search(u)
                if not m:
                    continue
                d = date.fromisoformat(m.group(1))
                if d < start_d or d > end_d:
                    continue

                row = self._parse_draw_page(u)
                if row is None:
                    # keep going, don't explode
                    continue
                rows.append(row)

        print(f"[businesslist] rows in-range: {len(rows)}", flush=True)
        return rows


def results_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["date", "draw_no", "operator", "top3", "starter", "consolation"]
        )
    df = pd.DataFrame(rows)
    for col in ["top3", "starter", "consolation"]:
        df[col] = df[col].apply(lambda x: ",".join(x) if isinstance(x, list) else str(x))
    return df
