from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_HISTORY_URL = "https://www.businesslist.my/damacai/results/history"
BASE_SITE = "https://www.businesslist.my"


def _month_iter(start_d: date, end_d: date) -> Iterable[str]:
    cur_y, cur_m = start_d.year, start_d.month
    end_y, end_m = end_d.year, end_d.month
    while (cur_y, cur_m) <= (end_y, end_m):
        yield f"{cur_y:04d}-{cur_m:02d}"
        if cur_m == 12:
            cur_y += 1
            cur_m = 1
        else:
            cur_m += 1


def _parse_ym(ym: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{4})-(\d{2})", ym.strip())
    if not m:
        raise ValueError(f"Invalid ym: {ym}")
    return int(m.group(1)), int(m.group(2))


def _safe_zfill4(s: str) -> str:
    s = (s or "").strip()
    if re.fullmatch(r"\d{1,4}", s):
        return s.zfill(4)
    return s


@dataclass
class BusinessListResults:
    rate_per_sec: float = 0.5

    def __post_init__(self) -> None:
        self._session = requests.Session()
        self._last_req_at = 0.0
        self._min_interval = (
            1.0 / self.rate_per_sec if self.rate_per_sec and self.rate_per_sec > 0 else 0.0
        )
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-GB,en;q=0.9",
                "Connection": "keep-alive",
                "Referer": BASE_HISTORY_URL,
            }
        )

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        now = time.time()
        wait = (self._last_req_at + self._min_interval) - now
        if wait > 0:
            time.sleep(wait)
        self._last_req_at = time.time()

    def _get(self, url: str) -> str:
        self._throttle()
        try:
            r = self._session.get(url, timeout=30)
            if r.status_code != 200:
                logger.warning("GET %s -> %s", url, r.status_code)
                return ""
            return r.text or ""
        except Exception as e:
            logger.warning("GET failed %s: %s", url, e)
            return ""

    def _post(self, url: str, data: dict[str, str]) -> str:
        self._throttle()
        try:
            r = self._session.post(url, data=data, timeout=30)
            if r.status_code != 200:
                logger.warning("POST %s -> %s", url, r.status_code)
                return ""
            return r.text or ""
        except Exception as e:
            logger.warning("POST failed %s: %s", url, e)
            return ""

    def _month_url(self, ym: str, page: int | None = None) -> str:
        params = {"data[Lottery][date]": ym}
        if page and page > 1:
            params["page"] = str(page)
        return f"{BASE_HISTORY_URL}?{urlencode(params)}"

    def _extract_4d_links(self, html: str, ym: str) -> list[str]:
        """
        Extract only 4D Damacai draw links for the target month.
        """
        year, month = _parse_ym(ym)
        want_prefix = f"4d-damacai-{year}-{month:02d}-"

        soup = BeautifulSoup(html, "html.parser")
        urls: set[str] = set()

        for a in soup.select('a[href^="/damacai/draw/"]'):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            # Only 4d-damacai
            # href example: /damacai/draw/4d-damacai-2022-01-01-3436
            if "/damacai/draw/4d-damacai-" not in href:
                continue
            if want_prefix not in href:
                continue
            urls.add(BASE_SITE + href)

        return sorted(urls)

    def _discover_draw_urls_for_month(self, ym: str) -> list[str]:
        """
        Robust discovery: BusinessList history month filter is often a POSTed form.
        So we:
          1) GET the history page (establish cookies)
          2) POST form data {data[Lottery][date]: ym}
          3) Fallback: GET with querystring if POST fails
        """
        # Step 1: seed cookies/session
        seed = self._get(BASE_HISTORY_URL)
        if not seed:
            return []

        # Step 2: POST form-style (most reliable)
        html = self._post(BASE_HISTORY_URL, {"data[Lottery][date]": ym})
        urls = self._extract_4d_links(html, ym) if html else []

        if urls:
            return urls

        # Step 3: fallback to GET with querystring
        html2 = self._get(self._month_url(ym))
        urls2 = self._extract_4d_links(html2, ym) if html2 else []
        return urls2

    def _parse_draw_page(self, url: str) -> dict | None:
        html = self._get(url)
        if not html:
            return None

        low = html.lower()
        if "just a moment" in low or "access denied" in low or "page not found" in low:
            return None

        soup = BeautifulSoup(html, "html.parser")

        # Date is in URL
        m_url_date = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", url)
        if not m_url_date:
            text = soup.get_text(" ", strip=True)
            m_text_date = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
            if not m_text_date:
                return None
            draw_date = m_text_date.group(1)
        else:
            draw_date = m_url_date.group(1)

        # Draw number
        text = soup.get_text(" ", strip=True)
        m_draw = re.search(r"\bDRAW\s+(\d{4}/\d{2})\b", text, flags=re.I)
        draw_no = m_draw.group(1) if m_draw else ""

        # Extract prize nodes using data-lotto-n
        nodes = soup.select(".l_prize[data-lotto-t][data-lotto-n]")
        if not nodes:
            return None

        buckets: dict[str, list[str]] = {}
        for n in nodes:
            t = (n.get("data-lotto-t") or "").strip()
            v = (n.get("data-lotto-n") or "").strip()
            if not t or not v:
                continue
            v = _safe_zfill4(v)
            buckets.setdefault(t, []).append(v)

        top1 = buckets.get("1st Prize", [])
        top2 = buckets.get("2nd Prize", [])
        top3 = buckets.get("3rd Prize", [])
        top = [top1[0] if top1 else "", top2[0] if top2 else "", top3[0] if top3 else ""]

        starter = buckets.get("Starter", [])
        consolation = buckets.get("Consolidate", []) or buckets.get("Consolation", [])

        if any(not re.fullmatch(r"\d{4}", x) for x in top):
            return None
        if len(starter) != 10:
            return None
        if len(consolation) != 10:
            return None

        return {
            "date": draw_date,
            "draw_no": draw_no,
            "operator": "DMC",
            "top3": top,
            "starter": starter,
            "consolation": consolation,
        }

    def fetch_range(self, start_d: date, end_d: date) -> list[dict]:
        rows: list[dict] = []
        seen: set[str] = set()

        print(f"[businesslist] {start_d} -> {end_d} (rate_per_sec={self.rate_per_sec})", flush=True)
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

                row = self._parse_draw_page(u)
                if not row:
                    continue

                try:
                    rd = datetime.strptime(row["date"], "%Y-%m-%d").date()
                except Exception:
                    continue

                if start_d <= rd <= end_d:
                    rows.append(row)

        print(f"[businesslist] rows in-range: {len(rows)}", flush=True)
        return rows
