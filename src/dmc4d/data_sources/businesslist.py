from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def _iso_from_ddmmyyyy(s: str) -> str:
    dt = datetime.strptime(s, "%d/%m/%Y")
    return dt.strftime("%Y-%m-%d")


def _extract_draw_date_from_url(url: str) -> date | None:
    """
    BusinessList draw URLs usually look like:
      .../4d-damacai-YYYY-MM-DD-<id>
    """
    m = re.search(r"4d-damacai-(\d{4})-(\d{2})-(\d{2})-", url)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return date(y, mo, d)


def _iter_months(start_d: date, end_d: date) -> list[str]:
    months: list[str] = []
    y, m = start_d.year, start_d.month
    while (y < end_d.year) or (y == end_d.year and m <= end_d.month):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


@dataclass
class BusinessListResults:
    rate_per_sec: float = 1.0
    timeout: int = 30

    def __post_init__(self) -> None:
        self._session = requests.Session()
        self._last_req_ts = 0.0

    @property
    def _min_interval(self) -> float:
        if self.rate_per_sec <= 0:
            return 0.0
        return 1.0 / self.rate_per_sec

    def _sleep_if_needed(self) -> None:
        now = time.time()
        wait = self._min_interval - (now - self._last_req_ts)
        if wait > 0:
            time.sleep(wait)

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self._sleep_if_needed()
        headers = kwargs.pop("headers", {}) or {}
        headers.setdefault("User-Agent", "Mozilla/5.0")
        r = self._session.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
        self._last_req_ts = time.time()
        r.raise_for_status()
        return r

    def _discover_month_urls(self) -> dict[str, str]:
        """
        Month pages:
          https://www.businesslist.my/damacai/results/history?data[Lottery][date]=YYYY-MM
        """
        base = "https://www.businesslist.my/damacai/results/history"
        # build from first available month on site if needed, but we can just generate dynamically
        # (caller filters by date range anyway)
        month_urls: dict[str, str] = {}

        # We keep it lazy: caller supplies months; we just format URL.
        # This function is retained for compatibility with earlier code structure.
        def working(ym: str) -> str:
            return f"{base}?data[Lottery][date]={ym}"

        # Generate a wide set (2018..current+1) so earlier debug scripts still show months.
        # fetch_range will only use the months it needs.
        current = date.today()
        for y in range(2018, current.year + 1):
            for m in range(1, 13):
                ym = f"{y:04d}-{m:02d}"
                month_urls[ym] = working(ym)
        # Add current month explicitly (already covered), but harmless.

        return month_urls

    def _discover_draw_urls_for_month(self, month_url: str) -> list[str]:
        r = self._request("GET", month_url)
        soup = BeautifulSoup(r.text or "", "lxml")
        urls: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/damacai/draw/4d-damacai-" in href:
                urls.append(urljoin(month_url, href))
        # Unique while preserving order
        out: list[str] = []
        seen: set[str] = set()
        for u in urls:
            if u not in seen:
                out.append(u)
                seen.add(u)
        return out

    def _extract_number_list(self, text: str, headings: list[str], n: int) -> list[str]:
        """
        Find a heading, then grab next ~4000 chars, extract 4-digit numbers.
        """
        lower = text.lower()
        for h in headings:
            pos = lower.find(h.lower())
            if pos == -1:
                continue
            window = text[pos : pos + 4000]
            nums = re.findall(r"\b\d{4}\b", window)
            out: list[str] = []
            for x in nums:
                if x not in out:
                    out.append(x)
            if len(out) >= n:
                return out[:n]
        return []

    def _parse_draw_page(self, url: str) -> dict | None:
        r = self._request("GET", url)
        soup = BeautifulSoup(r.text or "", "lxml")
        text = soup.get_text(" ", strip=True)

        # --- DATE (try text, else URL) ---
        iso_date: str | None = None
        for dp in [
            r"Draw\s*Date\s*:\s*(\d{2}/\d{2}/\d{4})",
            r"Date\s*:\s*(\d{2}/\d{2}/\d{4})",
            r"Draw\s*On\s*:\s*(\d{2}/\d{2}/\d{4})",
        ]:
            mm = re.search(dp, text, re.I)
            if mm:
                iso_date = _iso_from_ddmmyyyy(mm.group(1))
                break
        if iso_date is None:
            d = _extract_draw_date_from_url(url)
            if d is not None:
                iso_date = d.strftime("%Y-%m-%d")
        if iso_date is None:
            return None

        # --- DRAW NO (try text, else BL-id) ---
        draw_no: str | None = None
        mm = re.search(r"\b(\d{3,5}/\d{2})\b", text)
        if mm:
            draw_no = mm.group(1)
        if draw_no is None:
            mm2 = re.search(r"-(\d+)$", url)
            draw_no = f"BL-{mm2.group(1)}" if mm2 else f"BL-{iso_date}"

        # --- NUMBERS ---
        top3 = self._extract_number_list(
            text,
            headings=["1st Prize", "First Prize", "1st", "First"],
            n=3,
        )
        starter = self._extract_number_list(
            text,
            headings=["Starter Prizes", "Starter Prize", "Starter"],
            n=10,
        )
        consolation = self._extract_number_list(
            text,
            headings=["Consolation Prizes", "Consolation Prize", "Consolation"],
            n=10,
        )

        if len(top3) != 3 or len(starter) != 10 or len(consolation) != 10:
            return None

        return {
            "date": iso_date,
            "draw_no": draw_no,
            "operator": "DMC",
            "top3": top3,
            "starter": starter,
            "consolation": consolation,
        }

    def fetch_range(self, start_d: date, end_d: date) -> list[dict]:
        months = _iter_months(start_d, end_d)

        # Build month URLs directly (more reliable than scraping “history” index structure)
        base = "https://www.businesslist.my/damacai/results/history?data[Lottery][date]="

        print(
            f"[businesslist] months to fetch: {len(months)} ({months[0]} -> {months[-1]})",
            flush=True,
        )

        draw_urls: list[str] = []
        for ym in months:
            month_url = f"{base}{ym}"
            urls = self._discover_draw_urls_for_month(month_url)
            print(f"[businesslist] month {ym}: found {len(urls)} 4D draw pages", flush=True)
            draw_urls.extend(urls)

        # Filter draw URLs by date in URL first (fast)
        in_range_urls: list[str] = []
        for u in draw_urls:
            d = _extract_draw_date_from_url(u)
            if d is None:
                continue
            if start_d <= d <= end_d:
                in_range_urls.append(u)

        # Unique while preserving order
        uniq: list[str] = []
        seen: set[str] = set()
        for u in in_range_urls:
            if u not in seen:
                uniq.append(u)
                seen.add(u)

        print(f"[businesslist] total unique draw pages (in range): {len(uniq)}", flush=True)

        rows: list[dict] = []
        fail = 0
        for i, u in enumerate(uniq, 1):
            try:
                row = self._parse_draw_page(u)
            except Exception:
                row = None
            if row is None:
                fail += 1
                if fail <= 50:
                    print(f"[businesslist] ({i}/{len(uniq)}) parse failed: {u}", flush=True)
                continue
            rows.append(row)
            if i % 50 == 0:
                print(f"[businesslist] progress: {i}/{len(uniq)} parsed={len(rows)}", flush=True)

        print(f"[businesslist] parsed {len(rows)}/{len(uniq)}", flush=True)
        return rows
