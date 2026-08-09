"""
FbrefPlayerProvider — FBRef squad + player season stats (richest source).

League squad pages provide team IDs; each team's stats page has the
"Standard" (attack/possession) and "Keeper" tables with per-player season
totals including xG, xAG, shots, tackles, saves — everything the ratings
need.

Fetch layer: FBRef sits behind a Cloudflare JS challenge, so plain
requests / cloudscraper get 403'd. We use botasaurus with a real Chrome
window (headless=False, same approach as ScraperFC) and batch page
fetches per league to minimize browser sessions. Respects the
sports-reference 6s-per-request etiquette between navigations.

Season format in: '2425'  →  FBRef slug '2024-2025'.
"""

import logging
import re
import time
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd

from botasaurus.browser import browser, ElementWithSelectorNotFoundException

from data.player_providers.base import BasePlayerProvider
from data.player_providers.schema import PlayerRecord

logger = logging.getLogger(__name__)

# data/config.yaml league code → FBRef competition id (stable across seasons)
_LEAGUE_COMPS = {"E0": 9, "SP1": 12, "D1": 20, "I1": 11, "F1": 13}

_LEAGUE_SLUGS = {
    "E0": "Premier-League", "SP1": "La-Liga", "D1": "Bundesliga",
    "I1": "Serie-A", "F1": "Ligue-1",
}

# FBRef position tokens → our coarse bucket
_POS_MAP = [
    ("GK", "GK"),
    ("DF", "DF"), ("DEF", "DF"),
    ("MF", "MF"), ("MID", "MF"),
    ("FW", "FW"), ("ATT", "FW"), ("FWD", "FW"),
]

def _pos_tokens(pos):
    if not pos:
        return []
    return [p.strip().upper() for p in str(pos).split(",") if p.strip()]

# canonical team names → FBRef slug (names FBRef spells differently)
_TEAM_ALIASES = {
    "Inter": "Internazionale",
    "Bayern Munich": "Bayern Munich",
    "Paris Saint Germain": "Paris-Saint-Germain",
    "Man City": "Manchester-City",
    "Manchester Utd": "Manchester-United",
    "Leeds": "Leeds-United",
    "Wolves": "Wolverhampton-Wanderers",
}


def _season_slug(season: str) -> str:
    return f"20{season[:2]}-20{season[2:]}"


def _norm(s: str) -> str:
    return re.sub(r"[\s\-.'’]", "", s).lower()


def _coarse_position(pos):
    """Primary bucket from the full FBRef position list.

    FBRef lists wingers as 'MF,FW' with MF first — those belong in the FW
    bucket. Any other multi-position list (e.g. 'MF,DF' Valverde, 'DF,MF'
    Tchouameni) keeps FBRef's first token as the primary position.
    """
    tokens = _pos_tokens(pos)
    if not tokens:
        return None
    if "FW" in tokens and len(tokens) > 1:
        return "FW"
    if "GK" in tokens:
        return "GK"
    for token, bucket in _POS_MAP:
        if token == tokens[0]:
            return bucket
    return None


@browser(headless=False, block_images_and_css=False,
         wait_for_complete_page_load=False,
         output=None, create_error_logs=False)
def _browser_fetch(driver, urls):
    """Visit every URL in one Chrome session (6s etiquette delay between).

    Reloads through the Cloudflare challenge until <body class="fb"> shows.
    Returns list of page HTML strings, same order as urls (None on failure).
    """
    pages = []
    for url in urls:
        html = None
        for _ in range(6):
            try:
                driver.google_get(url)
                driver.wait_for_element("body.fb", wait=15)
                html = driver.page_html
                break
            except ElementWithSelectorNotFoundException:
                driver.reload()
            except Exception as e:
                logger.warning("FBRef fetch failed for %s: %s", url, e)
                break
        pages.append(html)
        if url is not urls[-1]:
            time.sleep(6)
    return pages


class FbrefPlayerProvider(BasePlayerProvider):
    provider_name = "fbref"

    def __init__(self, rate_limit_sec: float = 6.0, timeout: int = 30):
        self._delay = rate_limit_sec
        self._timeout = timeout
        self._last_fetch = 0.0
        self._team_ids: Dict[str, Dict[str, str]] = {}  # league_code -> {norm_name: team_id}

    def capabilities(self) -> Dict[str, bool]:
        # xG/xAG moved behind the Stathead paywall (2025) — understat covers it
        return dict(position=True, age=True, minutes=True, goals=True,
                    assists=True, xg=False, xa=False, shots=True,
                    shots_on_target=True, tackles=True, saves=True,
                    clean_sheets=True, goals_conceded=True, cards=True)

    # ── HTTP with etiquette ───────────────────────────────────────────────────

    def _get(self, url: str) -> str:
        return self._get_many([url])[0]

    def _get_many(self, urls: List[str]) -> List[str]:
        elapsed = time.time() - self._last_fetch
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        # botasaurus maps over the input list — wrap so ALL urls are one task
        pages = _browser_fetch([urls])[0]
        self._last_fetch = time.time()
        return pages or []

    # ── Team id discovery (league standings page, once per league) ────────────

    def _load_team_ids(self, league_codes: List[str], season: str) -> Dict[str, Dict[str, str]]:
        """Fetch every league page in ONE browser session; cache id maps."""
        missing = [lc for lc in league_codes if lc not in self._team_ids]
        if not missing:
            return self._team_ids
        urls = []
        for lc in missing:
            comp = _LEAGUE_COMPS.get(lc)
            slug = _LEAGUE_SLUGS.get(lc)
            if not comp or not slug:
                raise ValueError(f"FBRef league code not mapped: {lc}")
            urls.append(f"https://fbref.com/en/comps/{comp}/"
                        f"{_season_slug(season)}/{slug}-Stats")
        for lc, html in zip(missing, self._get_many(urls)):
            ids = {}
            if html:
                # only the standings table — the nav/footer list other leagues' clubs
                m = re.search(r'<table[^>]*?id="(results[^"]*?_overall)"', html)
                if m:
                    start = html.find('id="' + m.group(1) + '"')
                    end = html.find("</table>", start)
                    region = html[start:end]
                    for tm in re.finditer(
                            r'href="/en/squads/([0-9a-f]{8})/'
                            r'[^"]+?/([^"/]+?)-Stats"', region):
                        team_id, team_slug = tm.group(1), tm.group(2)
                        ids[_norm(team_slug.replace("-", " "))] = team_id
            self._team_ids[lc] = ids
            logger.info("FBRef %s: discovered %d team ids", lc, len(ids))
        return self._team_ids

    # ── Squad page parsing ────────────────────────────────────────────────────

    @staticmethod
    def _parse_stats_table(html: str, require_cols: List[str]) -> Optional[pd.DataFrame]:
        """First stats table containing ALL required columns.

        FBRef tables carry a two-level header (category, stat name) that we
        collapse to the stat-name level. Duplicated labels (e.g. the per-90
        group repeats Gls/Ast) are dropped keeping the first occurrence,
        which holds the season totals.
        """
        try:
            frames = pd.read_html(StringIO(html))
        except (ValueError, OSError):
            return None
        for df in frames:
            cols = [str(c[-1] if isinstance(c, tuple) else c) for c in df.columns]
            if not all(r in cols for r in require_cols):
                continue
            df = df.copy()
            df.columns = cols
            df = df.loc[:, ~df.columns.duplicated()]
            return df
        return None

    @staticmethod
    def _num(v, default=None):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        s = str(v).replace(",", "").replace("+", "").strip()
        # FBRef age cells look like "24-196" (years-days)
        if "-" in s and s.split("-")[0].strip().isdigit():
            s = s.split("-")[0].strip()
        try:
            return float(s)
        except (ValueError, TypeError):
            return default

    def fetch_team_squad(self, team: str, league_code: str, season: str) -> List[PlayerRecord]:
        return self.fetch_squads([(team, league_code)], season).get(team, [])

    def fetch_squads(self, teams, season: str) -> Dict[str, List[PlayerRecord]]:
        """Fetch squads for many (team, league_code) pairs.

        Batches all league pages into one browser session and all team
        stats pages into another — much cheaper than one window each.
        Returns {team_name: [PlayerRecord, ...]}.
        """
        league_codes = sorted({lc for _, lc in teams})
        ids = self._load_team_ids(league_codes, season)

        # Build per-team stats URLs (one browser session)
        team_urls, keys = [], []
        for team, league_code in teams:
            slug = _TEAM_ALIASES.get(team, team).replace(" ", "-")
            team_id = ids.get(league_code, {}).get(_norm(slug.replace("-", " ")))
            if not team_id:
                logger.warning("FBRef: team '%s' not found in league %s", team, league_code)
                continue
            url = (f"https://fbref.com/en/squads/{team_id}/"
                   f"{_season_slug(season)}/{slug}-Stats")
            team_urls.append(url)
            keys.append((team, league_code))
        pages = self._get_many(team_urls) if team_urls else []

        results: Dict[str, List[PlayerRecord]] = {}
        for (team, league_code), html in zip(keys, pages):
            std = self._parse_stats_table(html, ["Min", "Gls"]) if html else None
            keeper = self._parse_stats_table(html, ["Saves", "Save%"]) if html else None
            misc = self._parse_stats_table(html, ["TklW", "Int"]) if html else None
            shooting = self._parse_stats_table(html, ["Sh", "SoT"]) if html else None
            defense = self._parse_stats_table(html, ["Tkl", "Blocks"]) if html else None
            passing = self._parse_stats_table(html, ["KP", "PrgP"]) if html else None
            records: List[PlayerRecord] = []

            def row_map(df):
                rows = {}
                if df is None:
                    return rows
                for _, row in df.iterrows():
                    p = str(row.get("Player", "")).strip()
                    if not p or p.lower() in ("player", "squad total", "opponent total"):
                        continue
                    rows[p] = row
                return rows

            std_rows = row_map(std)
            keeper_rows = row_map(keeper)
            misc_rows = row_map(misc)
            shoot_rows = row_map(shooting)
            def_rows = row_map(defense)
            pass_rows = row_map(passing)

            names = (set(std_rows) | set(keeper_rows) | set(misc_rows)
                     | set(shoot_rows) | set(def_rows) | set(pass_rows))
            for p in sorted(names):
                row = std_rows.get(p)
                krow = keeper_rows.get(p)
                mrow = misc_rows.get(p)
                srow = shoot_rows.get(p)
                drow = def_rows.get(p)
                prow = pass_rows.get(p)
                rec = PlayerRecord(
                    name=p,
                    team=team,
                    league=_LEAGUE_SLUGS.get(league_code, league_code).replace("-", "_"),
                    season=season,
                    source=self.provider_name,
                    position=_coarse_position(row["Pos"] if row is not None else None),
                    age=self._num(row["Age"] if row is not None else None),
                    nationality=row["Nation"] if row is not None and "Nation" in row.index else None,
                    minutes=self._num(row["Min"] if row is not None else None),
                    appearances=self._num(row["MP"] if row is not None and "MP" in row.index else None),
                    goals=self._num(row["Gls"] if row is not None and "Gls" in row.index else None),
                    assists=self._num(row["Ast"] if row is not None and "Ast" in row.index else None),
                    shots=self._num(srow["Sh"] if srow is not None and "Sh" in srow.index
                                    else (row["Sh"] if row is not None and "Sh" in row.index else None)),
                    shots_on_target=self._num(srow["SoT"] if srow is not None and "SoT" in srow.index
                                              else (row["SoT"] if row is not None and "SoT" in row.index else None)),
                    tackles=self._num(mrow["TklW"] if mrow is not None and "TklW" in mrow.index
                                      else (drow["TklW"] if drow is not None and "TklW" in drow.index else None)),
                    interceptions=self._num(mrow["Int"] if mrow is not None and "Int" in mrow.index
                                            else (drow["Int"] if drow is not None and "Int" in drow.index else None)),
                    blocks=self._num(drow["Blocks"] if drow is not None and "Blocks" in drow.index else None),
                    clearances=self._num(drow["Clr"] if drow is not None and "Clr" in drow.index else None),
                    errors=self._num(drow["Err"] if drow is not None and "Err" in drow.index else None),
                    key_passes=self._num(prow["KP"] if prow is not None and "KP" in prow.index else None),
                    progressive_passes=self._num(prow["PrgP"] if prow is not None and "PrgP" in prow.index else None),
                    saves=self._num(krow["Saves"] if krow is not None and "Saves" in krow.index else None),
                    clean_sheets=self._num(krow["CS"] if krow is not None and "CS" in krow.index else None),
                    goals_conceded=self._num(krow["GA"] if krow is not None and "GA" in krow.index else None),
                    yellow_cards=int(self._num(row["CrdY"], 0)) if row is not None and "CrdY" in row.index else None,
                    red_cards=int(self._num(row["CrdR"], 0)) if row is not None and "CrdR" in row.index else None,
                    extra={"save_pct": self._num(krow["Save%"]) if krow is not None else None,
                           "pos_list": _pos_tokens(row["Pos"] if row is not None else None)},
                )
                records.append(rec)
            results[team] = records
            logger.info("FBRef %s %s: %d players", team, season, len(records))
        return results
