"""
Per-match player stats collector (Understat match pages).

For each league-season, fetch the match schedule and then every match
page, extracting the embedded playersData: per player per match —
minutes, goals, assists, shots, xG, xA, key passes. This is the raw
material for *through-the-season* player ratings (data/player_form.py):
cumulative stats as of any match date, so a prediction for a match in
March only uses data up to that date (no future leakage).

Output: data/raw/player_match/player_match_{league_code}_{season}.csv
(resumable — matches already collected are skipped).

CLI:
    python -m data.collectors.player_match_stats --leagues E0 --seasons 2425
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data.player_providers.understat import _LEAGUE_SLUGS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("player_match_stats")

_UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
}

_OUT_DIR = Path("data/raw/player_match")
_DELAY = 1.5

_HEADER = ("match_id,match_date,home_team,away_team,player_name,team,position,"
           "minutes,goals,assists,shots,xg,xa,key_passes,yellow_cards,red_cards,"
           "xg_chain,xg_buildup,understat_id\n")

_SESSION = None


def _session() -> requests.Session:
    """One shared session; warm-up visit unlocks the match-data endpoint."""
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update(_UA)
        try:
            s.get("https://understat.com/match/26602/", timeout=30)
        except Exception as e:
            logger.warning("session warm-up failed (continuing): %s", e)
        _SESSION = s
    return _SESSION


def _get(url: str, timeout: int = 30) -> str:
    headers = dict(_UA, Referer="https://understat.com/",
                   **{"X-Requested-With": "XMLHttpRequest"})
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def fetch_schedule(league_code: str, season: str) -> list:
    slug = _LEAGUE_SLUGS.get(league_code)
    if not slug:
        raise ValueError(f"understat slug not mapped: {league_code}")
    year = f"20{season[:2]}"
    url = f"https://understat.com/getLeagueData/{slug}/{year}/"
    data = json.loads(_get(url))
    return data.get("dates", [])


def fetch_match_players(match_id: str, home_title: str, away_title: str) -> list:
    """Rows of per-player stats for one match (both teams)."""
    url = f"https://understat.com/getMatchData/{match_id}/"
    headers = {"Referer": f"https://understat.com/match/{match_id}/",
               "X-Requested-With": "XMLHttpRequest"}
    r = _session().get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = json.loads(r.text)
    rosters = data.get("rosters") or {}
    rows = []
    side_titles = {"h": home_title, "a": away_title}
    for side, players in rosters.items():
        team = side_titles.get(side, "")
        if not isinstance(players, dict):
            continue
        for p in players.values():
            rows.append({
                "player_name": str(p.get("player", "")).strip(),
                "team": team,
                "position": str(p.get("position", "")).strip(),
                "minutes": _num(p.get("time")),
                "goals": _num(p.get("goals")),
                "assists": _num(p.get("assists")),
                "shots": _num(p.get("shots")),
                "xg": _num(p.get("xG")),
                "xa": _num(p.get("xA")),
                "key_passes": _num(p.get("key_passes")),
                "yellow_cards": _num(p.get("yellow_card")),
                "red_cards": _num(p.get("red_card")),
                "xg_chain": _num(p.get("xGChain")),
                "xg_buildup": _num(p.get("xGBuildup")),
                "understat_id": p.get("player_id"),
            })
    return rows


def _num(v, default=0.0):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def collect(league_code: str, season: str) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"player_match_{league_code}_{season}.csv"
    done_ids: set = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split(",")
                if parts and parts[0].strip().isdigit():
                    done_ids.add(parts[0].strip())

    schedule = fetch_schedule(league_code, season)
    pending = [m for m in schedule
               if str(m.get("id")) not in done_ids and m.get("isResult")]
    logger.info("%s %s: %d matches, %d already collected",
                league_code, season, len(schedule), len(done_ids))
    if not out_path.exists():
        with out_path.open("w", encoding="utf-8") as f:
            f.write(_HEADER)

    with out_path.open("a", encoding="utf-8", newline="") as f:
        for i, m in enumerate(pending, 1):
            mid = str(m.get("id"))
            match_date = str(m.get("datetime", ""))[:10]
            home_title = str(m.get("h", {}).get("title", ""))
            away_title = str(m.get("a", {}).get("title", ""))
            try:
                rows = fetch_match_players(mid, home_title, away_title)
            except Exception as e:
                logger.warning("match %s failed: %s", mid, e)
                time.sleep(2 * _DELAY)
                continue
            for r in rows:
                f.write(f"{mid},{match_date},{_csv(home_title)},"
                        f"{_csv(away_title)},{_csv(r['player_name'])},"
                        f"{_csv(r['team'])},{_csv(r['position'])},"
                        f"{r['minutes']},{r['goals']},{r['assists']},"
                        f"{r['shots']},{r['xg']},{r['xa']},{r['key_passes']},"
                        f"{r['yellow_cards']},{r['red_cards']},"
                        f"{r['xg_chain']},{r['xg_buildup']},"
                        f"{r['understat_id']}\n")
            f.flush()
            done_ids.add(mid)
            if i < len(pending):
                time.sleep(_DELAY)
            if i % 50 == 0:
                logger.info("%s %s: %d/%d", league_code, season, i, len(pending))
    logger.info("done %s %s → %s", league_code, season, out_path)
    return out_path


def _csv(v) -> str:
    s = str(v)
    return s.replace(",", " ").replace('"', "'")


def main():
    parser = argparse.ArgumentParser(description="Collect per-match player stats")
    parser.add_argument("--leagues", default="E0", help="comma-separated E0,SP1,D1,I1,F1")
    parser.add_argument("--seasons", default="2425", help="comma-separated season codes")
    args = parser.parse_args()
    for season in args.seasons.split(","):
        for lc in args.leagues.split(","):
            collect(lc.strip(), season.strip())


if __name__ == "__main__":
    main()
