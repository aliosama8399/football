"""
FAISS document builders (v2) — used by rag/build_faiss_index.py.

Collections:
  - match_stats     : every row of processed_matches.csv (13,293) — richer text
                      (referee strictness, weather, stadium, Over/BTTS flags)
  - team_season     : per team × league × season aggregate summaries (~1,300)
  - team_profile    : tactical profile text from team_tactics.json (96 teams)
  - analysis        : tactical analyses from football_tactical.jsonl (13,911)
                      — only indexed when rag.kb_index_analyses is true

Metadata always carries the filterable keys: doc_type | league | season | date,
plus collection-specific team keys (team / home_team / away_team) for
FAISSProvider.search(filter_meta=...).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _fmt(v, digits: int = 2) -> str:
    try:
        f = float(v)
        if f != f:  # NaN
            return "?"
        return f"{f:.{digits}f}"
    except (TypeError, ValueError):
        return "?" if pd.isna(v) else str(v)


def _int(v):
    try:
        f = float(v)
        if f != f:
            return 0
        return int(f)
    except (TypeError, ValueError):
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Match stat chunks (one per CSV row)
# ─────────────────────────────────────────────────────────────────────────────

def build_match_stat_chunks(df: pd.DataFrame) -> list[dict]:
    docs = []
    for _, row in df.iterrows():
        home = str(row.get("HomeTeam", "?"))
        away = str(row.get("AwayTeam", "?"))
        date = str(row.get("Date", "?"))
        lg = str(row.get("League", "?"))
        ssn = str(row.get("Season", "?"))
        ftr = str(row.get("FTR", "?"))
        hg, ag = _int(row.get("FTHG")), _int(row.get("FTAG"))

        text = (
            f"{home} (Home) vs {away} (Away) | {lg} | Season {ssn} | {date}\n"
            f"Result: {ftr} ({hg}-{ag}) | xG: {_fmt(row.get('Home_xG'))}-{_fmt(row.get('Away_xG'))}\n"
            f"Shots: {_int(row.get('HS'))}-{_int(row.get('AS'))} | "
            f"SOT: {_int(row.get('HST'))}-{_int(row.get('AST'))} | "
            f"Corners: {_int(row.get('HC'))}-{_int(row.get('AC'))} | "
            f"Fouls: {_int(row.get('HF'))}-{_int(row.get('AF'))} | "
            f"Yellows: {_int(row.get('HY'))}-{_int(row.get('AY'))}\n"
            f"Home Form (5): PPM={_fmt(row.get('HomeForm_5'))}, GF={_fmt(row.get('HomeGF_5'))}, "
            f"GA={_fmt(row.get('HomeGA_5'))}, xG={_fmt(row.get('HomexG_5'))}\n"
            f"Away Form (5): PPM={_fmt(row.get('AwayForm_5'))}, GF={_fmt(row.get('AwayGF_5'))}, "
            f"GA={_fmt(row.get('AwayGA_5'))}, xG={_fmt(row.get('AwayxG_5'))}\n"
            f"H2H: {_int(row.get('H2H_Matches'))} meetings | "
            f"Home wins: {_int(row.get('H2H_HomeWins'))} | "
            f"Away wins: {_int(row.get('H2H_AwayWins'))} | "
            f"Draws: {_int(row.get('H2H_Draws'))}\n"
            f"Over 2.5: {_int(row.get('Over2.5'))} | BTTS: {_int(row.get('BTTS'))} | "
            f"Clean sheets: H={_int(row.get('HomeCleanSheet'))} A={_int(row.get('AwayCleanSheet'))}\n"
            f"Referee strictness: {_fmt(row.get('Ref_Strictness'))} | "
            f"Temp: {_fmt(row.get('temperature'))}C | Rain: {_fmt(row.get('precipitation'))}mm | "
            f"Wind: {_fmt(row.get('wind_speed'))}km/h | Stadium: {row.get('stadium_name', '?')}"
        )

        docs.append({
            "text": text,
            "doc_type": "match_stats",
            "home_team": home,
            "away_team": away,
            "team": home,                      # primary team for single-team filters
            "date": date,
            "league": lg,
            "season": ssn,
            "result": ftr,
        })
    logger.info("Match-stat docs: %d", len(docs))
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Team × league × season summaries
# ─────────────────────────────────────────────────────────────────────────────

def build_team_season_summaries(df: pd.DataFrame) -> list[dict]:
    teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
    docs = []
    for team in sorted(teams):
        rows = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
        for (league, season), g in rows.groupby(["League", "Season"]):
            home = g[g["HomeTeam"] == team]
            away = g[g["AwayTeam"] == team]
            wins = len(home[home["FTR"] == "H"]) + len(away[away["FTR"] == "A"])
            draws = len(g[g["FTR"] == "D"])
            losses = len(g) - wins - draws
            gf = int(home["FTHG"].sum() + away["FTAG"].sum())
            ga = int(home["FTAG"].sum() + away["FTHG"].sum())
            points = 3 * wins + draws
            played = len(g)
            hg_avg = _fmt(home["FTHG"].mean()) if len(home) else "?"
            xg_avg = _fmt(pd.concat([home["Home_xG"], away["Away_xG"]]).mean()) \
                if len(g) else "?"

            text = (
                f"{team} — {league} {season} season summary\n"
                f"Played: {played} | W {wins} D {draws} L {losses} | "
                f"GF {gf} GA {ga} | GD {gf - ga} | Points {points} | Win rate {wins / played:.0%}\n"
                f"Avg goals home: {hg_avg} | Avg xG: {xg_avg}"
            )

            last_date = str(g["Date"].max()) if len(g) else ""
            docs.append({
                "text": text,
                "doc_type": "team_season",
                "team": team,
                "league": str(league),
                "season": str(season),
                "date": last_date,
            })
    logger.info("Team-season docs: %d", len(docs))
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Team tactical profiles (team_tactics.json)
# ─────────────────────────────────────────────────────────────────────────────

def build_team_profiles(tactics_path: Path, df: Optional[pd.DataFrame] = None) -> list[dict]:
    if not Path(tactics_path).exists():
        logger.warning("team_tactics.json not found — skipping team_profile docs: %s", tactics_path)
        return []

    with open(tactics_path, "r", encoding="utf-8") as f:
        tactics = json.load(f)

    league_of = {}
    if df is not None and "HomeTeam" in df.columns and "League" in df.columns:
        for name, g in df.groupby("HomeTeam")["League"]:
            val = g.dropna()
            league_of[str(name)] = str(val.mode()[0]) if len(val) else "Unknown"

    docs = []
    for team, t in tactics.items():
        strengths = " | ".join(t.get("strengths", [])[:5]) or "—"
        weaknesses = " | ".join(t.get("weaknesses", [])[:5]) or "—"
        text = (
            f"{team} tactical profile\n"
            f"Attack: {t.get('attack_headline', '')}\n{t.get('attack_tactic', '')}\n"
            f"Defense: {t.get('defense_headline', '')}\n{t.get('defense_tactic', '')}\n"
            f"Strengths: {strengths}\nWeaknesses: {weaknesses}"
        )
        docs.append({
            "text": text,
            "doc_type": "team_profile",
            "team": str(team),
            "league": league_of.get(str(team), "Unknown"),
            "season": "",
            "date": "",
        })
    logger.info("Team-profile docs: %d", len(docs))
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Tactical analyses (football_tactical.jsonl — full 13,911 records)
# ─────────────────────────────────────────────────────────────────────────────

def load_tactical_analyses(path: Path) -> list[dict]:
    docs = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            messages = row.get("messages", [])
            analysis = ""
            for m in messages:
                if m.get("role") == "assistant" and m.get("content"):
                    analysis = m["content"]
                    break
            if not analysis:
                skipped += 1
                continue

            home, away, league, season, date = _match_meta(row, messages)

            docs.append({
                "text": analysis,
                "doc_type": "analysis",
                "team": home,
                "home_team": home,
                "away_team": away,
                "date": date,
                "league": league,
                "season": season,
                "match_id": row.get("match_id", ""),
                "actual_result": row.get("actual_result", "?"),
                "gnn_prediction": row.get("gnn_prediction", "?"),
            })

    if skipped:
        logger.warning("Analysis docs skipped (malformed): %d", skipped)
    logger.info("Analysis docs: %d", len(docs))
    return docs


def _match_meta(row: dict, messages: list) -> tuple:
    """Extract home/away/league/season/date from the user message JSON
    (authoritative), falling back to match_id parsing."""
    home, away, league, season, date = "?", "?", "?", "?", "?"
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            try:
                payload = json.loads(m["content"])
            except (json.JSONDecodeError, TypeError):
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("match"), dict):
                md = payload["match"]
                home = str(md.get("home_team", "?"))
                away = str(md.get("away_team", "?"))
                league = str(md.get("league", "?"))
                season = str(md.get("season", "?"))
                date = str(md.get("date", "?"))
            break

    # Fallback: match_id like "2024-08-15_Athletic Club_vs_Getafe"
    if date == "?" or home == "?":
        mid = str(row.get("match_id", ""))
        if "_vs_" in mid:
            left, away = mid.split("_vs_", 1)
            parts = left.split("_", 1)
            date = parts[0] if parts else date
            home = parts[1] if len(parts) > 1 else home
            away = away.strip()
    return home, away, league, season, date


def build_all_docs(csv_path: Path, tactics_path: Path, analyses_path: Path,
                   index_analyses: bool) -> list[dict]:
    """Assemble the full v2 doc set per current config."""
    df = pd.read_csv(csv_path, low_memory=False)
    docs = build_match_stat_chunks(df)
    docs += build_team_season_summaries(df)
    docs += build_team_profiles(tactics_path, df=df)
    if index_analyses:
        docs += load_tactical_analyses(analyses_path)
    return docs
