"""
Player provider registry + factory — mirrors get_llm_provider().

    from data.player_providers.factory import get_player_provider, list_providers

    provider = get_player_provider("fbref")          # or "fod", "understat"
    squad = provider.fetch_team_squad("Arsenal", "E0", "2425")

Unknown names raise ValueError; 'all' returns every provider (for fusion).
"""

import logging
from typing import Dict, List

from data.player_providers.base import BasePlayerProvider

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, type] = {}


def _import_providers():
    from data.player_providers.fbref import FbrefPlayerProvider
    from data.player_providers.fod import FodPlayerProvider
    from data.player_providers.understat import UnderstatPlayerProvider
    _REGISTRY["fbref"] = FbrefPlayerProvider
    _REGISTRY["fod"] = FodPlayerProvider
    _REGISTRY["understat"] = UnderstatPlayerProvider


def get_player_provider(provider_type: str = "", **kwargs) -> BasePlayerProvider:
    """Factory: provider name (fbref | fod | understat | all) → instance."""
    _import_providers()
    if provider_type in ("", "all") or str(provider_type).lower() == "all":
        return _AllProvidersProxy()
    cls = _REGISTRY.get(str(provider_type).lower())
    if not cls:
        raise ValueError(
            f"Unknown player provider '{provider_type}'. "
            f"Supported: {sorted(_REGISTRY)} + ['all']"
        )
    return cls(**kwargs)


def list_providers() -> List[str]:
    _import_providers()
    return sorted(_REGISTRY)


class _AllProvidersProxy(BasePlayerProvider):
    """'all': delegate to every provider and merge records by name."""

    provider_name = "all"

    def __init__(self):
        self._providers = []
        for name in ("fbref", "fod", "understat"):
            try:
                self._providers.append(get_player_provider(name))
            except Exception as e:
                logger.warning("player provider '%s' unavailable: %s", name, e)

    def fetch_team_squad(self, team: str, league_code: str, season: str):
        merged = {}
        for p in self._providers:
            try:
                for rec in p.fetch_team_squad(team, league_code, season):
                    key = _norm_name(rec.name)
                    if key not in merged:
                        merged[key] = rec
                    else:
                        merged[key] = _fuse(merged[key], rec)
            except Exception as e:
                logger.warning("provider '%s' failed for %s: %s", p.provider_name, team, e)
        return _dedup_squad(merged.values())

    def capabilities(self) -> Dict[str, bool]:
        caps = {}
        for p in self._providers:
            for k, v in p.capabilities().items():
                caps[k] = caps.get(k, False) or v
        return caps


def _fuse(primary: "PlayerRecord", other: "PlayerRecord"):
    """Fill missing fields of primary from other (provider fusion)."""
    for field_name in ("position", "age", "nationality", "minutes", "appearances",
                       "goals", "assists", "xg", "xa", "shots", "shots_on_target",
                       "tackles", "interceptions", "saves", "clean_sheets",
                       "goals_conceded", "yellow_cards", "red_cards"):
        if getattr(primary, field_name) is None and getattr(other, field_name) is not None:
            setattr(primary, field_name, getattr(other, field_name))
    _merge_extra(primary, other)
    # Understat knows some wingers as forwards while FBRef lists them as pure
    # MF (e.g. Lamine Yamal). Trust understat's FW label when the player's
    # attacking output confirms it — otherwise attackers end up in midfield.
    if primary.position == "MF" and other.position == "FW":
        xg = getattr(other, "xg", 0) or 0
        shots = getattr(other, "shots", 0) or 0
        goals = getattr(other, "goals", 0) or 0
        if xg >= 4 or shots >= 30 or goals >= 6:
            primary.position = "FW"
    return primary


def _merge_extra(primary: "PlayerRecord", other: "PlayerRecord"):
    pe = dict(primary.extra or {})
    oe = dict(other.extra or {})
    for k, v in oe.items():
        if k == "pos_list":
            pe["pos_list"] = list(dict.fromkeys((pe.get("pos_list") or []) + (v or [])))
        elif pe.get(k) is None:
            pe[k] = v
    primary.extra = pe


def _dedup_squad(records):
    """Merge near-duplicate player spellings across providers — e.g.
    'Kylian Mbappe-Lottin' vs 'Kylian Mbappe', 'Raul' vs 'Raul Asencio',
    'Daniel Carvajal' vs 'Dani Carvajal'. The fbref record becomes the base."""
    import difflib
    out = []
    for rec in sorted(records, key=lambda r: 0 if r.source == "fbref" else 1):
        for base in out:
            if _is_same_player(base, rec, difflib):
                _fuse(base, rec)
                break
        else:
            out.append(rec)
    return sorted(out, key=lambda r: r.name)


def _is_same_player(a, b, difflib) -> bool:
    na, nb = _norm_name(a.name), _norm_name(b.name)
    if na == nb:
        return True
    if not na or not nb:
        return False
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    if ratio >= 0.80:
        return True
    ta, tb = set(na.split()), set(nb.split())
    same_pos = a.position and a.position == b.position
    close_minutes = bool(a.minutes and b.minutes and abs(a.minutes - b.minutes) < 260)
    if not (same_pos and close_minutes):
        return False
    if ratio >= 0.50:
        return True
    if ta and (ta <= tb or tb <= ta):
        return True
    return False


def _norm_name(name: str) -> str:
    """Diacritic-insensitive match key ('Álex Baena' == 'Alex Baena')."""
    import unicodedata
    s = unicodedata.normalize("NFKD", name)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()
