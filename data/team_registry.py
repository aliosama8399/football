"""
Team Name Registry — canonical normalization across data sources.

Replaces the hard-coded team_name_mapping dict in preprocess.py (which only
covered Premier League + La Liga) with a comprehensive 5-league registry
covering ~150 teams × 10 seasons × FDUK/Understat/FBRef aliases.

Usage:
    from data.team_registry import normalize_team_name
    canonical = normalize_team_name('Dortmund', 'D1')       # -> 'Borussia Dortmund'
    canonical = normalize_team_name('Man City', 'E0')      # -> 'Manchester City'
    canonical = normalize_team_name('Paris SG', 'F1')      # -> 'Paris Saint Germain'

If the alias is not in the registry, the input name is returned unchanged
(fuzzy fallback in preprocess.py C6 handles the long tail).
"""

# Registry schema: { league_code: { alias: canonical_name } }
# The canonical name is the Understat convention (longest, most descriptive).
# football-data.co.uk aliases are typically shortened; FBRef uses still other forms.
# When adding a new alias, copy it VERBATIM from the source CSV — these strings
# are matched exactly during the merge step.

TEAM_REGISTRY = {
    # ====================================
    # Premier League (E0)
    # ====================================
    'E0': {
        # FDUK aliases → Understat canonical
        'Man City':              'Manchester City',
        'Man United':            'Manchester United',
        "Nott'm Forest":         'Nottingham Forest',
        'Nottm Forest':          'Nottingham Forest',
        'Tottenham':             'Tottenham',
        'Tottenham Hotspur':     'Tottenham',
        'Sheffield Utd':         'Sheffield United',
        'Newcastle':             'Newcastle United',
        'Wolves':                'Wolverhampton Wanderers',
        'Brighton':              'Brighton',
        'West Ham':              'West Ham',
        'Luton':                 'Luton',
        'Leeds':                 'Leeds',
        'Leicester':             'Leicester',
        'Ipswich':               'Ipswich',
        # Already-canonical names (no-op, kept for explicitness)
        'Arsenal':               'Arsenal',
        'Aston Villa':           'Aston Villa',
        'Bournemouth':           'Bournemouth',
        'Brentford':             'Brentford',
        'Burnley':               'Burnley',
        'Chelsea':               'Chelsea',
        'Crystal Palace':        'Crystal Palace',
        'Everton':               'Everton',
        'Fulham':                'Fulham',
        'Liverpool':             'Liverpool',
        'Southampton':            'Southampton',
        'Sheffield United':      'Sheffield United',
        'Manchester City':       'Manchester City',
        'Manchester United':     'Manchester United',
        'Nottingham Forest':     'Nottingham Forest',
        'Newcastle United':      'Newcastle United',
        'Wolverhampton Wanderers': 'Wolverhampton Wanderers',
        'West Ham United':       'West Ham',
        'Luton Town':            'Luton',
        'Leeds United':          'Leeds',
        'Leicester City':        'Leicester',
        'Ipswich Town':          'Ipswich',
        'Brighton and Hove Albion': 'Brighton',
        # Historical PL teams from earlier seasons (10-season scope)
        'Norwich':               'Norwich City',
        'Norwich City':          'Norwich City',
        'Watford':               'Watford',
        'West Brom':             'West Bromwich Albion',
        'West Bromwich Albion':  'West Bromwich Albion',
        'Swansea':               'Swansea City',
        'Swansea City':          'Swansea City',
        'Stoke':                 'Stoke City',
        'Stoke City':            'Stoke City',
        'Huddersfield':          'Huddersfield Town',
        'Huddersfield Town':     'Huddersfield Town',
        'Cardiff':               'Cardiff City',
        'Cardiff City':          'Cardiff City',
        'Hull':                  'Hull City',
        'Hull City':             'Hull City',
        'QPR':                   'Queens Park Rangers',
        'Queens Park Rangers':   'Queens Park Rangers',
        'Sunderland':            'Sunderland',
        'Middlesbrough':         'Middlesbrough',
        'Birmingham City':       'Birmingham City',
        'Birmingham':            'Birmingham City',
        'Blackburn':             'Blackburn Rovers',
        'Blackburn Rovers':      'Blackburn Rovers',
        'Bolton':                'Bolton Wanderers',
        'Bolton Wanderers':      'Bolton Wanderers',
        'Fulham FC':             'Fulham',
    },

    # ====================================
    # La Liga (SP1)
    # ====================================
    'SP1': {
        # FDUK aliases → Understat canonical
        'Ath Madrid':            'Atletico Madrid',
        'Ath Bilbao':            'Athletic Club',
        'Betis':                 'Real Betis',
        'Sociedad':              'Real Sociedad',
        'Celta':                 'Celta Vigo',
        'Vallecano':             'Rayo Vallecano',
        'Valencia':              'Valencia',
        'Sevilla':               'Sevilla',
        'Barcelona':             'Barcelona',
        'Real Madrid':           'Real Madrid',
        'Villarreal':            'Villarreal',
        'Valladolid':            'Real Valladolid',
        'Getafe':                'Getafe',
        'Girona':                'Girona',
        'Mallorca':              'Mallorca',
        'Osasuna':               'Osasuna',
        'Espanyol':              'Espanyol',
        'Espanol':               'Espanyol',
        'Alaves':                'Alaves',
        'Almeria':               'Almeria',
        'Cadiz':                 'Cadiz',
        'Elche':                 'Elche',
        'Granada':               'Granada',
        'Las Palmas':            'Las Palmas',
        'Leganes':               'Leganes',
        'Levante':               'Levante',
        # Understat-canonical (already long-form)
        'Atletico Madrid':       'Atletico Madrid',
        'Athletic Club':         'Athletic Club',
        'Real Betis':            'Real Betis',
        'Real Sociedad':         'Real Sociedad',
        'Celta Vigo':            'Celta Vigo',
        'Rayo Vallecano':        'Rayo Vallecano',
        'Real Valladolid':       'Real Valladolid',
        'Espanyol CN':           'Espanyol',
        # Historical La Liga teams from earlier seasons
        'La Coruna':             'Deportivo La Coruna',
        'Dep La Coruna':         'Deportivo La Coruna',
        'Deportivo La Coruna':   'Deportivo La Coruna',
        'Gijon':                 'Sporting Gijon',
        'Sp Gijon':              'Sporting Gijon',
        'Sporting Gijon':        'Sporting Gijon',
        'Las Palmas CN':         'Las Palmas',
        'Oviedo':                'Oviedo',
        'Real Oviedo':           'Oviedo',
        'Tenerife':              'Tenerife',
        'Numancia':              'Numancia',
        'Malaga':                'Malaga',
        'Huesca':                'Huesca',
        'Racing Sant':           'Racing Santander',
        'Racing Santander':      'Racing Santander',
        'Recreativo':            'Recreativo Huelva',
        'Zaragoza':              'Zaragoza',
        'CD Zaragoza':           'Zaragoza',
        'Valladolid CF':         'Real Valladolid',
        'Athletic Bilbao':       'Athletic Club',
        'Real Betis Balompie':   'Real Betis',
    },

    # ====================================
    # Bundesliga (D1)
    # ====================================
    'D1': {
        # FDUK aliases → Understat canonical
        'Dortmund':              'Borussia Dortmund',
        'Leverkusen':            'Bayer Leverkusen',
        "M'gladbach":            'Borussia M.Gladbach',
        'Mgladbach':             'Borussia M.Gladbach',
        'FC Koln':               'FC Cologne',
        'Koln':                  'FC Cologne',
        'Ein Frankfurt':         'Eintracht Frankfurt',
        'Hertha':                'Hertha Berlin',
        'Braunschweig':          'Braunschweig',
        'Greuther Furth':        'Greuther Furth',
        'Darmstadt':             'Darmstadt',
        'Heidenheim':            'FC Heidenheim',
        'Augsburg':              'Augsburg',
        'Bayern Munich':         'Bayern Munich',
        'Bochum':                'Bochum',
        'Freiburg':              'Freiburg',
        'Hoffenheim':            'Hoffenheim',
        'Holstein Kiel':         'Holstein Kiel',
        'Mainz':                 'Mainz 05',
        'RB Leipzig':            'RasenBallsport Leipzig',
        'Schalke 04':            'Schalke 04',
        'St Pauli':              'St. Pauli',
        'Stuttgart':             'VfB Stuttgart',
        'Union Berlin':          'Union Berlin',
        'Werder Bremen':        'Werder Bremen',
        'Wolfsburg':             'Wolfsburg',
        # Understat-canonical forms (kept for explicitness)
        'Borussia Dortmund':     'Borussia Dortmund',
        'Bayer Leverkusen':      'Bayer Leverkusen',
        'Borussia M.Gladbach':   'Borussia M.Gladbach',
        'Borussia Monchengladbach': 'Borussia M.Gladbach',
        'FC Cologne':            'FC Cologne',
        'Eintracht Frankfurt':   'Eintracht Frankfurt',
        'Hertha Berlin':         'Hertha Berlin',
        'FC Heidenheim':         'FC Heidenheim',
        'Mainz 05':              'Mainz 05',
        'RasenBallsport Leipzig': 'RasenBallsport Leipzig',
        'St. Pauli':             'St. Pauli',
        'VfB Stuttgart':         'VfB Stuttgart',
        # Historical Bundesliga teams
        'Hamburg':               'Hamburger SV',
        'Hamburger SV':          'Hamburger SV',
        'HSV':                   'Hamburger SV',
        'Ingolstadt':            'Ingolstadt',
        'FC Ingolstadt':         'Ingolstadt',
        'Paderb':                'Paderborn',
        'Paderborn':             'Paderborn',
        'Nurnberg':              'Nurnberg',
        'Nuernberg':             'Nurnberg',
        '1. FC Nurnberg':        'Nurnberg',
        'Hannover':              'Hannover 96',
        'Hannover 96':           'Hannover 96',
        'Bielefeld':             'Arminia Bielefeld',
        'Arminia Bielefeld':     'Arminia Bielefeld',
        'Dusseldorf':            'Fortuna Dusseldorf',
        'Fortuna Dusseldorf':   'Fortuna Dusseldorf',
        'Stuttgart Kickers':     'Stuttgart Kickers',
        'Kaiserslautern':        'Kaiserslautern',
        '1. FC Kaiserslautern':  'Kaiserslautern',
        'Cottbus':               'Energie Cottbus',
        'Energie Cottbus':       'Energie Cottbus',
        'Karlsruher':            'Karlsruher SC',
        'Karlsruher SC':         'Karlsruher SC',
        'Bremen':                'Werder Bremen',
    },

    # ====================================
    # Serie A (I1)
    # ====================================
    'I1': {
        # FDUK aliases → Understat canonical
        'Milan':                 'AC Milan',
        'Inter':                 'Inter',
        'Roma':                  'Roma',
        'Lazio':                 'Lazio',
        'Napoli':                'Napoli',
        'Juventus':              'Juventus',
        'Fiorentina':            'Fiorentina',
        'Atalanta':              'Atalanta',
        'Bologna':               'Bologna',
        'Cagliari':              'Cagliari',
        'Genoa':                 'Genoa',
        'Sampdoria':             'Sampdoria',
        'Sassuolo':              'Sassuolo',
        'Torino':                'Torino',
        'Udinese':               'Udinese',
        'Verona':                'Verona',
        'Hellas Verona':         'Verona',
        'Chievo':                'Chievo',
        'Empoli':                'Empoli',
        'Salernitana':           'Salernitana',
        'Monza':                 'Monza',
        'Cremonese':             'Cremonese',
        'Spezia':                'Spezia',
        'Lecce':                 'Lecce',
        'Frosinone':             'Frosinone',
        'Como':                  'Como',
        'Venezia':               'Venezia',
        'Parma':                 'Parma Calcio 1913',
        # Understat-canonical
        'AC Milan':              'AC Milan',
        'Parma Calcio 1913':     'Parma Calcio 1913',
        # Historical Serie A teams
        'Benevento':             'Benevento',
        'Brescia':               'Brescia',
        'Spal':                  'SPAL',
        'SPAL':                  'SPAL',
        'Spal 2013':             'SPAL',
        'Crotone':               'Crotone',
        'Vicenza':               'Vicenza',
        'Vicenza Virtus':        'Vicenza',
        'Pescara':               'Pescara',
        'Palermo':               'Palermo',
        'Carpi':                 'Carpi',
        'Frosinone Calcio':      'Frosinone',
        'Hellas Verona FC':      'Verona',
        'Benevento Calcio':      'Benevento',
        'Venezia FC':            'Venezia',
        'US Salernitana 1919':   'Salernitana',
        'ACF Fiorentina':        'Fiorentina',
        'Hellas Verona':         'Verona',
        'AC Monza':              'Monza',
        'US Cremonese':          'Cremonese',
        'Spezia Calcio':         'Spezia',
        'US Lecce':              'Lecce',
        'AC Cagliari':           'Cagliari',
        'Genoa CFC':             'Genoa',
        'UC Sampdoria':          'Sampdoria',
        'US Sassuolo':           'Sassuolo',
        'Torino FC':             'Torino',
        'Udinese Calcio':        'Udinese',
        'SSC Napoli':            'Napoli',
        'Atalanta BC':           'Atalanta',
        'Bologna FC':            'Bologna',
        'AS Roma':               'Roma',
        'SS Lazio':              'Lazio',
        'Inter Milan':           'Inter',
        'FC Internazionale':     'Inter',
        'FC Crotone':            'Crotone',
        'Benevento Calcio S.p.A.': 'Benevento',
    },

    # ====================================
    # Ligue 1 (F1)
    # ====================================
    'F1': {
        # FDUK aliases → Understat canonical
        'Paris SG':              'Paris Saint Germain',
        'Paris SG                              ': 'Paris Saint Germain',  # trailing spaces seen in some CSV
        'St Etienne':            'Saint-Etienne',
        'Clermont':              'Clermont Foot',
        'Brest':                 'Brest',
        'Ajaccio':               'Ajaccio',
        'Angers':                'Angers',
        'Auxerre':               'Auxerre',
        'Le Havre':              'Le Havre',
        'Lens':                  'Lens',
        'Lille':                 'Lille',
        'Lorient':               'Lorient',
        'Lyon':                  'Lyon',
        'Marseille':             'Marseille',
        'Metz':                  'Metz',
        'Monaco':                'Monaco',
        'Montpellier':           'Montpellier',
        'Nantes':                'Nantes',
        'Nice':                  'Nice',
        'Reims':                 'Reims',
        'Rennes':                'Rennes',
        'Strasbourg':            'Strasbourg',
        'Toulouse':              'Toulouse',
        'Troyes':                'Troyes',
        # Understat-canonical
        'Paris Saint Germain':   'Paris Saint Germain',
        'Saint-Etienne':         'Saint-Etienne',
        'Clermont Foot':         'Clermont Foot',
        # Historical Ligue 1 teams
        'Bordeaux':              'Bordeaux',
        'FC Girondins Bordeaux': 'Bordeaux',
        'Nimes':                 'Nimes',
        'Nimes Olympique':       'Nimes',
        'Amiens':                'Amiens',
        'SC Amiens':             'Amiens',
        'Toulouse FC':           'Toulouse',
        'Stade Brestois 29':     'Brest',
        'Stade Brest':           'Brest',
        'Lorient FC':            'Lorient',
        'FC Lorient':            'Lorient',
        'Stade Rennais':         'Rennes',
        'Stade Rennais FC':      'Rennes',
        'Olympique Lyonnais':    'Lyon',
        'Olympique Marseille':   'Marseille',
        'Olympique de Marseille':'Marseille',
        'AS Monaco':             'Monaco',
        'AS Saint-Etienne':      'Saint-Etienne',
        'AS Nancy':              'Nancy',
        'Nancy':                 'Nancy',
        'AS Lorraine':           'Nancy',
        'SM Caen':               'Caen',
        'Caen':                  'Caen',
        'Stade de Reims':        'Reims',
        'Stade Malherbe Caen':   'Caen',
        'Montpellier HSC':       'Montpellier',
        'Montpellier Herault SC':'Montpellier',
        'FC Metz':               'Metz',
        'RC Strasbourg Alsace':  'Strasbourg',
        'RC Strasbourg':         'Strasbourg',
        'ESTAC Troyes':          'Troyes',
        'ESTAC':                 'Troyes',
        'LOSC Lille':            'Lille',
        'LOSC':                  'Lille',
        'RC Lens':               'Lens',
        'Racing Club de Lens':   'Lens',
        'OGC Nice':              'Nice',
        'FC Nantes':             'Nantes',
        'FC Lorient':            'Lorient',
        'Dijon':                 'Dijon',
        "Cote D'Opale Dijon":   'Dijon',
        'Guingamp':              'Guingamp',
        'En Avant Guingamp':     'Guingamp',
        'Evian':                 'Evian Thonon Gaillard',
        'Evian Thonon Gaillard': 'Evian Thonon Gaillard',
        'Sochaux':               'Sochaux',
        'FC Sochaux-Montbeliard': 'Sochaux',
        'Bastia':                'Bastia',
        'SC Bastia':             'Bastia',
        'Valenciennes':          'Valenciennes',
        'Valenciennes FC':       'Valenciennes',
        'AC Ajaccio':            'Ajaccio',
        'Gazalec Ajaccio':       'Ajaccio',
        'Athletico Marseille':   'Marseille',
        'Clermont Foot 63':      'Clermont Foot',
    },
}


import logging
import difflib

_logger = logging.getLogger("team_registry")
_UNMAPPED_LOGGED = set()


def _fuzzy_match(name: str, candidates: dict[str, str], threshold: float = 0.82) -> str | None:
    """Attempt fuzzy matching against known aliases / canonical names."""
    try:
        from rapidfuzz import process, fuzz
        match = process.extractOne(name, candidates.keys(), scorer=fuzz.token_sort_ratio)
        if match and match[1] >= (threshold * 100):
            return candidates[match[0]]
    except ImportError:
        # Fallback to difflib standard library
        close = difflib.get_close_matches(name, candidates.keys(), n=1, cutoff=threshold)
        if close:
            return candidates[close[0]]
    return None


def normalize_team_name(name: str, league_code: str = None) -> str:
    """
    Normalize a team name to its canonical form for the given league.

    Args:
        name: Team name as it appears in raw data (FDUK / Understat / FBRef)
        league_code: Fduk league code (E0, SP1, D1, I1, F1). If None, will
                     search all leagues (slower, used for cross-league merges).

    Returns:
        Canonical team name string. If alias is unknown, attempts fuzzy matching
        against the registry; if still unmapped, logs a warning and returns the
        stripped name.
    """
    import pandas as pd
    if pd.isna(name):
        return name
    name = str(name).strip()

    # 1. Exact match in specified league
    if league_code and league_code in TEAM_REGISTRY:
        league_map = TEAM_REGISTRY[league_code]
        if name in league_map:
            return league_map[name]
        # 2. Fuzzy match in specified league
        fuzzy = _fuzzy_match(name, league_map)
        if fuzzy:
            _logger.info("Fuzzy matched team '%s' -> '%s' (league %s)", name, fuzzy, league_code)
            return fuzzy
        # Log unmapped
        if (name, league_code) not in _UNMAPPED_LOGGED:
            _UNMAPPED_LOGGED.add((name, league_code))
            _logger.warning("Unmapped team name '%s' for league '%s' — falling back to raw.", name, league_code)
        return name

    # 3. League unspecified — search all leagues (cross-league case)
    all_mappings = {}
    for l_code, league_map in TEAM_REGISTRY.items():
        if name in league_map:
            return league_map[name]
        all_mappings.update(league_map)

    # 4. Fuzzy match across all leagues
    fuzzy = _fuzzy_match(name, all_mappings)
    if fuzzy:
        _logger.info("Fuzzy matched team '%s' -> '%s' (cross-league)", name, fuzzy)
        return fuzzy

    if (name, None) not in _UNMAPPED_LOGGED:
        _UNMAPPED_LOGGED.add((name, None))
        _logger.warning("Unmapped team name '%s' (cross-league) — falling back to raw.", name)
    return name


def get_registry_for_league(league_code: str) -> dict:
    """Return the {alias: canonical} mapping for a single league."""
    return TEAM_REGISTRY.get(league_code, {})


def list_all_canonical_teams() -> set:
    """Return a set of every canonical team name across all leagues."""
    return {canonical for league_map in TEAM_REGISTRY.values()
            for canonical in league_map.values()}


if __name__ == "__main__":
    # Quick smoke test
    test_cases = [
        ('Man City',              'E0', 'Manchester City'),
        ('Dortmund',              'D1', 'Borussia Dortmund'),
        ("M'gladbach",            'D1', 'Borussia M.Gladbach'),
        ('FC Koln',               'D1', 'FC Cologne'),
        ('Paris SG',              'F1', 'Paris Saint Germain'),
        ('St Etienne',            'F1', 'Saint-Etienne'),
        ('Ath Madrid',            'SP1', 'Atletico Madrid'),
        ('Celta',                 'SP1', 'Celta Vigo'),
        ('Milan',                 'I1', 'AC Milan'),
        ('Parma',                 'I1', 'Parma Calcio 1913'),
        ('Verona',                'I1', 'Verona'),
        ('Wolves',                'E0', 'Wolverhampton Wanderers'),
        ('Newcastle',             'E0', 'Newcastle United'),
        ('Sheffield Utd',         'E0', 'Sheffield United'),
        ('Brighton',              'E0', 'Brighton'),
        ('West Ham',              'E0', 'West Ham'),
        ('Clermont',              'F1', 'Clermont Foot'),
        ('Hertha',                'D1', 'Hertha Berlin'),
        ('Ein Frankfurt',         'D1', 'Eintracht Frankfurt'),
        ('RB Leipzig',            'D1', 'RasenBallsport Leipzig'),
    ]
    print("=== Team Registry Smoke Test ===")
    passed = 0
    failed = 0
    for raw, code, expected in test_cases:
        got = normalize_team_name(raw, code)
        if got == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: normalize_team_name('{raw}', '{code}') = '{got}' (expected '{expected}')")
    print(f"\nSmoke test: {passed} passed, {failed} failed out of {len(test_cases)}")
    print(f"Total leagues: {len(TEAM_REGISTRY)}")
    print(f"Total aliases: {sum(len(m) for m in TEAM_REGISTRY.values())}")
    print(f"Total canonical teams: {len(list_all_canonical_teams())}")
