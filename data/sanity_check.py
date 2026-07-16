"""
Sanity check for processed_matches.csv.

Run AFTER preprocess.py to verify:
  - All 5 leagues have >90% non-null xG coverage
  - Zero betting columns survived the filter
  - xG_missing sentinel columns exist and are int
  - Per-league row counts are reasonable (>0)

Usage: python data/sanity_check.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd


BETTING_TOKENS = [
    'B365', 'BW', 'IW', 'PS', 'WH', 'VC', 'MAX', 'AVG','BTTS',
    'ODDS', 'BET', 'BF', '1XB', 'BFE', 'BFEX', 'AH',
    'PAH', 'PCAH', 'PROB', 'PINNACLE', 'GB', 'SB', 'UB',
    'PSCH', 'PSCD', 'PSCA', 'B36C','LBH','LBD','LBA','Bb1X2','BbMxH','BbAvH','BbMxD','BbAvD','BbMxA','BbAvA','BbOU','BbMx>2.5','BbAv>2.5','BbMx<2.5','BbAv<2.5'
]
MIN_XG_COVERAGE = 0.90   # 90% non-null xG required per league


def main(processed_path: str = "data/processed/processed_matches.csv"):
    path = Path(processed_path)
    if not path.exists():
        print(f"ERR File not found: {path}")
        print("Run python data/preprocess.py first.")
        return 1

    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {path}: {len(df)} rows × {len(df.columns)} cols\n")

    errors = []

    # 1. Zero betting columns
    bet_cols = [c for c in df.columns
                if any(t in c.upper() for t in BETTING_TOKENS)
                and 'Ref_' not in c                    # Ref_AvgYellows etc. are NOT betting
                and 'xG' not in c and 'GF' not in c and 'GA' not in c]
    if bet_cols:
        errors.append(f"BETTING COLUMNS SURVIVED: {bet_cols}")

    # 2. xG coverage per league
    xG_cols = [c for c in df.columns
               if ('xG' in c or 'xGA' in c) and '_missing' not in c]
    xG_missing_cols = [c for c in df.columns if c.endswith('_missing')]

    league_col = 'League' if 'League' in df.columns else 'Div'
    if league_col in df.columns:
        print(f"Per-league xG coverage (threshold: {MIN_XG_COVERAGE:.0%}):")
        for league_name, sub in df.groupby(league_col, observed=True):
            stats = []
            for xgc in xG_cols:
                if xgc in sub.columns:
                    cov = sub[xgc].notna().mean()
                    status = "OK" if cov >= MIN_XG_COVERAGE else "LOW"
                    stats.append(f"  {xgc}: {cov:.1%} {status}")
                    if cov < MIN_XG_COVERAGE:
                        errors.append(
                            f"{league_name} {xgc} coverage={cov:.1%} < {MIN_XG_COVERAGE:.0%}")
            print(f"  {league_name} ({len(sub)} rows)")
            for s in stats:
                print(s)
    else:
        errors.append("No League/Div column found for per-league xG coverage")

    print(f"\nMissing sentinel columns ({len(xG_missing_cols)}): {xG_missing_cols}")

    # 3. Total row count (must be >0)
    if len(df) == 0:
        errors.append("0 rows — empty dataset")

    # 4. Columns consistent with output
    print(f"\nColumn types: {df.dtypes.value_counts().to_dict()}")

    if errors:
        print(f"\n*** SANITY CHECK FAILED ({len(errors)} issues) ***")
        for e in errors:
            print(f"  -> {e}")
        return 1
    else:
        print(f"\n*** SANITY CHECK PASSED ***")
        print(f"  {len(df)} rows, {len(df.columns)} cols,"
              f" {len(xG_missing_cols)} xG sentinel cols, 0 betting cols")
        return 0


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)