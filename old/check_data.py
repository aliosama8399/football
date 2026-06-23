import pandas as pd

# Check existing understat data
df = pd.read_csv('understat_per_game.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {df.columns.tolist()[:15]}')
print(f'Leagues: {df["league"].unique()}')
print(f'Years: {sorted(df["year"].unique())}')
print(f'\nSample team names:')
print(df['team'].unique()[:20])
