
import pandas as pd

# Load the datasets
understat_df = pd.read_csv('understat.com.csv')
understat_per_game_df = pd.read_csv('understat_per_game.csv')

# Display basic information for understat_df
print('--- understat.com.csv Info ---')
print(understat_df.head())
print(understat_df.info())
print(understat_df.describe())

# Display basic information for understat_per_game_df
print('\n--- understat_per_game.csv Info ---')
print(understat_per_game_df.head())
print(understat_per_game_df.info())
print(understat_per_game_df.describe())

