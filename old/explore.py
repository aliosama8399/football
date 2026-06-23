import pandas as pd
df = pd.read_csv('understat_test.csv')
with open('understat_columns.txt', 'w') as f:
    f.write("COLUMNS:\n")
    f.write(", ".join(df.columns) + "\n\n")
    f.write("SAMPLE ROW:\n")
    f.write(str(df.iloc[0].to_dict()))
