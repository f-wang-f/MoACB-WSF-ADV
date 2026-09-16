import pandas as pd
for f in ['windata.1h.xlsx', 'winddata.xlsx', 'winddata3.xlsx']:
    df = pd.read_excel(f)
    print(f, '| rows =', len(df), '| cols =', list(df.columns))
