import os
import pandas as pd

for rel in [os.path.join('data', 'electricity.xlsx'),
            os.path.join('data', 'ETTh1.xlsx'),
            os.path.join('data', 'winddata.xlsx')]:
    print('=' * 80)
    print(rel, 'exists:', os.path.exists(rel))
    if not os.path.exists(rel):
        continue
    xl = pd.ExcelFile(rel)
    print('  sheets:', xl.sheet_names)
    s = xl.sheet_names[0]
    df = xl.parse(s, nrows=3)
    cols = list(df.columns)
    print('  sheet:', s, 'shape(head):', df.shape)
    print('  cols:', cols[:6], '...', cols[-2:])
    print('  first col name:', cols[0], '| dtype:', df.dtypes.iloc[0])
    print('  all numeric:', all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes))
    # 尝试按用户要求方式读取
    try:
        d2 = pd.read_excel(rel, sheet_name=s, index_col=0, parse_dates=True)
        print('  index_col=0,parse_dates -> index dtype:', d2.index.dtype,
              '| monotonic:', d2.index.is_monotonic_increasing,
              '| cols numeric:', all(pd.api.types.is_numeric_dtype(t) for t in d2.dtypes))
    except Exception as e:
        print('  index_col=0,parse_dates FAILED:', type(e).__name__, e)
