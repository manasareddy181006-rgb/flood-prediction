import pandas as pd
d = pd.read_csv(r"C:\Users\bsnre\Downloads\chingka kalai\rfa_regions.csv")
s = d[d["station"].str.contains("seasonal", case=False, na=False)]
print(f"{len(s)} seasonal stations, {int(s['is_contributing'].sum())} contributing")
print(s[["station","region","is_contributing","t","n_used"]].to_string(index=False))