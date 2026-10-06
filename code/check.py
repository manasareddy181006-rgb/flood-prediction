import pandas as pd
d = pd.read_csv(r"C:\Users\bsnre\Downloads\chingka kalai\rfa_attributes.csv")
print(d["rain_max_mm"].describe())
print(d.nsmallest(10, "rain_max_mm")[["station", "rain_max_mm", "rain_annual_mean_mm"]])