# Regional Flood Frequency Analysis: Pipeline Overview

**Goal:** estimate the T-year flood (T = 2 to 100 years) at a catchment that has **no streamflow gauge**, using only its catchment attributes and data borrowed from similar gauged catchments (index-flood method, Hosking & Wallis 1997).

**The core idea:**

> **Q(T) = mu × q(T)**
>
> - **mu** (index flood) = mean annual maximum flow at the site, in cumec. Estimated from attributes (Step 4).
> - **q(T)** (growth curve) = dimensionless ratio "T-year flood ÷ mean flood". Estimated from a pooled region of similar catchments (Steps 2–3).

| Step | Question it answers | Script |
|---|---|---|
| 1 | Which stations, which attributes, and how do we summarise them? | `code/step1.py` |
| 2 | Which gauged catchments are "similar" to each target? | `code/roi.py` |
| 3 | What is the shape of the flood distribution in that region? | `code/step3.py` |
| 4 | How large is the mean flood, from attributes only? | `code/step4.py` |
| 5 | Does it work at a site we pretend is ungauged? | `code/step5.py` |

---

## Step 1: Data and PCA

1. **Stations.** 276 stations have morphometry, rainfall and temperature. **90 are kept**: those with more than 15 years of annual-maximum streamflow (185 too short, 1 duplicate).
2. **Annual maximum series (AMS).** The largest flow in each year, per station.
3. **10 attributes per catchment:**
   - log(area), drainage density, mean elevation, relief, mean slope
   - latitude, longitude
   - max rainfall, max temperature, min temperature

   Exact duplicates were removed, e.g. stream length = drainage density × area.
4. **Standardise** each attribute: z = (x − mean) / sd, so units don't matter.
5. **PCA** on the 10 standardised attributes gives 10 uncorrelated components:
   - PC1–PC4 have eigenvalue > 1 and together explain 83% of the variance.
   - All 10 PCs are kept as the "attribute space", so no information is lost.

**Output:** PC scores for every station, plus the eigenvectors (loadings) used to convert any catchment into PCs.

---

## Step 2: Region of Influence (which catchments are similar)

For **each** of the 90 catchments, a region of its own is built:

1. **Start** with the target catchment.
2. **Add the nearest catchment** in PC space (Euclidean distance) to the region's centre, one at a time.
3. **Check each new member.** A candidate is **rejected** if:
   - it is **discordant** (Di test: its L-moments are unusual for the region), or
   - adding it makes the region **heterogeneous** (H1 ≥ 1, kappa Monte-Carlo test).

   In both cases the next-nearest catchment is tried instead (H&W §4.4).
4. **Stop at about 500 station-years** of pooled record. This is the H&W "5T rule": a 100-year flood needs at least 5 × 100 station-years.

**Result:**
- Region size: 9–17 stations (median 13).
- Pooled record: median 515 station-years.
- **89/90 regions acceptably homogeneous** (median H1 = 0.52).
- 9 regions stop slightly short of 500 years (368–388), because no other station keeps them homogeneous.

---

## Step 3: Distribution fitting and growth curve

For each catchment's region:

1. **Normalise:** each station's floods are divided by its own mean (index flood), so every station has mean 1.
2. **Regional L-moments:** the members' L-CV, L-skew and L-kurtosis, averaged and weighted by record length.
3. **Fit 5 candidate distributions:** GLO, GEV, GPA, GNO, PE3. Each is matched to the regional mean (= 1), L-CV and L-skew.
4. **Choose the best one with Z^DIST:** how close the distribution's L-kurtosis is to the region's, measured against 500 simulated regions. A fit is acceptable if |Z| ≤ 1.64; the smallest |Z| wins.
5. **Growth curve q(T)** from the chosen distribution, at T = 2, 5, 10, 25, 50, 100.

**Result:**
- **90/90 regions have an acceptable fit.**
- Chosen: GNO 33, GLO 19, GPA 17, GEV 13, PE3 8.

---

## Step 4: Regression for the mean flood (mu) on PCs

The growth curve is only a ratio, so an ungauged site still needs its **mu**.

1. **Response:** log10(mu) for the 90 gauged stations.
   - mu is the station's mean annual maximum, in real cumec (not normalised).
   - Logs are used because mu spans 7 to 33,000 cumec.
2. **Predictors:** the PC scores from Step 1. They are uncorrelated, so there is no multicollinearity.
3. **Forward stepwise selection:** add the PC with the smallest p-value each round, only while p < 0.05.
4. **Coefficients** by ordinary least squares.
5. **Back-transform** to cumec with Duan's smearing factor (1.67), which corrects the log-retransformation bias.

**Equation:**

> log10(mu) = 2.734 + 0.124·PC1 + 0.421·PC7 + 0.115·PC3 + 0.127·PC4 − 0.189·PC6 − 0.366·PC9
>
> R² = 0.43, adjusted R² = 0.39, F-test p = 2 × 10⁻⁸

In terms of the original attributes, the largest effects are **area (+)**, **rainfall (+)** and **max temperature (−)**.

**Ungauged site:**
1. Standardise its attributes with the gauged stations' mean and sd.
2. Multiply by the eigenvector loadings to get its PC scores.
3. Put the scores into the equation.

**Accuracy (observed vs estimated mean):**

| | R² (log) | Median error |
|---|---|---|
| In-sample | 0.43 | 57% |
| Leave-one-out (ungauged test) | 0.08 | 66% |

The attributes only partly explain mu. That is a data limitation: soil, land use and geology are not available.

---

## Step 5: Ungauged-site test (leave-one-out), in progress

For each of the 90 stations in turn:

1. **Hide** its streamflow record.
2. **Convert** it to PCs using the other 89 stations' mean, sd and eigenvectors.
3. **Rebuild its region** from gauged stations only, with the same rule as Step 2.
4. **Fit the growth curve** on that region (Step 3 method) to get q(T).
5. **Predict mu** from the Step 4 regression, refitted without this station.
6. **Estimate:** Q_est(T) = mu_hat × q(T).
7. **Compare** with Q_computed(T), from a distribution fitted to the station's own real record:
   - error at each return period
   - Estimated vs Computed plots for the 50- and 100-year floods
   - whether the station rejoins the same region as before

---

## Known data issues

- **Khanditar:** one year reads 641,030 cumec against a median of 3,651. This is a likely unit or digit error, and it inflates its mean to about 33,000.
- **Tuichang and Sevanur:** each has one extreme attribute (|z| > 3), which makes the regression extrapolate badly for them.
