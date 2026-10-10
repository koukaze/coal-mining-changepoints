# Coal-mining disaster data

`data/coal_intervals_days.csv` contains 190 chronological intervals, in days,
between 191 British coal-mining disasters. The source describes explosions
with at least 10 fatalities, spanning 15 March 1851 to 22 March 1962.
The data were exported from the public `coal` dataset in the R package
[`boot`, version 1.3-32](https://CRAN.R-project.org/package=boot). See the
[`coal` documentation](https://stat.ethz.ch/R-manual/R-devel/library/boot/html/coal.html).

## Format and interpretation

The CSV has one integer column, `interval_days`, with 190 rows and no missing
values. Its sum is **40,549 days**. One interval is zero; retain it because it
represents two recorded events on the same day. The intervals are not annual
accident counts and should not be sorted independently.

The model places the first event at day zero and reconstructs all 191 event
times as `[0, cumsum(intervals)]`. Time in the likelihood is elapsed days, and
the Poisson intensity is events per day. A year scale based on division by
365.25 is a display convention, not an exact reconstruction of calendar dates.

## Source and reuse terms

The `boot` package's `DESCRIPTION` declares **`License: Unlimited`**. The
[R package licensing specification](https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Licensing)
defines this as unrestricted distribution and use, subject to relevant laws.
The package contains no separate data-specific restriction for `coal`.
The numerical export is distributed under these terms with the source attribution below.

The package credits *A Handbook of Small Data Sets* (Hand et al., 1994) as its
data source and cites [Jarrett (1979)](https://doi.org/10.1093/biomet/66.1.191).
These attributions apply to the data independently of the repository's code.

## Reproducible export

The pinned upstream archive is
[`boot_1.3-32.tar.gz`](https://cran.r-project.org/src/contrib/boot_1.3-32.tar.gz).
Its SHA-256 is:

```text
3a05aced6fea42a5c310c5c6ab7a2019f69f757f5e77c4961183977747136c97
```

The conversion from the package's 191 decimal-year dates is:

```r
intervals <- as.integer(round(diff(boot::coal$date) * 365.25))
```

The largest residual before integer rounding is below `8e-11` days. The conversion
uses the package's decimal-year encoding to recover integer inter-event intervals.

To regenerate the CSV using R and `boot` 1.3-32, run from the repository root:

```bash
Rscript scripts/export_coal_data.R
```

R is needed only for this optional export; Python analyses use the bundled CSV.

The bundled CSV has SHA-256:

```text
79f2838c87c018c44a72a08e5c0b5b30875351a360cfa32a50735606c6137c0f
```
