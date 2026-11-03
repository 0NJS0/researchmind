# Eval corpus

The papers used by the `public_health_economics` eval (`evals/gold/public_health_economics.json`) are not
included in this repository because they belong to their authors and publishers. Download them yourself
and save each one in `evals/corpus/public_health_economics/` under the file name shown below.

A corpus paper can be either a `.txt` file (first line = title, then the full text) or a `.pdf`.

| File name | Paper | Source |
|---|---|---|
| `neet_young_adults_scoping_review.txt` | *Young adults not in education, employment, or training (NEET): a global scoping review*. BMC Public Health (2025). | [doi:10.1186/s12889-025-24781-y](https://doi.org/10.1186/s12889-025-24781-y) |
| `suicide_unemployed_youth_cross_sectional.txt` | *Prevalence and factors associated with suicide and suicidal behaviours among unemployed youth in sub-Saharan Africa: a systematic review protocol*. BMJ Open (2026). | [doi:10.1136/bmjopen-2026-118613](https://doi.org/10.1136/bmjopen-2026-118613) |
| `unemployment_metabolic_syndrome_40y_cohort.txt` | *Unemployment during life can lead to metabolic syndrome in adult age. A 40-year follow-up of the Northern Swedish Cohort*. European Journal of Public Health. | Search the title on the journal site or Google Scholar |
| `disability_rolls_decline_in_unemployment.pdf` | Autor, D. H. & Duggan, M. G. *The Rise in the Disability Rolls and the Decline in Unemployment*. The Quarterly Journal of Economics 118(1), 2003. | [doi:10.1162/00335530360535171](https://doi.org/10.1162/00335530360535171) |

The gold file only stores short quotes from each paper (the sentence each checked fact comes from), so the
eval's expected answers can be reviewed without the full texts.
