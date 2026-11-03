# Prompt evaluation — Unemployment and health

*2026-09-26T19:37:29+00:00 · model `nvidia/nemotron-3-ultra-550b-a55b:free` (screening `nvidia/nemotron-3-super-120b-a12b:free`) · 7 LLM requests*

## Headline

| Metric | Value |
|---|---|
| Summary fact recall (gold facts found) | 0.9 |
| Summaries empty or failed | 0 |
| Summary statements with numbers not in the paper | 2 |
| Ask answers with no citations at all | 0 |
| Relevance screening accuracy (incl. focus exclusions) | 1.0 |

## Summaries

| Paper | Recall | Missing facts | Unsupported numbers | Requests |
|---|---|---|---|---|
| unemployment_metabolic_syndrome_40y_cohort | 1.0 | — | 0 | 1 |
| suicide_unemployed_youth_cross_sectional | 1.0 | — | 0 | 1 |
| neet_young_adults_scoping_review | 0.8 | ne_11 | 2 | 1 |
| disability_rolls_decline_in_unemployment | 0.8 | di_53 | 0 | 1 |
- neet_young_adults_scoping_review: “67% quantitative (survey/register/international databases), 24% qualitative, 9% mixed methods; categorized into 11 thematic topics.” → not in paper: 9
- neet_young_adults_scoping_review: “Geographic distribution: Europe 57% (91 studies), Africa 12% (19), Asia-Pacific 13% (21), Australia 6% (9), Northern America 7% (11), South America 4% (6).” → not in paper: 4, 6, 7

## Relevance screening

Accuracy 1.0 · precision 1.0 · recall 1.0

