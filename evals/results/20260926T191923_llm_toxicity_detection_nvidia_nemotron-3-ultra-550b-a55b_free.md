# Prompt evaluation — LLM_Toxicity_Detection

*2026-09-26T19:19:23+00:00 · model `nvidia/nemotron-3-ultra-550b-a55b:free` (screening `nvidia/nemotron-3-super-120b-a12b:free`) · 28 LLM requests*

## Headline

| Metric | Value |
|---|---|
| Summary fact recall (gold facts found) | 1.0 (+0.075 vs previous) |
| Summaries empty or failed | 0 (+0 vs previous) |
| Summary statements with numbers not in the paper | 2 (-1 vs previous) |
| Ask answers containing all expected facts | 4/4 |
| Ask citations matching a retrieved passage | 1.0 (+0.0 vs previous) |
| Ask answers with no citations at all | 0 |
| Relevance screening accuracy (incl. focus exclusions) | 1.0 (+0.0 vs previous) |

## Summaries

| Paper | Recall | Missing facts | Unsupported numbers | Requests |
|---|---|---|---|---|
| Efficient_Detection_of_Toxic_Prompts_in_Large_Language_ | 1.0 | — | 1 | 1 |
| Efficient_Toxic_Content_Detection_by_Bootstrapping_and_ | 1.0 | — | 0 | 1 |
| Holistic_Evaluation_of_Large_Language_Models_Robustness | 1.0 | — | 0 | 1 |
| How_Quantization_Shapes_Bias_in_Large_Language_Models | 1.0 | — | 1 | 3 |
| Toxic_Text_Classification_in_Portuguese_Is_LLaMA_3.1_8B | 1.0 | — | 0 | 1 |
| When_LLMs_Meet_Acoustic_Landmarks_Speech_for_Depression | 1.0 | — | 0 | 1 |
- Efficient_Detection_of_Toxic_Prompts_in_: “Concept prompt augmentation boosts F1 significantly (e.g., Insult +0.1557); processing time 0.0780 seconds/prompt; training ~89 seconds for 100 epochs.” → not in paper: 89
- How_Quantization_Shapes_Bias_in_Large_La: “Greedy decoding (temperature=0), max 2000 tokens for reasoning tasks, 250 tokens for toxicity/sentiment; significance via approximate randomization test (10,000” → not in paper: 2000

## Ask answers

| Question | Correct | Expected paper cited | Citations verified |
|---|---|---|---|
| What accuracy and false positive rate does ToxicDetector achieve on Re | True | True | 2/2 |
| How much do student models fine-tuned with DToT rationales improve acc | True | True | 3/3 |
| What F1-score does the fine-tuned LLaMA 3.1 8B model reach on Portugue | True | True | 1/1 |
| Which quantization methods were studied for their effect on bias in la | True | True | 4/4 |

## Relevance screening

Accuracy 1.0 · precision 1.0 · recall 1.0


## Live gap quality (no LLM calls)

21 gaps · generic 0 · backed by 2+ papers 13 · checked against wider literature 4 · confidence {'high': 4, 'medium': 7, 'low': 10}
