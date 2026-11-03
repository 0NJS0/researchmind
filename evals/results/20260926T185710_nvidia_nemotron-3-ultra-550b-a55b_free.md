# Prompt evaluation — LLM_Toxicity_Detection

*2026-09-26T18:57:10+00:00 · model `nvidia/nemotron-3-ultra-550b-a55b:free` (screening `nvidia/nemotron-3-super-120b-a12b:free`) · 27 LLM requests*

## Headline

| Metric | Value |
|---|---|
| Summary fact recall (gold facts found) | 0.925 |
| Summaries empty or failed | 0 |
| Summary statements with numbers not in the paper | 3 |
| Ask answers containing all expected facts | 4/4 |
| Ask citations matching a retrieved passage | 1.0 |
| Relevance screening accuracy (incl. focus exclusions) | 1.0 |

## Summaries

| Paper | Recall | Missing facts | Unsupported numbers | Requests |
|---|---|---|---|---|
| Efficient_Detection_of_Toxic_Prompts_in_Large_Language_ | 0.8 | td_acc_spc | 0 | 1 |
| Efficient_Toxic_Content_Detection_by_Bootstrapping_and_ | 0.75 | bd_46 | 1 | 1 |
| Holistic_Evaluation_of_Large_Language_Models_Robustness | 1.0 | — | 1 | 1 |
| How_Quantization_Shapes_Bias_in_Large_Language_Models | 1.0 | — | 0 | 3 |
| Toxic_Text_Classification_in_Portuguese_Is_LLaMA_3.1_8B | 1.0 | — | 0 | 1 |
| When_LLMs_Meet_Acoustic_Landmarks_Speech_for_Depression | 1.0 | — | 1 | 1 |
- Efficient_Toxic_Content_Detection_by_Boo: “Experiments use Toxigen (8,960 train, 940 test), SBIC (4,000 train, 1,000 test), DHate (4,000 train, 1,000 test), and Amazon private dataset (10,000 test only).” → not in paper: 10000
- Holistic_Evaluation_of_Large_Language_Mo: “Evaluation uses 16 datasets including RealToxicityPrompts (1200 toxic prompts), MMLU clinical subsets, BoolQ, TruthfulQA, MedMCQA, MedQA, Bigbench, and others.” → not in paper: 16
- When_LLMs_Meet_Acoustic_Landmarks_Speech: “Three-stage pipeline: landmark extraction via Python algorithm (6 frequency bands, two-pass), hint cross-modal instruction fine-tuning with LoRA, P-tuning class” → not in paper: 900100

## Ask answers

| Question | Correct | Expected paper cited | Citations verified |
|---|---|---|---|
| What accuracy and false positive rate does ToxicDetector achieve on Re | True | True | 8/8 |
| How much do student models fine-tuned with DToT rationales improve acc | True | True | 1/1 |
| What F1-score does the fine-tuned LLaMA 3.1 8B model reach on Portugue | True | True | 0/0 |
| Which quantization methods were studied for their effect on bias in la | True | True | 1/1 |

## Relevance screening

Accuracy 1.0 · precision 1.0 · recall 1.0


## Live gap quality (no LLM calls)

17 gaps · generic 0 · backed by 2+ papers 10 · checked against wider literature 2 · confidence {'high': 3, 'medium': 5, 'low': 9}
