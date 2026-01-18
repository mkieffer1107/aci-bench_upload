# What is this?

Many of the ACI-Bench data entries contain swapped speaker tags. We ran each transcript through a group of LLMs (GPT-5 mini, Grok 4.1 Fast, and Gemini 3 Flash Preview) to identify, as best as possible with a minimal setup, all such instances.

To run yourself, simply run

```py
find_errors.py
```

You can configure the judge models by modifying the `MODELS` list, currently set as

```py
MODELS = ["google/gemini-3-flash-preview", "x-ai/grok-4.1-fast", "openai/gpt-5-mini"]
```

Note that we use OpenRouter for all models to reduce friction.

Results can be found in the [errors folder](errors). 
- [progress.json](errors/progress.json) is used to store progress when the main script
- [metadata.json](errors/metadata.json) gives high-level results from various angles
- [subset]_errors.json files contain all records judged by at least one LLM to contain an error with explanations (subset = aci/virtassist/virtscribe)

All error entries look something like this

```json
    "subset": "aci",
    "split": "train",
    "encounter_id": "D2N034",
    "transcript_version": "asr",
    "errors": {
      "google/gemini-3-flash-preview": [
        {
          "lines": [
            2,
            -1
          ],
          "reason": "Starting at line 2, the speaker tags are consistently swapped. The tag [patient] introduces the appointment, asks medical history questions, and performs a physical exam, which are responsibilities of a doctor. Conversely, the tag [doctor] describes symptoms, mentions working in the basement, and asks if the pain will go away, which are characteristically patient behaviors."
        }
```

See the **Corrections Output Format** section below for more. It's easiest to follow along with our [HuggingFace upload](https://huggingface.co/datasets/mkieffer/ACI-Bench-MedARC) of the dataset using the following steps:

1) Open an error JSON file and find a record you with to view (we are looking at [aci_errors.json](errors/aci_errors.json) here)
2) Go to the [HuggingFace dataset](https://huggingface.co/datasets/mkieffer/ACI-Bench-MedARC) 
1) Select the subset (`aci`)
2) Select the split (`train`)
3) Copy the `encounter_id` (`D2N034`) from the json file and ctrl+f in HF to search
4) Ensure the record it pulls up contains the correct `transcript_version`
5) Look in the `enumerated_dialogue` column to find all relevant lines

Note that we have not implemented any corrections yet.

---

# Errors Found

**By subset and transcript version:**

A few things to notics:
1) All models agreed that both transcript versions of `aci`, `asr` (default asr) and `asrcorr` (human corrections to `asr`), contain errors. This means that the human edits to `asr` only 
2) The `virtassist` subset only contains one transcript version, `humantrans`, meaning a human transcribed these audio recordings. As expected, there is a low error rate.
3) The `virtscribe` subset contains two transcription types. The first, `asr`, has a high error rate. However, when those same audio recordings were transcribed by humans, `humantrans`, only one error exists. Looking more closely at that record, only one model finds an error, and we believe that it was confused because there is a third speaker tag, [patient_guest], preceding the section it marked as erroneous.

| subset     | transcript_version | all_models_found_errors | majority_consensus_errors |
| ---------- | ------------------ | ----------------------: | ------------------------: |
| aci        | asr                |                      34 |                        27 |
| aci        | asrcorr            |                      34 |                        27 |
| virtscribe | asr                |                      34 |                        29 |
| virtscribe | humantrans         |                       1 |                         0 |
| virtassist | humantrans         |                       4 |                         1 |



**By subset and split:**

| subset     | split | total_notes | all_models_found_errors | majority_consensus_errors |
| ---------- | ----- | ----------: | ----------------------: | ------------------------: |
| aci        | train |          70 |                      25 |                        19 |
| aci        | valid |          22 |                       9 |                         8 |
| aci        | test1 |          44 |                      16 |                        15 |
| aci        | test2 |          44 |                       8 |                         6 |
| aci        | test3 |          44 |                      10 |                         6 |
| virtassist | train |          20 |                       2 |                         0 |
| virtassist | valid |           5 |                       0 |                         0 |
| virtassist | test1 |          10 |                       0 |                         0 |
| virtassist | test2 |          10 |                       1 |                         0 |
| virtassist | test3 |          10 |                       1 |                         1 |
| virtscribe | train |          24 |                      10 |                         8 |
| virtscribe | valid |           8 |                       4 |                         4 |
| virtscribe | test1 |          16 |                       5 |                         4 |
| virtscribe | test2 |          16 |                       7 |                         6 |
| virtscribe | test3 |          16 |                       9 |                         7 |

---

# Corrections Output Format

`find_errors.py` outputs JSON files named `{subset}_errors.json` with the following structure:

- `lines` is a range `[start, end]` inclusive
- `[1, -1]` indicates the entire transcript has swapped speaker tags (-1 = last line)
- `[-1, -1]` indicates no errors found (transcript is correct)

```json
[
    {
        "subset": "aci/virtassist/virtscribe",
        "split": "train/valid/test1/test2/test3",
        "encounter_id": "1234567890",
        "transcript_version": "asr/asrcorr/humantrans",
        "models_with_errors": ["google/gemini-3-flash-preview", "x-ai/grok-4.1-fast", "openai/gpt-5-mini"],
        "errors": {
            "google/gemini-3-flash-preview": [
                {
                    "lines": [1, 4],
                    "reason": "Lines 1-4 have swapped speaker tags..."
                },
                {
                    "lines": [15, 22],
                    "reason": "Lines 15-22 have swapped speaker tags..."
                }
            ],
            "x-ai/grok-4.1-fast": [
                {
                    "lines": [1, -1],
                    "reason": "entire transcript is swapped"
                }
            ],
            "openai/gpt-5-mini": [
                {
                    "lines": [-1, -1],
                    "reason": "no errors found"
                }
            ]
        }
    }
]
```



