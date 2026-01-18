# Errors Found

| subset     | transcript_version | all_models_found_errors | majority_consensus_errors |
| ---------- | ------------------ | ----------------------: | ------------------------: |
| aci        | asr                |                      34 |                        27 |
| aci        | asrcorr            |                      34 |                        27 |
| virtscribe | asr                |                      34 |                        29 |
| virtscribe | humantrans         |                       1 |                         0 |
| virtassist | humantrans         |                       4 |                         1 |



By subset and split: 
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




Notice in D2N034 (aci, train)

```txt
[doctor] how you doing is it is it gotten any better
[patient] yeah yeah i've been having a lot of pain of my shoulder for the last three weeks now and it's not getting better okay do you remember what you were doing when the pain first started
[doctor] so i i was thinking that i i ca n't recall like falling on it injuring it getting hit
[patient] hmmm
[doctor] i have been doing a lot of work in my basement and i even i put in a new ceiling so i do n't know if it's from all that activity doing that but otherwise that's that's all i can think of
```