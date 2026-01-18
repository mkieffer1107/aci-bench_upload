# ACI-Bench


HuggingFace upload of ACI-Bench, which evaluates a model's ability to convert clinical dialogue into structured clinical notes. This dataset includes the [benchmark itself](https://huggingface.co/datasets/mkieffer/ACI-Bench), as well as data from ablation studies testing different transcription methods. If used, please cite the original authors using the citation below.



## Dataset Details


| subset     | transcript_version | train | valid | test1 | test2 | test3 | Total |
| ---------- | ------------------ | ----- | ----- | ----- | ----- | ----- | ----- |
| aci        | asr                | 35    | 11    | 22    | 22    | 22    | 112   |
| aci        | asrcorr            | 35    | 11    | 22    | 22    | 22    | 112   |
| aci        | humantrans         | 0     | 0     | 0     | 0     | 0     | 0     |
| virtassist | asr                | 0     | 0     | 0     | 0     | 0     | 0     |
| virtassist | asrcorr            | 0     | 0     | 0     | 0     | 0     | 0     |
| virtassist | humantrans         | 20    | 5     | 10    | 10    | 10    | 55    |
| virtscribe | asr                | 12    | 4     | 8     | 8     | 8     | 40    |
| virtscribe | asrcorr            | 0     | 0     | 0     | 0     | 0     | 0     |
| virtscribe | humantrans         | 12    | 4     | 8     | 8     | 8     | 40    |
| ALL        | ALL                | 114   | 35    | 70    | 70    | 70    | 359   |

### Dataset Description

The dataset consists of different subsets capturing different clinical workflows

1) ambient clinical intelligence (`aci`): doctor-patient dialogue
2) virtual assistant (`virtassist`): doctor-patient dialogue with queues to trigger Dragon Copilot, e.g., "hey, dragon. show me the chest x-ray"
3) virtual scribe (`virtscribe`): doctor-patient dialogue with a short dictation from the doctor about the patient at the very beginning

There are three different transcription versions:
1) `asr`: machine-transcribed
2) `asrcorr`: human corrections to `asr`, for example: "nonsmile" in D2N081 --> "non-small" in ACI006
3) `humantrans`: transcribed by a human

The subsets have the following transcription versions
1) `aci`: `asr` and `asrcorr`
2) `virtassist`: `humantrans` only
3) `virtscribe`: `asr` and `humantrans`


### Dataset Sources 

- **GitHub:** https://github.com/wyim/aci-bench
- **Paper:** https://www.nature.com/articles/s41597-023-02487-3


### Direct Use

```python
from datasets import load_dataset

SUBSETS = ["virtassist", "virtscribe", "aci"]
SPLITS = ["train", "valid", "test1", "test2", "test3"]

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # 1) Load ONE subset (config) with ALL splits
    # ---------------------------------------------------------------------
    virtassist_all = load_dataset("mkieffer/ACI-Bench-MedARC", name="virtassist")

    # ---------------------------------------------------------------------
    # 2) Load ONE subset (config) with ONE split
    # ---------------------------------------------------------------------
    virtassist_train = load_dataset("mkieffer/ACI-Bench-MedARC", name="virtassist", split="train")

    # ---------------------------------------------------------------------
    # 3) Load TWO subsets (configs), all splits for each
    # ---------------------------------------------------------------------
    two_subsets = {
        "virtassist": load_dataset("mkieffer/ACI-Bench-MedARC", name="virtassist"),
        "aci": load_dataset("mkieffer/ACI-Bench-MedARC", name="aci"),
    }

    # ---------------------------------------------------------------------
    # 4) Load ALL subsets (virtassist, virtscribe, aci), all splits each
    # ---------------------------------------------------------------------
    all_subsets = {subset: load_dataset("mkieffer/ACI-Bench-MedARC", name=subset) for subset in SUBSETS}
    aci_all = all_subsets["aci"]  # DatasetDict
    aci_train = aci_all["train"]  # Dataset
    aci_valid = aci_all["valid"]

    # ---------------------------------------------------------------------
    # 5) Load multiple splits at once
    # ---------------------------------------------------------------------
    # load each split, concatenated
    aci_all_test_concat = load_dataset("mkieffer/ACI-Bench-MedARC", name="aci", split=["train", "test1+test2+test3"])
    
    # load each split separately
    aci_all_test_separate = load_dataset("mkieffer/ACI-Bench-MedARC", name="aci", split=["train", "test1", "test2", "test3"])
```


## Citation 

```bibtex
@article{aci-bench,
  author = {Wen{-}wai Yim and
                Yujuan Fu and
                Asma {Ben Abacha} and
                Neal Snider and Thomas Lin and Meliha Yetisgen},
  title = {ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation},
  journal = {Nature Scientific Data},
  year = {2023}
}
```