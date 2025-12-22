from datasets import load_dataset

SUBSETS = ["virtassist", "virtscribe", "aci"]
SPLITS = ["train", "valid", "test1", "test2", "test3"]

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # 1) Load ONE subset (config) with ALL splits
    # ---------------------------------------------------------------------
    # virtassist_all = load_dataset("mkieffer/ACI-Bench", name="virtassist")

    # # ---------------------------------------------------------------------
    # # 2) Load ONE subset (config) with ONE split
    # # ---------------------------------------------------------------------
    # virtassist_train = load_dataset("mkieffer/ACI-Bench", name="virtassist", split="train")

    # # ---------------------------------------------------------------------
    # # 3) Load TWO subsets (configs), all splits for each
    # # ---------------------------------------------------------------------
    # two_subsets = {
    #     "virtassist": load_dataset("mkieffer/ACI-Bench", name="virtassist"),
    #     "aci": load_dataset("mkieffer/ACI-Bench", name="aci"),
    # }

    # # ---------------------------------------------------------------------
    # # 4) Load ALL subsets (virtassist, virtscribe, aci), all splits each
    # # ---------------------------------------------------------------------
    # all_subsets = {subset: load_dataset("mkieffer/ACI-Bench", name=subset) for subset in SUBSETS}
    # aci_all = all_subsets["aci"]  # DatasetDict
    # aci_train = aci_all["train"]  # Dataset
    # aci_valid = aci_all["valid"]

    # ---------------------------------------------------------------------
    # 5) Load multiple splits at once
    # ---------------------------------------------------------------------
    # load each split, concatenated
    aci_all_test_concat = load_dataset("mkieffer/ACI-Bench", name="aci", split=["train", "test1+test2+test3"])

    # load each split separately
    aci_all_test_separate = load_dataset("mkieffer/ACI-Bench", name="aci", split=["train", "test1", "test2", "test3"])