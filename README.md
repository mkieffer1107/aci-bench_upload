**Errors:**

Many of the ACI-Bench data entries contain swapped speaker tags. We ran each transcript through a group of LLMs (GPT-5 mini, Grok 4.1 Fast, and Gemini 3 Flash Preview) to identify, as best as possible with a minimal setup, all such instances.

See more in the [corrections folder](corrections).

**How to run:**

First, create the venv:
```
uv venv --python 3.10 --seed
source .venv/bin/activate
uv sync
```

Login to the HF CLI:
```sh
huggingface-cli login 
```

And enter the configs you like:

**Challenge data only:**
```sh
cd aci-bench
chmod +x run.sh
./run.sh --username <username> \
         --repo ACI-Bench \
         --private false
```

The dataset now lives [on HuggingFace](https://huggingface.co/datasets/mkieffer/ACI-Bench).


**All data:**
```sh
cd aci-bench-medarc
chmod +x run.sh
./run.sh --username <username> \
         --repo ACI-Bench-MedARC \
         --private false
```

The dataset now lives [on HuggingFace](https://huggingface.co/datasets/mkieffer/ACI-Bench-MedARC).


All credit belongs to the original authors: [paper](https://www.nature.com/articles/s41597-023-02487-3), [data](https://github.com/wyim/aci-bench)
