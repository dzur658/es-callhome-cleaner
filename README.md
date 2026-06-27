# CallHome Spanish (Español) Cleaner
This repository contains a cleaner script (`main.py`) and a Jupyter notebook
in the `./experiments` directory. The cleaner script will simply produce the final dataset from the `spa.zip` file. If you want to see how I did my analysis on the transcripts, the Jupyter notebook shows my thoughts and tests.

## HuggingFace Dataset 🤗

If you want to use the resulting dataset, you can download the parquet file from HuggingFace [here](#)

## Acquire the Transcripts
The CallHome transcripts are distributed by TalkBank and not included in this repository. To acquire the CallHome dataset go [here](https://talkbank.org/ca/access/CallHome/spa.html) and click "Download transcripts". You will need to create a free account to download transcripts. The download will give you a file, `spa.zip`, with all 140 transcripts. <u>Do not open or unzip `spa.zip`!</u> Place `spa.zip` in the root of the repository and the scripts will read directly from the zip file.

## Quick Start
There are two ways to initialize the repository. Either way ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed before continuing for dependency management.

#### Clean Dataset (Method 1)
If you just wish to clean the dataset with my method, and don't want to perform any further experimentation or analysis on the raw dataset. You might not want the dev dependencies as heavy libraries like PyTorch and Transformers are included. Method 1 provides minimal dependency install and does not require ML models or dependencies.

**NOTE: Both methods (if left unchanged) yield the exact same final dataset**

1) Initialize the venv without the dev dependencies.

```bash
uv sync --no-dev
```

2) Run `main.py`. This will create `callhome_es_cleaned_transcript_pretraining_docs.parquet` in the root of the repository.

```bash
uv run main.py
```

#### Experiment with the Dataset (Method 2)
If you have a different use for the dataset, or want different information extracted, you can modify the Jupyter notebook in `./experiments/cpt_extraction.ipynb`. This notebook also shows my exploration of the dataset, and how I originally curated it.

**NOTE: The notebook itself can be viewed without installing dev dependencies**

1) Initialize the venv with dev dependencies

```bash
uv sync
```

Open `./experiments/cpt_extraction.ipynb` with your editor of choice.

## Intended Use Case

The intended use case of this project is to clean the Spanish CallHome transcripts, and prepare them into documents for Continued Pretraining (CPT) of a base LLM. The dataset could potentially be used to increase a model's awareness to how unscripted, and highly conversational Spanish exchanges occur. Each document in the resulting dataset is a reconstructed and cleaned transcript for each of the 140 calls in the CallHome Spanish transcripts. The resulting dataset contains a total of approximately 257,000 Spanish words, averaging about 1,800 words per document.

## System Requirements

- Cleaning the dataset with the prebuilt script does not require any specialized hardware due to the relatively small size of the dataset (~10 mb)
- It is recommended (but not required) to have access to a machine with a PyTorch compatible GPU (Cuda, Metal, Rocm) for running the LangID model if you will be doing development work/experimentation.

## Repository Structure

- `main.py`: The main script to execute for cleaning the corpus. Contains most of the settings passed to the helper functions for cleaning and also handles loading as well as exporting to parquet.
- `helper_functions.py`: Helper functions called by `main.py`. These functions contain the actual cleaning logic implementation.
- `./experiments/cpt_extraction.ipynb`: The experimental Jupyter notebook where the data was inspected and provides explanation for why cleaning had to be done, and my analysis of the corpus that led to certain cleaning decisions being made.

## Acknowledgements

TalkBank Citation:

```
Linguistic Data Consortium (2008). CABank Spanish CallHome Corpus. TalkBank. doi:10.21415/T51K54

Canavan, A., & Zipperlen, G. (1996). CALLHOME Spanish Speech LDC96S35. Philadelphia: Linguistic Data Consortium. 
```

Thank you to [sargosarker](https://huggingface.co/sagorsarker) for providing the Spanish/English [LangID model](https://huggingface.co/sagorsarker/codeswitch-spaeng-lid-lince) used for analysis.