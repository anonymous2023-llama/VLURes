# VLURes
The repo contains data for the NeurIPS 2025 submission

# VLM Response Generation Scripts

This repository contains scripts to automate the process of generating responses from Vision-Language Models (VLMs) using their respective APIs. It's designed to process a dataset of images and image-text pairs across multiple tasks and languages as part of the VLURes benchmark.

Currently, scripts are provided for:
1.  OpenAI's Batch API (e.g., for GPT-4o models)
2.  Google's Gemini API (e.g., for Gemini 1.5 Flash)

## Prerequisites

1.  **Python 3:** Ensure Python 3.6 or higher is installed.
2.  **Required Python Libraries:**
    *   **For OpenAI Script (`run_openai_batch.py`):**
        ```bash
        pip install openai tqdm
        ```
    *   **For Gemini Script (`run_gemini_async.py`):**
        ```bash
        pip install google-generativeai pillow python-dotenv tqdm nest_asyncio
        ```
    You can install all with:
    ```bash
    pip install openai google-generativeai pillow python-dotenv tqdm nest_asyncio
    ```
3.  **API Keys:**
    *   **OpenAI:** A valid OpenAI API key.
    *   **Google:** A valid Google API Key for Gemini.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/anonymous2023-llama/VLURes.git
    cd VLURes
    ```

2.  **Set API Keys:**
    The scripts read API keys from environment variables. **Do not hardcode your API keys into the scripts or commit them to the repository.**

    *   **For OpenAI Script (`run_openai_batch.py`):**
        Set the `OPENAI_API_KEY` environment variable:
        ```bash
        export OPENAI_API_KEY="your_actual_openai_api_key_here"
        ```

    *   **For Gemini Script (`run_gemini_async.py`):**
        Set the `GOOGLE_API_KEY` environment variable:
        ```bash
        export GOOGLE_API_KEY="your_actual_google_api_key_here"
        ```
        Alternatively, for the Gemini script, you can create a `.env` file in the `VLURes` directory with the following content:
        ```
        GOOGLE_API_KEY="your_actual_google_api_key_here"
        ```

    Replace placeholder keys with your real API keys. To make environment variables permanent for your session or system, add the `export` lines to your shell's configuration file (e.g., `~/.bashrc`, `~/.zshrc`) and then source it (e.g., `source ~/.bashrc`).

3.  **Prepare Your Data:**
    *   Create a root data directory within the `VLURes` repository. By default, the scripts expect a folder named `data`:
        ```bash
        # Assuming you are in the VLURes directory
        mkdir -p ./data
        ```
    *   Inside the `./data/` directory, place your specific dataset folder. The scripts are pre-configured for a dataset named `ImagesTextEn1K`. If your dataset folder has a different name, update the `dataset_name` field in the `BASE_CONFIG` section of the respective Python script.
        ```bash
        # Example: If your dataset is in ImagesTextEn1K.zip
        # mkdir -p ./data/ImagesTextEn1K
        # unzip /path/to/your/ImagesTextEn1K.zip -d ./data/ImagesTextEn1K/
        ```
    *   The dataset folder (e.g., `./data/ImagesTextEn1K/`) should contain:
        *   Image files (e.g., `image1.jpg`, `photo_002.png`).
        *   Corresponding text files for image-text tasks (e.g., `text1.txt`, `text_002.txt`). The scripts attempt to match images to texts based on numeric IDs in their filenames.

4.  **Configure Scripts (Optional):**
    Open the Python scripts (`run_openai_batch.py`, `run_gemini_async.py`) and review the `BASE_CONFIG` dictionary near the top of each. You can adjust:
    *   `data_root_directory`, `dataset_name`.
    *   `output_directory_root`, `model_name_for_paths` (used for naming output folders).
    *   `model_api_name`: The specific API model ID to use.
    *   Other parameters like `batch_size` (for OpenAI), `concurrency` (for Gemini), `max_output_tokens`, `temperature`, etc.

5.  **Make Scripts Executable:**
    Grant execute permissions to the scripts:
    ```bash
    chmod +x run_openai_batch.py
    chmod +x run_gemini_async.py
    ```

## Running the Scripts

Execute the desired script from your terminal (ensure you are in the `VLURes` directory):

*   **For OpenAI Models (e.g., GPT-4o):**
    ```bash
    ./run_openai_batch.py
    ```

*   **For Google Gemini Models:**
    ```bash
    ./run_gemini_async.py
    ```

Each script will:
*   Load images and corresponding text files.
*   Iterate through the specified languages and tasks defined in its `LANGUAGE_CONFIGS`.
*   For each language-task combination:
    *   Load existing checkpoint data to resume progress.
    *   Send requests to the respective API (using Batch API for OpenAI, concurrent async requests for Gemini).
    *   Poll for completion (for OpenAI Batch API).
    *   Process results and save them to JSON files.
    *   Save checkpoint data.

Output and checkpoint files will be saved in subdirectories under the path specified by `BASE_CONFIG["output_directory_root"]` in each script.

## Output Structure

The scripts will generate outputs (by default under `./outputs/` within the `VLURes` directory) in model-specific subfolders:
*   `./outputs/<model_name_for_paths>/results_1shot_rationales/<Language>/`: Contains the final JSON results for each task and language.
*   `./outputs/<model_name_for_paths>/checkpoints_1shot_rationales/`: Stores checkpoint files to resume processing.
*   `./outputs/common_batch_files/batch_files/` (For OpenAI script): Temporarily stores `.jsonl` files.

## Notes

*   **Cost:** Running these scripts will incur costs based on API usage for the respective services. Monitor your usage and costs in your OpenAI and Google Cloud dashboards.
*   **API Limits & Quotas:** Be mindful of API rate limits and quotas for your accounts. The OpenAI script uses the Batch API to mitigate some rate limit concerns. The Gemini script uses controlled concurrency.
*   **Error Handling:** The scripts include basic retry mechanisms. Check the console output and log messages for any errors during processing.
*   **Customization:** The `EXAMPLES` and `LANGUAGE_CONFIGS` dictionaries within each script define the 1-shot examples, system prompts, and task descriptions. Modify these to suit your specific experimental setup. Ensure the examples are appropriate for the model being used (e.g., the `EXAMPLES` are in English and used for cross-lingual prompting unless language-specific examples are provided in `LANGUAGE_CONFIGS`).
