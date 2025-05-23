# VLURes
The repo contains data for the NeurIPS 2025 submission

# VLM Response Generation using OpenAI Batch API

This script automates the process of generating responses from Vision-Language Models (VLMs) like GPT-4o using OpenAI's Batch API. It's designed to process a dataset of images and image-text pairs across multiple tasks and languages.

## Prerequisites

1.  **Python 3:** Ensure Python 3.6 or higher is installed.
2.  **Required Python Libraries:** Install the necessary packages using pip:
    ```bash
    pip install openai tqdm
    ```
3.  **OpenAI API Key:** You need a valid OpenAI API key.

## Setup

1.  **Clone the Repository (if applicable):**
    If this script is part of a Git repository, clone it first.
    ```bash
    git clone https://github.com/anonymous2023-llama/VLURes.git
    cd VLURes
    ```

2.  **Set OpenAI API Key:**
    The script reads the API key from an environment variable named `OPENAI_API_KEY`. **Do not hardcode your API key into the script.** Set it in your terminal session before running the script:
    ```bash
    export OPENAI_API_KEY="your_actual_api_key_here"
    ```
    Replace `"your_actual_api_key_here"` with your real OpenAI API key. To make this permanent for your session or system, add this line to your shell's configuration file (e.g., `~/.bashrc`, `~/.zshrc`) and then source it (e.g., `source ~/.bashrc`).

3.  **Prepare Your Data:**
    *   Create a root data directory. By default, the script expects a folder named `data` in the same directory as the script:
        ```bash
        mkdir -p ./data
        ```
    *   Inside the `./data/` directory, place your specific dataset folder. The script is pre-configured for a dataset named `ImagesTextEn1K`. If your dataset folder has a different name, update the `dataset_name` field in the `BASE_CONFIG` section of the Python script.
        ```bash
        # Example: If your dataset is in ImagesTextEn1K.zip
        # mkdir -p ./data/ImagesTextEn1K
        # unzip /path/to/your/ImagesTextEn1K.zip -d ./data/ImagesTextEn1K/
        ```
    *   The dataset folder (e.g., `./data/ImagesTextEn1K/`) should contain:
        *   Image files (e.g., `image1.jpg`, `photo_002.png`).
        *   Corresponding text files for image-text tasks (e.g., `text1.txt`, `text_002.txt`). The script attempts to match images to texts based on numeric IDs in their filenames.

4.  **Configure Script (Optional):**
    Open the Python script (`run_vlm_batch.py`) and review the `BASE_CONFIG` dictionary near the top. You can adjust:
    *   `data_root_directory`: If your main data folder isn't `./data/`.
    *   `dataset_name`: The name of your specific image/text dataset folder.
    *   `output_directory_root`: Where output files (results, checkpoints, batch files) will be stored (defaults to `./outputs/`).
    *   `model_api_name`: The specific OpenAI model ID to use (e.g., "gpt-4o-2024-08-06").
    *   `batch_size`: Number of items to process in each API batch request.
    *   Other parameters like `max_tokens`, `temperature`, etc.

5.  **Make Script Executable:**
    Grant execute permissions to the script:
    ```bash
    chmod +x run_vlm_batch.py
    ```

## Running the Script

Execute the script from your terminal:

```bash
./run_vlm_batch.py
