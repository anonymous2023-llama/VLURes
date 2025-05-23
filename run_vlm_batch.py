#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================================
# Script for VLM Response Generation using OpenAI Batch API
#
# To run this script:
# 1. Ensure you have Python 3 installed.
# 2. Install necessary libraries:
#    pip install openai tqdm
# 3. Set your OpenAI API Key as an environment variable:
#    export OPENAI_API_KEY="your_actual_api_key_here"
# 4. Prepare your data:
#    - Create a root data directory (e.g., "./data/").
#    - Inside it, place your dataset folder (e.g., "ImagesTextEn1K").
#    - This folder should contain your images (e.g., image1.jpg) and
#      corresponding text files (e.g., text1.txt) for image-text tasks.
# 5. Adjust BASE_CONFIG below if your paths or model details differ.
# 6. Make the script executable (chmod +x your_script_name.py) and run it:
#    ./your_script_name.py
#
# Note: The original Cell 1 for unzipping in Colab is commented out.
# Ensure your data is accessible at the path specified in BASE_CONFIG.
# ================================

# ================================
# Cell 1: (Original Colab Cell - Data Loading/Unzipping)
# Data should be manually placed in the directory specified by BASE_CONFIG.
# ================================
# import zipfile
# import os
#
# zip_path = '/content/ImagesTextEn1K.zip' # Example path
# extract_path = './data/ImagesTextEn1K'   # Example relative path
#
# # Create the target directory if it doesn't exist
# os.makedirs(os.path.dirname(extract_path), exist_ok=True) # Ensure ./data exists
# os.makedirs(extract_path, exist_ok=True)
#
# # Unzip the file
# if os.path.exists(zip_path):
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_path)
#     print(f"Extracted {zip_path} to {extract_path}")
# else:
#     print(f"Zip file not found at {zip_path}. Ensure data is manually placed in {extract_path}")

# ================================
# Cell 2: Import Libraries and Define Configuration
# ================================
import os
import json
import base64
import time
import glob
from collections import OrderedDict
from tqdm import tqdm # Changed from tqdm.notebook
import tempfile

# Import OpenAI client
from openai import OpenAI

# Configure OpenAI API client
# API Key will be read from the OPENAI_API_KEY environment variable
try:
    client = OpenAI() # The client automatically looks for OPENAI_API_KEY
    if client.api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure the OPENAI_API_KEY environment variable is set correctly.")
    exit(1)

# Base configuration
BASE_CONFIG = {
    "data_root_directory": "./data/", # Root for all datasets
    "dataset_name": "ImagesTextEn1K", # Specific dataset folder within the root
    "output_directory_root": "./outputs/", # Root for all outputs
    "model_name_for_paths": "gpt-4o", # For creating model-specific output subfolders
    "checkpoint_subfolder": "checkpoints_1shot_rationales",
    "batch_files_subfolder": "batch_files",
    "results_subfolder": "results_1shot_rationales",
    "model_api_name": "gpt-4o-2024-08-06", # Actual model name for API calls
    "max_tokens": 1024,
    "temperature": 0,
    "batch_size": 200,
    "retry_delay": 5,
    "max_retries": 3,
}

# Construct full paths dynamically
BASE_CONFIG["data_input_directory"] = os.path.join(BASE_CONFIG["data_root_directory"], BASE_CONFIG["dataset_name"])
BASE_CONFIG["output_directory"] = os.path.join(BASE_CONFIG["output_directory_root"], BASE_CONFIG["model_name_for_paths"], BASE_CONFIG["results_subfolder"])
BASE_CONFIG["checkpoint_directory"] = os.path.join(BASE_CONFIG["output_directory_root"], BASE_CONFIG["model_name_for_paths"], BASE_CONFIG["checkpoint_subfolder"])
BASE_CONFIG["batch_directory"] = os.path.join(BASE_CONFIG["output_directory_root"], "common_batch_files", BASE_CONFIG["batch_files_subfolder"])


# Define 1-shot examples for prompting (English examples used for all languages)
EXAMPLES = {
    1: {
        "question": "Analyze this image and list all objects present. Categorize each object into groups such as furniture, electronic devices, clothing, etc. Be thorough and specific in your identification.",
        "example_response": "**Image Analysis**\n\n**Objects Present:**\n\n1. **Furniture:**\n   - Airport seating/benches\n\n2. **Electronic Devices:**\n   - Cameras\n   - Smartphones\n\n3. **Clothing:**\n   - Jackets\n   - Hats\n   - Glasses\n   - Scarves\n\n4. **Signs/Posters:**\n   - Various protest signs and posters\n   - American flag\n\n5. **Miscellaneous:**\n   - Backpacks\n   - Luggage\n   - Stanchions/barriers\n\nThis image appears to depict a gathering or protest in an airport setting."
    },
    2: {
        "question": "Describe the overall scene in this image. What is the setting, and what activities or events are taking place? Provide a comprehensive overview of the environment and any actions occurring.",
        "example_response": "**Image Analysis:**\n\nThe scene takes place in an airport terminal. The environment is bustling with a large group of people gathered together, indicating a protest or demonstration. Many individuals are seated on the floor, while others stand, holding signs and banners. The signs display various messages, suggesting that the gathering is organized around a specific cause or issue. One person is holding an American flag, which may indicate a connection to national or political themes. The atmosphere appears to be peaceful, with people engaged in the protest, some taking photos or videos. The setting is indoors, with typical airport features like seating areas and signage visible in the background."
    },
    3: {
        "question": "Identify any interactions or relationships between objects or entities in this image. How are they related or interacting with each other? Explain any spatial, functional, or social connections you observe.",
        "example_response": "**Analysis:**\n\nThe image depicts a large group of people gathered in what appears to be an airport terminal. The interactions and relationships between the objects and entities in the image can be described as follows:\n\n1. **Spatial Arrangement:**\n   - The people are closely packed together, indicating a collective gathering or event. Many are seated on the floor, while others are standing, suggesting limited space or a long-duration event.\n   - Signs and banners are held by several individuals, indicating a protest or demonstration.\n\n2. **Functional Connections:**\n   - The presence of signs and banners suggests that the group is participating in a protest or rally. The content of the signs, though not fully legible, appears to convey messages or slogans typical of such events.\n   - Some individuals are holding cameras or phones, likely documenting the event, which is common in public demonstrations.\n\n3. **Social Interactions:**\n   - The group appears to be unified in purpose, as indicated by their collective focus and the presence of similar signs. This suggests a shared cause or message.\n   - The presence of an American flag among the signs may indicate a connection to national issues or policies.\n\nOverall, the image captures a social and political interaction where individuals are gathered to express a collective viewpoint or protest, likely related to a specific cause or event."
    },
    4: {
        "question": "Divide this image into different semantic regions. Label each region (e.g., sky, buildings, people, street) and briefly describe its contents. Provide a clear breakdown of the image's composition.",
        "example_response": "**Image Analysis: Semantic Regions**\n\n1. **Foreground (People and Signs)**\n   - **Contents**: A group of people sitting and standing, holding signs. The signs have various messages, indicating a protest or demonstration. Some individuals are holding up phones, possibly recording or taking pictures.\n\n2. **Middle Ground (Crowd and Flags)**\n   - **Contents**: A larger crowd of people, some holding signs and an American flag. This area shows more participants in the demonstration, with a mix of standing and seated individuals.\n\n3. **Background (Airport Interior)**\n   - **Contents**: The interior of an airport terminal, with visible signage and architectural elements like pillars and ceiling lights. There are additional people in the background, contributing to the sense of a busy, crowded space.\n\n4. **Left Side (Signage)**\n   - **Contents**: A large sign with text related to TSA Pre✓, indicating the location within an airport. This provides context for the setting of the image.\n\n5. **Right Side (Seating Area)**\n   - **Contents**: Rows of airport seating, some occupied by people. This area is part of the terminal's waiting area, reinforcing the airport environment.\n\nOverall, the image depicts a protest or demonstration taking place inside an airport terminal, with various groups of people actively participating and observing."
    },
    5: {
        "question": "Provide a detailed, natural language description of what is happening in this image. Narrate the scene as if you were explaining it to someone who cannot see it, including all relevant details and actions.",
        "example_response": "**Image Analysis:**\n\nThe image depicts a large group of people gathered inside what appears to be an airport terminal. The scene is lively and crowded, with individuals engaged in a protest or demonstration. Many people are sitting on the floor, while others stand around them, holding signs and banners. The signs display various messages, some of which are partially visible, indicating themes of protest or advocacy. One sign prominently features the word \"Ban,\" suggesting opposition to a specific policy or action.\n\nIn the background, a person is holding an American flag, adding a patriotic element to the gathering. The crowd is diverse, with people of different ages and appearances, indicating a broad participation in the event. Some individuals are taking photos or videos, capturing the moment on their devices.\n\nThe setting is a busy terminal area, with signs indicating directions and services, such as TSA PreCheck. The atmosphere is one of organized activism, with participants seemingly united in their cause. The overall mood is serious yet peaceful, as people express their views and support for the issue at hand."
    },
    6: {
        "question": "Extract and list the specific parts of the text that closely match or directly reference entities, objects, or scenes depicted in the image. Be precise in identifying these connections and explain the visual evidence that supports each textual reference.",
        "example_response": "The image depicts a large group of people gathered in what appears to be an airport terminal. Many individuals are holding signs, and there is a visible presence of protest activity. This scene aligns with several elements from the text:\n\n1. **Location - Airport**: The text mentions John F. Kennedy International Airport in New York, where individuals were detained. The image shows a crowd in an airport setting, supporting this reference.\n\n2. **Protest Activity:** The text describes \"upheaval\" and \"widespread confusion\" at airports due to the executive order. The presence of protest signs and a large gathering of people in the image visually supports this description.\n\n3. **Signs and Messages:** Although the specific text on the signs in the image is not fully legible, the presence of signs suggests protest or demonstration, which is consistent with the text's mention of public reaction to the executive order.\n\n4. **Crowd and Atmosphere:** The image shows a diverse group of people, which aligns with the text's implication of a broad public response to the travel ban.\n\nThese visual elements in the image directly support the textual references to protests and detentions at airports following the executive order."
    },
    7: {
        "question": "Identify which parts of the text are not relevant to or not represented in the image. Explain why these elements are unrelated by describing what is missing in the image that would be needed to illustrate these textual elements.",
        "example_response": "The image depicts a protest scene at an airport, with people holding signs and an American flag, which aligns with the text's mention of protests and confusion at airports due to the travel ban. However, several elements from the text are not represented in the image:\n\n1. **Judicial Rulings:** The text discusses emergency rulings by judges in Brooklyn and Virginia halting deportations. The image does not show any courtrooms, judges, or legal proceedings.\n\n2. **Specific Individuals:** The text mentions Hameed Khalid Darweesh, an Iraqi translator, and other detained individuals. The image does not identify or depict any specific individuals mentioned in the text.\n\n3. **Government Statements:** The text includes statements from Donald Trump and descriptions from media outlets like The New York Times and The Verge. The image does not show any government officials or media coverage.\n\n4. **Detention and Boarding Prevention:** The text details the detention of individuals at JFK Airport and others being prevented from boarding planes. The image does not show any detention areas or boarding gates.\n\n5. **Legal Documents:** The text references legal documents like visas and green cards. The image does not depict any such documents.\n\nOverall, while the image captures the protest aspect related to the travel ban, it lacks representation of the legal, governmental, and individual-specific details provided in the text."
    },
    8: {
        "question": "What places are mentioned in the text or shown in the image? For each place identified, indicate whether it appears in the text, the image, or both. If any of these places are famous or well-known locations, explain why they are significant.",
        "example_response": "Based on the text and the image, the following places are mentioned or shown:\n\n1. **Brooklyn, New York**\n   - **Text:** Mentioned as the location where a U.S. federal judge issued an emergency ruling.\n   - **Significance:** Brooklyn is a well-known borough of New York City, significant for its cultural diversity and historical landmarks.\n\n2. **John F. Kennedy International Airport (JFK)**\n   - **Text:** Mentioned as the location where individuals were detained.\n   - **Image:** The image shows a scene likely at an airport, with people holding signs, which could be related to protests at JFK.\n   - **Significance:** JFK is one of the major airports in the U.S., located in New York City, and is a significant hub for international travel.\n\n3. **Virginia**\n   - **Text:** Mentioned as the location where another judge issued a similar ruling.\n   - **Significance:** Virginia is a U.S. state with historical importance, being one of the original thirteen colonies.\n\n4. **New York City**\n   - **Text:** Mentioned in relation to JFK Airport.\n   - **Significance:** New York City is a major global city known for its influence in finance, culture, and politics.\n\n5. **United States**\n   - **Text:** Mentioned multiple times in relation to the executive order and legal actions.\n   - **Significance:** The U.S. is a significant country globally, often involved in international political and legal matters.\n\nThe image likely depicts a protest or gathering at an airport, which aligns with the text's mention of protests and detentions at JFK Airport in New York City. The presence of signs and a crowd suggests a public demonstration, possibly in response to the executive order discussed in the text."
    }
}

# ================================
# Language-specific configurations with 1-shot examples and rationales
# ================================
LANGUAGE_CONFIGS = {
    "English": {
        "code": "En",
        "system_prompt": "You are an AI assistant that analyzes images and text.",
        "prompt_template_image_only": """
You are an intelligent assistant tasked with analyzing images.
Below is a 1-shot example (including the expected analysis and detailed reasoning) that demonstrates how to solve this task:

Example:
{one_shot_example}

Now, please perform the following task for the given image:
{task_description}

Provide your analysis with detailed step-by-step reasoning (rationales).
""",
        "prompt_template_image_text": """
You are an intelligent assistant tasked with analyzing the relationship between images and text.
Below is a 1-shot example (including the expected analysis and detailed reasoning) that demonstrates how to solve this task:

Example:
{one_shot_example}

Text associated with the image:
{text_content}

Task:
{task_description}

Provide your analysis with detailed step-by-step reasoning (rationales).
""",
        "tasks": {
            1: "Analyze this image and list all objects present. Categorize each object into groups such as furniture, electronic devices, clothing, etc. Be thorough and specific in your identification.",
            2: "Describe the overall scene in this image. What is the setting, and what activities or events are taking place? Provide a comprehensive overview of the environment and any actions occurring.",
            3: "Identify any interactions or relationships between objects or entities in this image. How are they related or interacting with each other? Explain any spatial, functional, or social connections you observe.",
            4: "Divide this image into different semantic regions. Label each region (e.g., sky, buildings, people, street) and briefly describe its contents. Provide a clear breakdown of the image's composition.",
            5: "Provide a detailed, natural language description of what is happening in this image. Narrate the scene as if you were explaining it to someone who cannot see it, including all relevant details and actions.",
            6: "Extract and list the specific parts of the text that closely match or directly reference entities, objects, or scenes depicted in the image. Be precise in identifying these connections and explain the visual evidence that supports each textual reference.",
            7: "Identify which parts of the text are not relevant to or not represented in the image. Explain why these elements are unrelated by describing what is missing in the image that would be needed to illustrate these textual elements.",
            8: "What places are mentioned in the text or shown in the image? For each place identified, indicate whether it appears in the text, the image, or both. If any of these places are famous or well-known locations, explain why they are significant."
        }
    },
    "Japanese": {
        "code": "Jp",
        "system_prompt": "あなたは画像とテキストを分析し、日本語で回答する AI アシスタントです。",
        "prompt_template_image_only": """
以下は、1-shot の例（期待される分析と詳細な根拠を含む）です。この例を参考にして、与えられた画像に対して次のタスクを実行してください：

例:
{one_shot_example}

タスク:
{task_description}

画像の分析結果と、ステップ・バイ・ステップの詳細な根拠を提供してください。
""",
        "prompt_template_image_text": """
以下は、画像とテキストの関係を分析するための 1-shot の例（期待される分析と詳細な根拠を含む）です。この例を参考にして、与えられた画像および関連テキストに基づいてタスクを実行してください：

例:
{one_shot_example}

画像に関連するテキスト:
{text_content}

タスク:
{task_description}

画像とテキストの両方に基づいた分析結果と、詳細な根拠を提供してください。
""",
        "tasks": {
            1: "この画像に存在するすべてのオブジェクトを分析し、家具、電子機器、衣類などのグループに分類してください。徹底的かつ具体的に識別してください。",
            2: "この画像の全体的な場面を説明してください。どのような環境で、どのような活動や出来事が起こっているかを包括的に記述してください。",
            3: "この画像内のオブジェクトや実体間の相互作用・関係を特定し、空間的、機能的、または社会的な接点を詳述してください。",
            4: "この画像を異なる意味領域に分割し、各領域（例：空、建物、人物、通り）の内容を簡潔に説明してください。",
            5: "この画像で何が起こっているかを、見ることのできない人に説明するかのように詳細に記述してください。",
            6: "テキストのうち、画像内に描かれているエンティティ、オブジェクト、またはシーンと密接に一致する部分を抽出し、視覚的証拠とともに説明してください。",
            7: "テキスト中の、画像に対応していない部分を特定し、それらがなぜ関連性がないかを説明してください。",
            8: "画像またはテキスト内で言及されている場所を特定し、それぞれが画像、テキスト、またはその両方に現れているかを示してください。有名な場所については、その重要性も説明してください。"
        }
    },
    "Swahili": {
        "code": "Sw",
        "system_prompt": "Wewe ni AI msaidizi unayechambua picha na maandishi na kutoa majibu kwa lugha ya Kiswahili.",
        "prompt_template_image_only": """
Hapa chini kuna mfano wa 1-shot (unaojumuisha uchambuzi na sababu za kina) wa jinsi ya kutekeleza kazi hii.
Mfano:
{one_shot_example}

Sasa, tafadhali fanya kazi ifuatayo kwa picha iliyotolewa:
{task_description}

Toa uchambuzi wako kwa hatua kwa hatua ukiongoza kwa sababu (rationales).
""",
        "prompt_template_image_text": """
Hapa chini kuna mfano wa 1-shot (unaojumuisha uchambuzi na sababu za kina) wa jinsi ya kutekeleza kazi hii.
Mfano:
{one_shot_example}

Maandishi yanayohusiana na picha:
{text_content}

Kazi:
{task_description}

Toa uchambuzi wako kwa hatua kwa hatua ukitumia ushahidi kutoka kwa picha na maandishi pamoja na sababu (rationales).
""",
        "tasks": {
            1: "Changanua picha hii na uorodheshe vitu vyote vilivyomo, ukae makini na wazi unapoweka kwenye makundi kama samani, vifaa vya elektroniki, mavazi, n.k.",
            2: "Elezea mandhari nzima katika picha hii, ukielezea mazingira na shughuli au matukio yanayofanyika.",
            3: "Tambua na elezea mwingiliano au uhusiano kati ya vitu au viumbe katika picha hii, ukizingatia uhusiano wa sehemu, shughuli, au kijamii.",
            4: "Gawanya picha hii katika maeneo tofauti ya maana, ukielezea yaliyomo katika kila eneo kwa ufupi na kwa uwazi.",
            5: "Toa maelezo ya kina ya kinachoendelea katika picha hii kama unavyoweza kuelezea kwa mtu asiyeiona.",
            6: "Toa orodha ya sehemu katika maandishi zinazofanana au zinazorejelea moja kwa moja vitu au matukio yaliyopo katika picha na elezea ushahidi wa kuona.",
            7: "Tambua ni sehemu gani za maandishi ambazo hazitingani na picha na elezea kwa nini hazilingani, ukielezea kile ambacho kinasahaulika katika picha.",
            8: "Elezea maeneo yaliyotajwa katika picha au maandishi, ukionyesha kama yanapatikana kwenye picha, maandishi, au vyote viwili, na kama ni muhimu elezea umuhimu wake."
        }
    },
    "Urdu": {
        "code": "Ur",
        "system_prompt": "آپ ایک ایسے AI اسسٹنٹ ہیں جو تصاویر اور متن کا تجزیہ کرتے ہیں اور اردو میں جوابات فراہم کرتے ہیں۔",
        "prompt_template_image_only": """
نیچے ایک 1-shot مثال (جس میں متوقع تجزیہ اور تفصیلی بنیادیں شامل ہیں) دی گئی ہے۔ اس مثال کو مدِنظر رکھتے ہوئے، براہِ کرم دی گئی تصویر کے لیے درج ذیل ٹاسک انجام دیں:

مثال:
{one_shot_example}

ٹاسک:
{task_description}

براہِ کرم اپنے تجزیے میں مرحلہ وار تفصیلی بنیادیں (rationales) شامل کریں۔
""",
        "prompt_template_image_text": """
نیچے ایک 1-shot مثال (جس میں متوقع تجزیہ اور تفصیلی بنیادیں شامل ہیں) دی گئی ہے۔ اس مثال کو مدِنظر رکھتے ہوئے، براہِ کرم تصویر اور متعلقہ متن کے مطابق ٹاسک انجام دیں:

مثال:
{one_shot_example}

تصویر سے متعلق متن:
{text_content}

ٹاسک:
{task_description}

براہِ کرم مرحلہ وار تفصیلی بنیادوں کے ساتھ اپنا تجزیہ فراہم کریں۔
""",
        "tasks": {
            1: "اس تصویر کا تجزیہ کریں اور موجود تمام اشیاء کو درجہ بندی کریں (مثلاً فرنیچر، الیکٹرانک آلات، کپڑے وغیرہ)۔",
            2: "تصویر میں مجموعی منظر کی وضاحت کریں کہ ماحول کیسا ہے اور کون سی سرگرمیاں جاری ہیں۔",
            3: "تصویر میں اشیاء یا افراد کے درمیان تعامل اور تعلقات کی نشاندہی کریں اور تفصیل سے بیان کریں۔",
            4: "تصویر کو مختلف معنی خیز علاقوں میں تقسیم کریں اور ہر علاقے کی مختصر وضاحت فراہم کریں۔",
            5: "تصویر میں کیا ہو رہا ہے اس کا تفصیلی بیانیہ پیش کریں جیسے کہ آپ کسی کو سنا رہے ہوں جو تصویر نہیں دیکھ سکتا۔",
            6: "متن کے ان حصوں کی نشاندہی کریں جو تصویر میں دکھائی دینے والے مناظر یا اشیاء کے ساتھ میل کھاتے ہوں اور انہیں واضح کریں۔",
            7: "متن کے ان حصوں کی نشاندہی کریں جو تصویر سے مطابقت نہیں رکھتے اور بتائیں کہ تصویر میں اُن کی عدم موجودگی کی وجہ کیا ہے۔",
            8: "متن یا تصویر میں ذکر کیے گئے مقامات کی شناخت کریں اور ظاہر کریں کہ وہ کس صورت میں موجود ہیں (متن، تصویر یا دونوں میں)؛ اگر کوئی مقام مشہور ہے تو اس کی اہمیت بیان کریں۔"
        }
    }
}

# Ensure output directories exist
os.makedirs(BASE_CONFIG["output_directory"], exist_ok=True)
os.makedirs(BASE_CONFIG["checkpoint_directory"], exist_ok=True)
os.makedirs(BASE_CONFIG["batch_directory"], exist_ok=True)

# ================================
# Cell 3: Define Utility Functions for File Processing and Batch API
# ================================
def get_checkpoint_filename(language, task_num):
    """Generate a consistent checkpoint filename"""
    lang_code = LANGUAGE_CONFIGS[language]["code"]
    return os.path.join(
        BASE_CONFIG["checkpoint_directory"],
        f"checkpoint_task{task_num}_{lang_code}.json"
    )

def load_checkpoint(language, task_num):
    """Load progress from checkpoint if it exists"""
    checkpoint_file = get_checkpoint_filename(language, task_num)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                print(f"Loaded checkpoint for {language} Task {task_num}")
                print(f"Previously processed {len(checkpoint_data)} items")
                return checkpoint_data
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch")
    return {}

def save_checkpoint(language, task_num, results):
    """Save current progress to checkpoint file"""
    checkpoint_file = get_checkpoint_filename(language, task_num)
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")

def encode_image(image_path):
    """Encode an image to base64 format"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def read_text_file(text_path):
    """Read text from a text file"""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        try:
            with open(text_path, 'r', encoding='latin-1') as f: # Fallback encoding
                return f.read().strip()
        except Exception as e_latin1:
            print(f"Error reading text file {text_path} with utf-8 and latin-1 encodings: {e_latin1}")
            return None
    except Exception as e:
        print(f"Error reading text file {text_path}: {e}")
        return None

def get_image_id(image_name):
    """Extract numeric ID from image filename (e.g., 'image123.jpg' -> 123)"""
    base = os.path.basename(image_name)
    name_part = os.path.splitext(base)[0]
    return int(''.join(filter(str.isdigit, name_part)))

def find_matching_text_file(image_path):
    """Find the corresponding text file for an image in the same directory."""
    directory = os.path.dirname(image_path)
    image_base_name = os.path.splitext(os.path.basename(image_path))[0]
    # Extract numeric ID part assuming format like 'image1', 'img001'
    image_id_str = ''.join(filter(str.isdigit, image_base_name))
    if not image_id_str: # If no digits found in image name, cannot reliably find text
        print(f"Warning: Could not extract numeric ID from image name: {os.path.basename(image_path)}")
        return None

    # Common text file naming conventions
    potential_text_names = [
        f"text{image_id_str}.txt",
        f"text{image_id_str}.text",
        f"{image_id_str}.txt",
        f"{image_id_str}.text",
        f"{image_base_name}.txt", # If text file has exact same name as image but .txt
        f"{image_base_name}.text"
    ]
    for text_name in potential_text_names:
        text_path = os.path.join(directory, text_name)
        if os.path.exists(text_path):
            return text_path
    # print(f"Warning: No matching text file found for image {image_path}")
    return None


def load_images_and_texts(directory):
    """Load image files and find their corresponding text files using full paths"""
    if not os.path.isdir(directory):
        print(f"Error: Data directory not found: {directory}")
        return [], [], []
        
    image_files_found = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Deduplicate and sort images by their numeric ID
    unique_images_dict = {}
    for img_path in image_files_found:
        try:
            img_id = get_image_id(img_path)
            if img_id not in unique_images_dict: # Keep first encountered if duplicate IDs (rare)
                 unique_images_dict[img_id] = img_path
        except ValueError:
            print(f"Warning: Could not parse ID from image {img_path}, skipping.")
            continue
    
    sorted_image_paths = [unique_images_dict[key] for key in sorted(unique_images_dict.keys())]

    image_text_pairs = []
    image_only = []

    for img_path in sorted_image_paths:
        text_path = find_matching_text_file(img_path)
        if text_path and os.path.exists(text_path):
            image_text_pairs.append((img_path, text_path))
        else:
            image_only.append(img_path)
            
    total_items_to_process = len(sorted_image_paths)
    print(f"Found {len(sorted_image_paths)} unique images to process from {directory}")
    print(f"Formed {len(image_text_pairs)} image-text pairs.")
    print(f"{len(image_only)} images will be processed as image-only (no matching text file).")

    # Limiting to 1000 as per original script logic, can be made configurable
    limit = 1000
    if total_items_to_process > limit:
        print(f"Warning: Found {total_items_to_process} items, limiting to first {limit} for processing.")
        return sorted_image_paths[:limit], image_text_pairs[:limit], image_only[:limit] # Ensure pairs/only are also sliced consistently

    # Adjust image_text_pairs and image_only if the main list was sliced
    # This logic needs care if slicing happens. For now, assume sorted_image_paths is the final list.
    # Rebuild image_text_pairs and image_only based on the final (potentially sliced) sorted_image_paths
    final_image_text_pairs = []
    final_image_only = []
    final_images_list = sorted_image_paths[:limit] if total_items_to_process > limit else sorted_image_paths

    for img_path in final_images_list:
        text_path = find_matching_text_file(img_path) # Re-find to match the potentially sliced list
        if text_path and os.path.exists(text_path):
            final_image_text_pairs.append((img_path, text_path))
        else:
            final_image_only.append(img_path)

    return final_images_list, final_image_text_pairs, final_image_only


def get_remaining_images(all_images, processed_ids_str_or_int):
    """Get list of images that haven't been processed yet. Handles string or int IDs."""
    remaining = []
    # Convert all keys in processed_ids to string for consistent checking
    processed_ids_str = {str(k) for k in processed_ids_str_or_int}

    for img_path in all_images:
        try:
            img_id = get_image_id(os.path.basename(img_path))
            if str(img_id) not in processed_ids_str:
                remaining.append(img_path)
        except ValueError:
            print(f"Warning: Could not parse ID from image {img_path} during remaining check.")
    return remaining

def get_remaining_image_text_pairs(all_pairs, processed_ids_str_or_int):
    """Get list of image-text pairs that haven't been processed yet. Handles string or int IDs."""
    remaining = []
    processed_ids_str = {str(k) for k in processed_ids_str_or_int}

    for img_path, text_path in all_pairs:
        try:
            img_id = get_image_id(os.path.basename(img_path))
            if str(img_id) not in processed_ids_str:
                remaining.append((img_path, text_path))
        except ValueError:
            print(f"Warning: Could not parse ID from image {img_path} in pair during remaining check.")
    return remaining


# ================================
# Cell 4: Define Functions for Preparing and Processing Batch Files
# ================================
def prepare_batch_file_image_only(image_paths, lang_config, task_prompt, model_api_name):
    batch_lines = []
    image_id_map = {}
    for img_path in image_paths:
        try:
            img_id = get_image_id(os.path.basename(img_path))
        except ValueError:
            print(f"Skipping image due to ID parsing error: {img_path}")
            continue
        custom_id = f"img_{img_id}_{int(time.time()*1000)}" # Add timestamp for more uniqueness
        image_id_map[custom_id] = {"path": img_path, "id": img_id}
        encoded_img = encode_image(img_path)
        if not encoded_img:
            print(f"Skipping image due to encoding error: {img_path}")
            continue
        batch_entry = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_api_name,
                "messages": [
                    {"role": "system", "content": lang_config["system_prompt"]},
                    {"role": "user", "content": [
                        {"type": "text", "text": task_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
                    ]}
                ],
                "max_tokens": BASE_CONFIG["max_tokens"],
                "temperature": BASE_CONFIG["temperature"]
            }
        }
        batch_lines.append(json.dumps(batch_entry))
    
    if not batch_lines:
        return None, {}

    # Use tempfile for batch file path to avoid collisions if script is run in parallel
    with tempfile.NamedTemporaryFile(mode='w', delete=False, prefix="batch_img_only_", suffix=".jsonl", dir=BASE_CONFIG["batch_directory"]) as tmp_file:
        tmp_file.write('\n'.join(batch_lines))
        batch_file_path = tmp_file.name
    return batch_file_path, image_id_map

def prepare_batch_file_image_text(image_text_pairs, lang_config, task_prompt_template, model_api_name):
    batch_lines = []
    pair_id_map = {}
    for img_path, text_path in image_text_pairs:
        try:
            img_id = get_image_id(os.path.basename(img_path))
        except ValueError:
            print(f"Skipping image-text pair due to ID parsing error on image: {img_path}")
            continue
        custom_id = f"pair_{img_id}_{int(time.time()*1000)}" # Add timestamp
        pair_id_map[custom_id] = {"image_path": img_path, "text_path": text_path, "id": img_id}
        
        text_content = read_text_file(text_path)
        encoded_img = encode_image(img_path)
        
        if not encoded_img:
            print(f"Skipping image-text pair due to image encoding error: {img_path}")
            continue
        if text_content is None : # Explicitly check for None if read_text_file can return it
            print(f"Skipping image-text pair due to text reading error: {text_path}")
            continue

        formatted_task_prompt = task_prompt_template.format(text_content=text_content) # text_content is for the task, one_shot_example already in template
        
        batch_entry = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_api_name,
                "messages": [
                    {"role": "system", "content": lang_config["system_prompt"]},
                    {"role": "user", "content": [
                        {"type": "text", "text": formatted_task_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
                    ]}
                ],
                "max_tokens": BASE_CONFIG["max_tokens"],
                "temperature": BASE_CONFIG["temperature"]
            }
        }
        batch_lines.append(json.dumps(batch_entry))

    if not batch_lines:
        return None, {}

    with tempfile.NamedTemporaryFile(mode='w', delete=False, prefix="batch_img_text_", suffix=".jsonl", dir=BASE_CONFIG["batch_directory"]) as tmp_file:
        tmp_file.write('\n'.join(batch_lines))
        batch_file_path = tmp_file.name
    return batch_file_path, pair_id_map

def upload_batch_file(file_path):
    """Upload a batch file and return its file ID"""
    try:
        with open(file_path, 'rb') as f:
            response = client.files.create(
                file=f,
                purpose="batch"
            )
        return response.id
    except Exception as e:
        print(f"Error uploading batch file {os.path.basename(file_path)}: {e}")
        return None
    finally:
        # Clean up the temporary batch file after upload attempt
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                # print(f"Removed temporary batch file: {os.path.basename(file_path)}")
            except Exception as e_rem:
                print(f"Warning: Could not remove temporary batch file {os.path.basename(file_path)}: {e_rem}")


def create_batch_job(file_id):
    """Create a batch job and return the batch object"""
    try:
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions", # Standard endpoint
            completion_window="24h"
        )
        return batch
    except Exception as e:
        print(f"Error creating batch job for file_id {file_id}: {e}")
        return None

def poll_batch_status(batch_id, max_polls=120, poll_interval=60): # Increased max_polls and interval
    """
    Poll the batch status until it completes or fails.
    Returns the final batch status object.
    """
    print(f"Polling status for batch ID: {batch_id}...")
    for i in range(max_polls):
        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            completed_req = batch.request_counts.completed if batch.request_counts else 0
            total_req = batch.request_counts.total if batch.request_counts else 0
            failed_req = batch.request_counts.failed if batch.request_counts else 0
            
            print(f"Batch status: {status} - Attempt {i+1}/{max_polls} - "
                  f"Completed: {completed_req}/{total_req}, Failed: {failed_req}")
            
            if status == "completed":
                print("Batch completed successfully.")
                return batch
            elif status in ["failed", "cancelled", "expired"]:
                print(f"Batch job {status}.")
                return batch
            time.sleep(poll_interval)
        except Exception as e:
            print(f"Error polling batch status for {batch_id}: {e}")
            print("Retrying poll...")
            time.sleep(poll_interval * 2) # Longer wait on error
    print(f"Reached maximum polling attempts ({max_polls}) for batch {batch_id}. Status may be incomplete.")
    # Try to retrieve one last time
    try:
        return client.batches.retrieve(batch_id)
    except Exception:
        return None


def process_batch_results(output_file_id, id_map):
    """
    Process the results of a completed batch job.
    Returns a dictionary mapping original item IDs to their analysis results.
    """
    if not output_file_id:
        print("No output file ID provided for processing results.")
        return {}
    try:
        # Retrieve file content
        response_content_obj = client.files.content(output_file_id)
        # The 'content' attribute of the HttpxBinaryResponse needs to be decoded.
        # For JSONL, it's typically UTF-8 text.
        content = response_content_obj.text # .read().decode('utf-8') if it were raw bytes

        results = {}
        if not content.strip():
            print("Warning: Batch output file is empty.")
            return {}
            
        for line_num, line in enumerate(content.strip().split('\n')):
            try:
                result_obj = json.loads(line)
                custom_id = result_obj.get('custom_id')
                
                if custom_id in id_map:
                    item_info = id_map[custom_id]
                    item_id = str(item_info.get('id')) # Ensure ID is string for dict keys
                    
                    response_data = result_obj.get('response', {})
                    if response_data and response_data.get('status_code') == 200:
                        body = response_data.get('body', {})
                        choices = body.get('choices', [])
                        if choices:
                            content_result = choices[0].get('message', {}).get('content', '')
                            results[item_id] = content_result
                        else:
                            results[item_id] = "Error: No choices in response."
                            print(f"Warning: No choices found for {custom_id} (ID: {item_id}). Full body: {body}")
                    else:
                        error_data = result_obj.get('error', {})
                        error_message = error_data.get('message', 'Unknown error')
                        status_code = response_data.get('status_code', 'N/A')
                        results[item_id] = f"Error: Status {status_code}, Message: {error_message}"
                        print(f"Error for {custom_id} (ID: {item_id}): Status {status_code}, {error_message}")
                else:
                    print(f"Warning: custom_id {custom_id} from batch result not found in id_map.")
            except json.JSONDecodeError as json_e:
                print(f"Error decoding JSON from line {line_num + 1} in batch result: {json_e}")
                print(f"Problematic line: {line[:200]}...") # Print start of line
            except Exception as e_line:
                print(f"Unexpected error processing line {line_num + 1} from batch result: {e_line}")

        return results
    except Exception as e:
        print(f"Fatal error processing batch results from file ID {output_file_id}: {e}")
        return {}

# ================================
# Cell 5: Define Main Processing Function with Batch API
# ================================
def process_task_for_language_batch(language, task_num, all_images, image_text_pairs):
    lang_config = LANGUAGE_CONFIGS[language]
    task_description = lang_config["tasks"][task_num]
    is_image_text_task = task_num >= 6 # Tasks 6, 7, 8 are image-text
    
    checkpoint_data = load_checkpoint(language, task_num)
    # Ensure checkpoint keys are strings if they were saved as int
    task_results = OrderedDict({str(k): v for k, v in checkpoint_data.items()})

    model_api_name = BASE_CONFIG["model_api_name"]
    
    current_items_to_process = []
    id_map_for_current_run = {}

    if is_image_text_task:
        prompt_template = lang_config["prompt_template_image_text"]
        task_prompt_preformatted = prompt_template.format(
            task_description=task_description,
            text_content="{text_content}", # Placeholder for actual text_content
            one_shot_example=EXAMPLES[task_num]["example_response"]
        )
        remaining_pairs = get_remaining_image_text_pairs(image_text_pairs, task_results.keys())
        items_for_batch_prep = remaining_pairs
        item_type_name = "image-text pairs"
        prepare_func = prepare_batch_file_image_text
        args_for_prepare = (lang_config, task_prompt_preformatted, model_api_name)

    else: # Image-only task
        prompt_template = lang_config["prompt_template_image_only"]
        task_prompt = prompt_template.format(
            task_description=task_description,
            one_shot_example=EXAMPLES[task_num]["example_response"]
        )
        remaining_images = get_remaining_images(all_images, task_results.keys())
        items_for_batch_prep = remaining_images
        item_type_name = "images"
        prepare_func = prepare_batch_file_image_only
        args_for_prepare = (lang_config, task_prompt, model_api_name)

    if not items_for_batch_prep:
        print(f"All {item_type_name} already processed for {language} Task {task_num} based on checkpoint.")
        return task_results # Return existing sorted results

    print(f"Processing Task {task_num} for {language}: {len(items_for_batch_prep)} {item_type_name} remaining.")

    # Process in batches
    for i in tqdm(range(0, len(items_for_batch_prep), BASE_CONFIG["batch_size"]), desc=f"{language} Task {task_num} Batches"):
        batch_items = items_for_batch_prep[i:i + BASE_CONFIG["batch_size"]]
        print(f"\nPreparing batch {i//BASE_CONFIG['batch_size'] + 1} with {len(batch_items)} {item_type_name}")
        
        batch_file_path, current_id_map = prepare_func(batch_items, *args_for_prepare)
        
        if not batch_file_path or not current_id_map:
            print("Failed to prepare batch file or ID map is empty. Skipping this batch.")
            continue

        print(f"Batch file created: {os.path.basename(batch_file_path)}")
        
        file_id = upload_batch_file(batch_file_path) # Temp file is removed in upload_batch_file
        if not file_id:
            print("Failed to upload batch file. Skipping this batch.")
            continue
        print(f"Batch file uploaded. File ID: {file_id}")

        retries = 0
        batch_succeeded = False
        while retries <= BASE_CONFIG["max_retries"] and not batch_succeeded:
            batch_job = create_batch_job(file_id)
            if not batch_job:
                print(f"Failed to create batch job (attempt {retries+1}). Retrying after delay if possible...")
                retries += 1
                if retries <= BASE_CONFIG["max_retries"]:
                    time.sleep(BASE_CONFIG["retry_delay"] * (2**retries)) # Exponential backoff
                continue
            
            print(f"Created batch job with ID: {batch_job.id}")
            completed_batch = poll_batch_status(batch_job.id)

            if completed_batch:
                if completed_batch.status == "completed":
                    print(f"Batch job {batch_job.id} completed.")
                    if completed_batch.output_file_id:
                        batch_run_results = process_batch_results(completed_batch.output_file_id, current_id_map)
                        task_results.update(batch_run_results)
                        save_checkpoint(language, task_num, task_results) # Save after each successful batch
                        print(f"Successfully processed and saved {len(batch_run_results)} items from batch {batch_job.id}")
                        batch_succeeded = True
                    else:
                        print(f"Error: Batch {batch_job.id} completed but no output_file_id found.")
                        handle_batch_errors(completed_batch.error_file_id, current_id_map)
                        # Decide if this counts as a retry or if we should stop for this batch
                        retries +=1 # Count as a retry attempt
                        if retries <= BASE_CONFIG["max_retries"]:
                           print(f"Retrying batch creation due to missing output file (attempt {retries}).")
                           time.sleep(BASE_CONFIG["retry_delay"])


                elif completed_batch.status in ["failed", "cancelled", "expired"]:
                    print(f"Batch job {batch_job.id} {completed_batch.status}.")
                    handle_batch_errors(completed_batch.error_file_id, current_id_map)
                    retries += 1
                    if retries <= BASE_CONFIG["max_retries"]:
                        print(f"Retrying batch creation (attempt {retries+1} of {BASE_CONFIG['max_retries']+1})...")
                        time.sleep(BASE_CONFIG["retry_delay"] * (2**retries))
                    else:
                        print(f"Max retries reached for batch from file {file_id}. Skipping remaining items in this batch.")
                        break # Break from retry loop, move to next chunk of items_for_batch_prep
                else: # e.g. validating, in_progress - should ideally not happen if poll_batch_status waits
                    print(f"Batch job {batch_job.id} in unexpected state: {completed_batch.status}. Treating as error for retry.")
                    retries += 1
                    if retries <= BASE_CONFIG["max_retries"]:
                         time.sleep(BASE_CONFIG["retry_delay"])
            else: # completed_batch is None
                print(f"Polling failed to retrieve final batch status for {batch_job.id} (attempt {retries+1}).")
                retries += 1
                if retries <= BASE_CONFIG["max_retries"]:
                     time.sleep(BASE_CONFIG["retry_delay"])
        
        if not batch_succeeded:
            print(f"Failed to process batch starting with item {i} after multiple retries. Moving to next batch if any.")
            # Potentially log these failed items for manual review

    # Sort final results by image ID (which are strings in the dict keys)
    sorted_results = OrderedDict(sorted(task_results.items(), key=lambda kv: int(kv[0])))
    return sorted_results


# ================================
# Cell 6: Functions to Handle Batch Errors and Cleanup Batch Files
# ================================
def cleanup_batch_files(keep_last_n=10): # Increased number to keep during active runs
    """
    Clean up old batch files in the batch_directory to save space.
    Keeps the most recent 'keep_last_n' .jsonl files.
    Note: This cleans files from BASE_CONFIG["batch_directory"], not tempfiles used during individual uploads.
    """
    try:
        # General cleanup of .jsonl files in the main batch directory
        main_batch_dir_files = glob.glob(os.path.join(BASE_CONFIG["batch_directory"], "*.jsonl"))
        main_batch_dir_files.sort(key=os.path.getmtime, reverse=True) # Sort by modification time, newest first
        
        for file_path in main_batch_dir_files[keep_last_n:]:
            try:
                os.remove(file_path)
                print(f"Cleaned up old batch file: {os.path.basename(file_path)}")
            except Exception as e_rem:
                print(f"Warning: Could not remove old batch file {os.path.basename(file_path)}: {e_rem}")
                
    except Exception as e:
        print(f"Error during batch file cleanup: {e}")


def handle_batch_errors(error_file_id, id_map):
    """
    Process and log the errors from a batch job's error file.
    """
    if not error_file_id:
        print("No error file ID provided for batch.")
        return
    try:
        response_content_obj = client.files.content(error_file_id)
        content = response_content_obj.text # Assuming text content for errors

        if not content.strip():
            print("Batch error file is empty or contains no processable errors.")
            return
            
        print("\n--- Batch Errors Reported ---")
        error_count = 0
        for line_num, line in enumerate(content.strip().split('\n')):
            try:
                error_obj = json.loads(line)
                custom_id = error_obj.get('custom_id')
                error_details = error_obj.get('error', {})
                response_details = error_obj.get('response', {}) # Sometimes errors are in response section
                
                item_id_display = "Unknown Item"
                if custom_id and custom_id in id_map:
                    item_info = id_map[custom_id]
                    item_id_display = f"Item ID: {item_info.get('id')} (CustomID: {custom_id})"

                if error_details:
                    error_count +=1
                    print(f"Error for {item_id_display}: Code: {error_details.get('code')}, Message: {error_details.get('message')}")
                elif response_details and response_details.get('status_code') != 200: # Check response for non-200s
                    error_count +=1
                    status_code = response_details.get('status_code')
                    response_body = response_details.get('body',{})
                    message = response_body.get('error',{}).get('message', 'Error in response body')
                    print(f"Error for {item_id_display}: HTTP Status {status_code}, Message: {message}")
                # else:
                #     print(f"Non-standard error object for {custom_id}: {error_obj}")

            except json.JSONDecodeError:
                print(f"Could not decode JSON from error line {line_num + 1}: {line[:200]}...")
            except Exception as e_line:
                print(f"Unexpected error processing error line {line_num + 1}: {e_line}")
        if error_count == 0:
            print("No specific errors parsed from error file content, though file existed.")
        else:
            print(f"Total errors parsed from error file: {error_count}")
        print("--- End of Batch Errors ---\n")

    except Exception as e:
        print(f"Error retrieving or processing batch error file {error_file_id}: {e}")

# ================================
# Cell 7: Main Execution - Process All Tasks and Languages
# ================================
def main():
    """Main function to orchestrate the VLM processing."""
    print("Starting VLM response generation...")
    print(f"Data will be loaded from: {BASE_CONFIG['data_input_directory']}")
    print(f"Outputs will be saved to subdirectories within: {BASE_CONFIG['output_directory_root']}")
    
    # Load images and text files from the specified directory
    # This assumes one common dataset for all language variations of prompts.
    all_images, image_text_pairs, _ = load_images_and_texts(BASE_CONFIG["data_input_directory"])

    if not all_images and not image_text_pairs:
        print(f"No images or image-text pairs found in {BASE_CONFIG['data_input_directory']}. Exiting.")
        return

    # Define the languages to process; you can process all or a subset.
    languages_to_process = ["English", "Japanese", "Swahili", "Urdu"]
    # languages_to_process = ["English"] # For testing a single language

    # Loop over languages and tasks.
    for lang in languages_to_process:
        if lang not in LANGUAGE_CONFIGS:
            print(f"Warning: Language '{lang}' not found in LANGUAGE_CONFIGS. Skipping.")
            continue
        
        # Create language-specific subfolder for results if it doesn't exist
        lang_output_dir = os.path.join(BASE_CONFIG["output_directory"], lang)
        os.makedirs(lang_output_dir, exist_ok=True)

        for task_num in range(1, 9): # Tasks 1 through 8
            print("\n" + "="*60)
            print(f"PROCESSING: Language: {lang}, Task: {task_num} (1-shot with Rationales)")
            print("="*60 + "\n")
            
            task_results = process_task_for_language_batch(lang, task_num, all_images, image_text_pairs)
            
            # Define output filename within the language-specific directory
            output_filename = os.path.join(
                lang_output_dir, # Save in language-specific folder
                f"results_{BASE_CONFIG['model_name_for_paths']}_1shot_{LANGUAGE_CONFIGS[lang]['code']}_task{task_num}_Rationales.json"
            )
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(task_results, f, ensure_ascii=False, indent=4)
                print(f"Saved results for {lang} Task {task_num} to {output_filename}")
            except Exception as e_save:
                print(f"Error saving results for {lang} Task {task_num} to {output_filename}: {e_save}")
            
            cleanup_batch_files() # Clean up general batch files area

    print("\nAll specified tasks processed!")

if __name__ == "__main__":
    main()