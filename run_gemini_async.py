#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================================
# Script for VLM Response Generation using Google Gemini API
#
# To run this script:
# 1. Ensure you have Python 3 installed.
# 2. Install necessary libraries:
#    pip install google-generativeai pillow python-dotenv tqdm nest_asyncio
# 3. Set your Google API Key for Gemini as an environment variable:
#    export GOOGLE_API_KEY="your_actual_google_api_key_here"
#    Alternatively, create a .env file in the script's directory with:
#    GOOGLE_API_KEY="your_actual_google_api_key_here"
# 4. Prepare your data:
#    - Create a root data directory (e.g., "./data/").
#    - Inside it, place your dataset folder (e.g., "ImagesTextEn1K").
#    - This folder should contain your images (e.g., image1.jpg) and
#      corresponding text files (e.g., text1.txt) for image-text tasks.
# 5. Adjust BASE_CONFIG below if your paths or model details differ.
# 6. Make the script executable (chmod +x your_gemini_script_name.py) and run it:
#    ./your_gemini_script_name.py
# ================================

import os
import json
import base64
import asyncio
import logging
from collections import OrderedDict
from tqdm.asyncio import tqdm as async_tqdm
from PIL import Image
from io import BytesIO
import nest_asyncio
import google.generativeai as genai
from dotenv import load_dotenv

# Allow nested asyncio loops (useful in notebooks, harmless in scripts)
nest_asyncio.apply()
load_dotenv()  # Load GOOGLE_API_KEY from .env file if it exists

# ================================
# Configuration
# ================================
BASE_CONFIG = {
    "data_root_directory": "./data/",
    "dataset_name": "ImagesTextEn1K",
    "output_directory_root": "./outputs/",
    "model_name_for_paths": "gemini-2.0-flash-lite", # For creating model-specific output subfolders
    "checkpoint_subfolder": "checkpoints_1shot_rationales",
    # "batch_directory": "./outputs/gemini_batch_files/", # Gemini API doesn't use batch files like OpenAI's Batch API
    "results_subfolder": "results_1shot_rationales",
    "model_api_name": "gemini-1.5-flash-latest", # Or specific version like "models/gemini-1.5-flash-001"
                                                 # For "gemini-2.0-flash-lite-001" - ensure this model ID is valid for the API
    "concurrency": 10, # Max concurrent requests to Gemini API
    "max_output_tokens": 1024,
    "temperature": 0.0,
    "retry_attempts": 3,
    "retry_delay": 5, # Seconds
    "progress_update_frequency": 10, # Not directly used with tqdm, but good to keep for other contexts
}

# Construct full paths dynamically
BASE_CONFIG["data_input_directory"] = os.path.join(BASE_CONFIG["data_root_directory"], BASE_CONFIG["dataset_name"])
BASE_CONFIG["output_directory"] = os.path.join(BASE_CONFIG["output_directory_root"], BASE_CONFIG["model_name_for_paths"], BASE_CONFIG["results_subfolder"])
BASE_CONFIG["checkpoint_directory"] = os.path.join(BASE_CONFIG["output_directory_root"], BASE_CONFIG["model_name_for_paths"], BASE_CONFIG["checkpoint_subfolder"])
# BASE_CONFIG["batch_files_directory"] = os.path.join(BASE_CONFIG["output_directory_root"], "common_batch_files", BASE_CONFIG["batch_files_subfolder"]) # Not needed for Gemini direct API

# 1-shot examples (English examples used as base for all languages unless overridden)
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


# Language-specific configurations
LANGUAGE_CONFIGS = {
    "English": {
        "code": "En",
        "system_prompt": "You are an AI assistant that analyzes images and text. Provide your analysis with detailed step-by-step reasoning (rationales).",
        "prompt_template_image_only": """Below is a 1-shot example (including the expected analysis and detailed reasoning) that demonstrates how to solve this task:

Example Question: {example_question}
Example Response:
{example_response}

Now, please perform the following task for the given image:
{task_description}""",
        "prompt_template_image_text": """Below is a 1-shot example (including the expected analysis and detailed reasoning) that demonstrates how to solve this task:

Example Question: {example_question}
Example Response:
{example_response}

Text associated with the image:
{text_content}

Task:
{task_description}""",
        "tasks": { # Using your more detailed English tasks
            1: "Analyze this image and list all objects present. Categorize each object into groups such as furniture, electronic devices, clothing, etc. Be thorough and specific in your identification.",
            2: "Describe the overall scene in this image. What is the setting, and what activities or events are taking place? Provide a comprehensive overview of the environment and any actions occurring.",
            3: "Identify any interactions or relationships between objects or entities in this image. How are they related or interacting with each other? Explain any spatial, functional, or social connections you observe.",
            4: "Divide this image into different semantic regions. Label each region (e.g., sky, buildings, people, street) and briefly describe its contents. Provide a clear breakdown of the image's composition.",
            5: "Provide a detailed, natural language description of what is happening in this image. Narrate the scene as if you were explaining it to someone who cannot see it, including all relevant details and actions.",
            6: "Extract and list the specific parts of the text that closely match or directly reference entities, objects, or scenes depicted in the image. Be precise in identifying these connections and explain the visual evidence that supports each textual reference.",
            7: "Identify which parts of the text are not relevant to or not represented in the image. Explain why these elements are unrelated by describing what is missing in the image that would be needed to illustrate these textual elements.",
            8: "What places are mentioned in the text or shown in the image? For each place identified, indicate whether it appears in the text, the image, or both. If any of these places are famous or well-known locations, explain why they are significant."
        },
        "examples": EXAMPLES # Use the globally defined English EXAMPLES
    },
    "Japanese": {
        "code": "Jp",
        "system_prompt": "あなたは画像とテキストを分析し、日本語で回答する AI アシスタントです。ステップ・バイ・ステップの詳細な根拠も提供してください。",
        "prompt_template_image_only": """以下は、1-shot の例（期待される分析と詳細な根拠を含む）です。この例を参考にして、与えられた画像に対して次のタスクを実行してください：

例の質問: {example_question}
例の回答:
{example_response}

タスク:
{task_description}""",
        "prompt_template_image_text": """以下は、画像とテキストの関係を分析するための 1-shot の例（期待される分析と詳細な根拠を含む）です。この例を参考にして、与えられた画像および関連テキストに基づいてタスクを実行してください：

例の質問: {example_question}
例の回答:
{example_response}

画像に関連するテキスト:
{text_content}

タスク:
{task_description}""",
        "tasks": { # Using your Japanese tasks
            1: "この画像に存在するすべてのオブジェクトを分析し、家具、電子機器、衣類などのグループに分類してください。徹底的かつ具体的に識別してください。",
            2: "この画像の全体的な場面を説明してください。どのような環境で、どのような活動や出来事が起こっているかを包括的に記述してください。",
            3: "この画像内のオブジェクトや実体間の相互作用・関係を特定し、空間的、機能的、または社会的な接点を詳述してください。",
            4: "この画像を異なる意味領域に分割し、各領域（例：空、建物、人物、通り）の内容を簡潔に説明してください。",
            5: "この画像で何が起こっているかを、見ることのできない人に説明するかのように詳細に記述してください。",
            6: "テキストのうち、画像内に描かれているエンティティ、オブジェクト、またはシーンと密接に一致する部分を抽出し、視覚的証拠とともに説明してください。",
            7: "テキスト中の、画像に対応していない部分を特定し、それらがなぜ関連性がないかを説明してください。",
            8: "画像またはテキスト内で言及されている場所を特定し、それぞれが画像、テキスト、またはその両方に現れているかを示してください。有名な場所については、その重要性も説明してください。"
        },
        "examples": EXAMPLES # Using English examples for 1-shot, assuming cross-lingual transfer
    },
    "Swahili": {
        "code": "Sw",
        "system_prompt": "Wewe ni AI msaidizi unayechambua picha na maandishi na kutoa majibu kwa lugha ya Kiswahili. Toa uchambuzi wako kwa hatua kwa hatua ukiongoza kwa sababu (rationales).",
        "prompt_template_image_only": """Hapa chini kuna mfano wa 1-shot (unaojumuisha uchambuzi na sababu za kina) wa jinsi ya kutekeleza kazi hii.
Mfano Swali: {example_question}
Mfano Jibu:
{example_response}

Sasa, tafadhali fanya kazi ifuatayo kwa picha iliyotolewa:
{task_description}""",
        "prompt_template_image_text": """Hapa chini kuna mfano wa 1-shot (unaojumuisha uchambuzi na sababu za kina) wa jinsi ya kutekeleza kazi hii.
Mfano Swali: {example_question}
Mfano Jibu:
{example_response}

Maandishi yanayohusiana na picha:
{text_content}

Kazi:
{task_description}""",
        "tasks": { # Using your Swahili tasks
            1: "Changanua picha hii na uorodheshe vitu vyote vilivyomo, ukae makini na wazi unapoweka kwenye makundi kama samani, vifaa vya elektroniki, mavazi, n.k.",
            2: "Elezea mandhari nzima katika picha hii, ukielezea mazingira na shughuli au matukio yanayofanyika.",
            3: "Tambua na elezea mwingiliano au uhusiano kati ya vitu au viumbe katika picha hii, ukizingatia uhusiano wa sehemu, shughuli, au kijamii.",
            4: "Gawanya picha hii katika maeneo tofauti ya maana, ukielezea yaliyomo katika kila eneo kwa ufupi na kwa uwazi.",
            5: "Toa maelezo ya kina ya kinachoendelea katika picha hii kama unavyoweza kuelezea kwa mtu asiyeiona.",
            6: "Toa orodha ya sehemu katika maandishi zinazofanana au zinazorejelea moja kwa moja vitu au matukio yaliyopo katika picha na elezea ushahidi wa kuona.",
            7: "Tambua ni sehemu gani za maandishi ambazo hazitingani na picha na elezea kwa nini hazilingani, ukielezea kile ambacho kinasahaulika katika picha.",
            8: "Elezea maeneo yaliyotajwa katika picha au maandishi, ukionyesha kama yanapatikana kwenye picha, maandishi, au vyote viwili, na kama ni muhimu elezea umuhimu wake."
        },
        "examples": EXAMPLES # Using English examples for 1-shot
    },
    "Urdu": {
        "code": "Ur",
        "system_prompt": "آپ ایک ایسے AI اسسٹنٹ ہیں جو تصاویر اور متن کا تجزیہ کرتے ہیں اور اردو میں جوابات فراہم کرتے ہیں۔ براہِ کرم اپنے تجزیے میں مرحلہ وار تفصیلی بنیادیں (rationales) شامل کریں۔",
        "prompt_template_image_only": """نیچے ایک 1-shot مثال (جس میں متوقع تجزیہ اور تفصیلی بنیادیں شامل ہیں) دی گئی ہے۔ اس مثال کو مدِنظر رکھتے ہوئے، براہِ کرم دی گئی تصویر کے لیے درج ذیل ٹاسک انجام دیں:

مثال کا سوال: {example_question}
مثال کا جواب:
{example_response}

ٹاسک:
{task_description}""",
        "prompt_template_image_text": """نیچے ایک 1-shot مثال (جس میں متوقع تجزیہ اور تفصیلی بنیادیں شامل ہیں) دی گئی ہے۔ اس مثال کو مدِنظر رکھتے ہوئے، براہِ کرم تصویر اور متعلقہ متن کے مطابق ٹاسک انجام دیں:

مثال کا سوال: {example_question}
مثال کا جواب:
{example_response}

تصویر سے متعلق متن:
{text_content}

ٹاسک:
{task_description}""",
        "tasks": { # Using your Urdu tasks
            1: "اس تصویر کا تجزیہ کریں اور موجود تمام اشیاء کو درجہ بندی کریں (مثلاً فرنیچر، الیکٹرانک آلات، کپڑے وغیرہ)۔",
            2: "تصویر میں مجموعی منظر کی وضاحت کریں کہ ماحول کیسا ہے اور کون سی سرگرمیاں جاری ہیں۔",
            3: "تصویر میں اشیاء یا افراد کے درمیان تعامل اور تعلقات کی نشاندہی کریں اور تفصیل سے بیان کریں۔",
            4: "تصویر کو مختلف معنی خیز علاقوں میں تقسیم کریں اور ہر علاقے کی مختصر وضاحت فراہم کریں۔",
            5: "تصویر میں کیا ہو رہا ہے اس کا تفصیلی بیانیہ پیش کریں جیسے کہ آپ کسی کو سنا رہے ہوں جو تصویر نہیں دیکھ سکتا۔",
            6: "متن کے ان حصوں کی نشاندہی کریں جو تصویر میں دکھائی دینے والے مناظر یا اشیاء کے ساتھ میل کھاتے ہوں اور انہیں واضح کریں۔",
            7: "متن کے ان حصوں کی نشاندہی کریں جو تصویر سے مطابقت نہیں رکھتے اور بتائیں کہ تصویر میں اُن کی عدم موجودگی کی وجہ کیا ہے۔",
            8: "متن یا تصویر میں ذکر کیے گئے مقامات کی شناخت کریں اور ظاہر کریں کہ وہ کس صورت میں موجود ہیں (متن، تصویر یا دونوں میں)؛ اگر کوئی مقام مشہور ہے تو اس کی اہمیت بیان کریں۔"
        },
        "examples": EXAMPLES # Using English examples for 1-shot
    }
}


# Ensure output directories exist
os.makedirs(BASE_CONFIG["output_directory"], exist_ok=True)
os.makedirs(BASE_CONFIG["checkpoint_directory"], exist_ok=True)
# os.makedirs(BASE_CONFIG["batch_files_directory"], exist_ok=True) # Not needed for Gemini direct API

# ================================
# Utility Functions
# ================================
def setup_logging():
    """Sets up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()] # Log to console
    )

def get_checkpoint_filename(language, task_num):
    """Generate a consistent checkpoint filename."""
    lang_code = LANGUAGE_CONFIGS[language]["code"]
    return os.path.join(
        BASE_CONFIG["checkpoint_directory"],
        f"checkpoint_task{task_num}_{lang_code}.json"
    )

def load_checkpoint(language, task_num):
    """Load progress from checkpoint if it exists."""
    checkpoint_file = get_checkpoint_filename(language, task_num)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                logging.info(f"Loaded checkpoint for {language} Task {task_num} with {len(checkpoint_data)} items.")
                return checkpoint_data
        except Exception as e:
            logging.warning(f"Error loading checkpoint {checkpoint_file}: {e}. Starting fresh.")
    return {}

def save_checkpoint(language, task_num, results):
    """Save current progress to checkpoint file."""
    checkpoint_file = get_checkpoint_filename(language, task_num)
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.warning(f"Warning: Failed to save checkpoint {checkpoint_file}: {e}")

def encode_image_to_base64(image_path):
    """Encodes an image file to base64 and determines its MIME type."""
    try:
        img = Image.open(image_path)
        # Ensure image is loaded, especially for some formats like webp
        img.load() 
        # Determine format, default to JPEG if not available or common
        img_format = img.format if img.format else 'JPEG'
        # Get MIME type, default to image/jpeg
        mime_type = Image.MIME.get(img_format.upper(), 'image/jpeg')

        buffered = BytesIO()
        # Convert to RGB if it has an alpha channel (e.g., PNG) to avoid issues with JPEG saving
        if img.mode == 'RGBA' or img.mode == 'LA' or (img.mode == 'P' and 'transparency' in img.info):
             if img_format.upper() == 'JPEG': # JPEG doesn't support alpha
                img = img.convert('RGB')
        
        img.save(buffered, format=img_format)
        img_byte_arr = buffered.getvalue()
        base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
        return base64_str, mime_type
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None, None

def read_text_file(text_path):
    """Read text from a text file with fallback encodings."""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        try:
            with open(text_path, 'r', encoding='latin-1') as f:
                return f.read().strip()
        except Exception as e_latin1:
            logging.error(f"Error reading text file {text_path} with utf-8 and latin-1: {e_latin1}")
            return None
    except Exception as e:
        logging.error(f"Error reading text file {text_path}: {e}")
        return None

def get_image_id_from_path(image_path):
    """Extract numeric ID from image filename (e.g., 'image123.jpg' -> 123)."""
    image_name = os.path.basename(image_path)
    name_part = os.path.splitext(image_name)[0]
    digits = ''.join(filter(str.isdigit, name_part))
    try:
        return int(digits) if digits else None
    except ValueError: # Should not happen if filter(str.isdigit) is used, but good practice
        logging.warning(f"Could not convert digits '{digits}' to int for {image_name}")
        return None


def find_matching_text_file(image_path):
    """Find the corresponding text file for an image in the same directory."""
    directory = os.path.dirname(image_path)
    image_base_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]
    
    # Try exact name match first (e.g., image1.jpg -> image1.txt)
    exact_match_txt = os.path.join(directory, image_base_name_no_ext + ".txt")
    if os.path.exists(exact_match_txt):
        return exact_match_txt
    exact_match_text = os.path.join(directory, image_base_name_no_ext + ".text")
    if os.path.exists(exact_match_text):
        return exact_match_text

    # Try ID-based match (e.g., image123.jpg -> text123.txt)
    image_id_num = get_image_id_from_path(image_path)
    if image_id_num is not None:
        potential_text_names = [
            f"text{image_id_num}.txt",
            f"text{image_id_num}.text",
            f"{image_id_num}.txt",
            f"{image_id_num}.text"
        ]
        for text_name in potential_text_names:
            text_path = os.path.join(directory, text_name)
            if os.path.exists(text_path):
                return text_path
    return None


def load_data_items(data_dir):
    """Loads images and finds corresponding text files."""
    if not os.path.isdir(data_dir):
        logging.error(f"Data directory not found: {data_dir}")
        return [], []

    all_files = os.listdir(data_dir)
    image_paths_found = sorted(
        [os.path.join(data_dir, f) for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))],
        key=lambda p: get_image_id_from_path(p) if get_image_id_from_path(p) is not None else float('inf') # Sort by ID
    )
    
    # Limit to 1000 items as per original implicit logic
    # This limit should ideally be configurable or based on dataset splits
    limit = 1000 
    if len(image_paths_found) > limit:
        logging.warning(f"Found {len(image_paths_found)} images, limiting to first {limit}.")
        image_paths_found = image_paths_found[:limit]

    image_only_items = []
    image_text_items = []

    for img_path in image_paths_found:
        text_path = find_matching_text_file(img_path)
        if text_path and os.path.exists(text_path):
            image_text_items.append({'image_path': img_path, 'text_path': text_path})
        else:
            image_only_items.append({'image_path': img_path})
            
    logging.info(f"Loaded {len(image_paths_found)} images for processing.")
    logging.info(f"Found {len(image_text_items)} image-text pairs.")
    logging.info(f"{len(image_only_items)} images will be processed as image-only.")
    return image_only_items, image_text_items


# ================================
# Gemini API Interaction
# ================================
async def generate_gemini_response(model, prompt_parts, retry_attempts, retry_delay):
    """Sends a request to Gemini and handles retries."""
    for attempt in range(retry_attempts):
        try:
            # Use asyncio.to_thread for synchronous SDK calls in an async context
            response = await asyncio.to_thread(model.generate_content, prompt_parts)
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            else: # Handle cases like safety blocks or empty responses
                finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                safety_ratings = response.candidates[0].safety_ratings if response.candidates else []
                logging.warning(f"No content in response. Finish reason: {finish_reason}. Safety: {safety_ratings}")
                return f"Error: No content in response (Finish reason: {finish_reason}, Safety: {safety_ratings})"

        except Exception as e:
            logging.error(f"Gemini API error (attempt {attempt + 1}/{retry_attempts}): {e}")
            if attempt < retry_attempts - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt)) # Exponential backoff
            else:
                return f"Error: After {retry_attempts} attempts - {e}"
    return "Error: Max retries reached without success."


async def process_single_item_gemini(item_data, language, task_num, model_instance, semaphore):
    """Processes a single image or image-text pair with Gemini."""
    async with semaphore: # Control concurrency
        image_path = item_data['image_path']
        text_path = item_data.get('text_path') # Will be None for image-only

        img_id = get_image_id_from_path(image_path)
        if img_id is None:
            return os.path.basename(image_path), "Error: Could not parse image ID"

        lang_config = LANGUAGE_CONFIGS[language]
        task_description = lang_config["tasks"][task_num]
        one_shot_example_q = lang_config["examples"][task_num]["question"]
        one_shot_example_r = lang_config["examples"][task_num]["example_response"]

        base64_image, mime_type = encode_image_to_base64(image_path)
        if not base64_image:
            return str(img_id), "Error: Image encoding failed"

        image_part = {"mime_type": mime_type, "data": base64_image}
        prompt_parts = []

        if text_path: # Image-text task
            text_content = read_text_file(text_path)
            if text_content is None:
                return str(img_id), "Error: Could not read text file"
            
            full_prompt_text = lang_config["prompt_template_image_text"].format(
                example_question=one_shot_example_q,
                example_response=one_shot_example_r,
                text_content=text_content,
                task_description=task_description
            )
            prompt_parts = [full_prompt_text, image_part]
        else: # Image-only task
            full_prompt_text = lang_config["prompt_template_image_only"].format(
                example_question=one_shot_example_q,
                example_response=one_shot_example_r,
                task_description=task_description
            )
            prompt_parts = [full_prompt_text, image_part]
        
        response_text = await generate_gemini_response(
            model_instance,
            prompt_parts,
            BASE_CONFIG["retry_attempts"],
            BASE_CONFIG["retry_delay"]
        )
        return str(img_id), response_text


async def run_task_for_language_gemini(language, task_num, image_only_items, image_text_items, model_instance):
    """Processes all items for a given language and task using Gemini."""
    
    is_image_text_task = task_num >= 6
    items_to_process_info = image_text_items if is_image_text_task else image_only_items
    
    # Load checkpoint and determine remaining items
    checkpoint_data = load_checkpoint(language, task_num)
    task_results = OrderedDict({str(k): v for k, v in checkpoint_data.items()}) # Ensure keys are strings
    processed_ids = set(task_results.keys())

    remaining_items = []
    for item_data in items_to_process_info:
        img_id = get_image_id_from_path(item_data['image_path'])
        if img_id is not None and str(img_id) not in processed_ids:
            remaining_items.append(item_data)

    if not remaining_items:
        logging.info(f"All items already processed for {language} Task {task_num} based on checkpoint.")
        return task_results # Return existing sorted results

    logging.info(f"Processing {len(remaining_items)} remaining items for {language} Task {task_num}.")

    semaphore = asyncio.Semaphore(BASE_CONFIG["concurrency"])
    async_tasks = [
        process_single_item_gemini(item_data, language, task_num, model_instance, semaphore)
        for item_data in remaining_items
    ]

    for future in async_tqdm(asyncio.as_completed(async_tasks), total=len(async_tasks), desc=f"{language} Task {task_num}"):
        item_id_str, result_text = await future
        task_results[item_id_str] = result_text
        # Potentially save checkpoint more frequently, e.g., every N results
        if len(task_results) % BASE_CONFIG["progress_update_frequency"] == 0:
             save_checkpoint(language, task_num, task_results)
    
    save_checkpoint(language, task_num, task_results) # Final save for the task
    
    # Sort final results by image ID (which are strings in the dict keys)
    sorted_results = OrderedDict(sorted(task_results.items(), key=lambda kv: int(kv[0])))
    return sorted_results

# ================================
# Main Execution
# ================================
def main():
    """Main function to orchestrate the VLM processing with Gemini."""
    setup_logging()

    # Initialize Gemini API
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    if not google_api_key:
        logging.error("GOOGLE_API_KEY environment variable not set. Please set it to your Gemini API key.")
        # Optionally, prompt for it, but environment variable is preferred for scripts
        # google_api_key = input("Please enter your GOOGLE_API_KEY: ").strip()
        # if not google_api_key:
        #     print("No API key provided. Exiting.")
        #     return
        return # Exit if no key
        
    try:
        genai.configure(api_key=google_api_key)
        # Instantiate the model once for reuse
        model_instance = genai.GenerativeModel(
            BASE_CONFIG["model_api_name"],
            system_instruction=None, # System prompt is set per language later
            generation_config=genai.GenerationConfig(
                temperature=BASE_CONFIG["temperature"],
                max_output_tokens=BASE_CONFIG["max_output_tokens"]
            )
        )
        logging.info(f"Successfully initialized Gemini model: {BASE_CONFIG['model_api_name']}")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model: {e}")
        return

    logging.info(f"Data will be loaded from: {BASE_CONFIG['data_input_directory']}")
    logging.info(f"Outputs will be saved to subdirectories within: {BASE_CONFIG['output_directory_root']}")

    image_only_data, image_text_data = load_data_items(BASE_CONFIG["data_input_directory"])

    if not image_only_data and not image_text_data:
        logging.error(f"No items found in {BASE_CONFIG['data_input_directory']}. Exiting.")
        return

    languages_to_process = ["English", "Japanese", "Swahili", "Urdu"]
    # languages_to_process = ["English"] # For testing

    for lang in languages_to_process:
        if lang not in LANGUAGE_CONFIGS:
            logging.warning(f"Language '{lang}' not found in LANGUAGE_CONFIGS. Skipping.")
            continue
        
        lang_output_dir = os.path.join(BASE_CONFIG["output_directory"], lang)
        os.makedirs(lang_output_dir, exist_ok=True)
        
        # Set system prompt for the current language on the model instance
        # This is a simplification; if system prompts change frequently, it's better to pass them with each call
        # or re-instantiate. For now, assuming the model_instance can have its system_instruction updated,
        # or that system_instruction is effectively part of the first message in Gemini.
        # The Gemini Python SDK's `GenerativeModel` takes `system_instruction` at init.
        # If we want per-language system prompts, we might need to re-initialize the model
        # or ensure the first message in `prompt_parts` acts as the system prompt.
        # The current `LANGUAGE_CONFIGS` includes `system_prompt` which `process_single_item_gemini` uses
        # when constructing the payload IF the Gemini SDK supports it per-call (it does via messages).
        # The `system_instruction` parameter at `GenerativeModel` init is for a model-level one.
        # The script correctly constructs messages including system prompt, so a single model_instance is fine.

        current_lang_model_instance = genai.GenerativeModel(
            BASE_CONFIG["model_api_name"],
            system_instruction=LANGUAGE_CONFIGS[lang]["system_prompt"], # Set language-specific system prompt
            generation_config=genai.GenerationConfig(
                temperature=BASE_CONFIG["temperature"],
                max_output_tokens=BASE_CONFIG["max_output_tokens"]
            )
        )


        for task_num in range(1, 9): # Tasks 1 through 8
            logging.info("\n" + "="*60)
            logging.info(f"PROCESSING: Language: {lang}, Task: {task_num} (1-shot with Rationales)")
            logging.info("="*60)
            
            # Determine which dataset to use
            # Tasks 1-5 are image-only from `all_images`. Tasks 6-8 are image-text from `image_text_pairs`.
            # The `load_data_items` function returns `image_only_data` and `image_text_data`
            # which are lists of dicts like {'image_path': path} or {'image_path': path, 'text_path': path}

            task_results = asyncio.run(run_task_for_language_gemini(
                lang, task_num, image_only_data, image_text_data, current_lang_model_instance
            ))
            
            output_filename = os.path.join(
                lang_output_dir,
                f"results_{BASE_CONFIG['model_name_for_paths']}_1shot_{LANGUAGE_CONFIGS[lang]['code']}_task{task_num}_Rationales.json"
            )
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(task_results, f, ensure_ascii=False, indent=4)
                logging.info(f"Saved results for {lang} Task {task_num} to {output_filename}")
            except Exception as e_save:
                logging.error(f"Error saving results for {lang} Task {task_num} to {output_filename}: {e_save}")

    logging.info("\nAll specified tasks processed!")

if __name__ == "__main__":
    main()