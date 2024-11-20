# LLM Reprocessing
# Model selection
model_by_device = {
    "cuda": {
        "model_id": "bartowski/gemma-2-9b-it-GGUF",
        "model_file": "gemma-2-9b-it-Q8_0.gguf"
    },
    "mps": {
        "model_id": "bartowski/gemma-2-9b-it-GGUF",
        "model_file": "gemma-2-9b-it-Q4_K_M.gguf"
    },
    "cpu": {
        "model_id": "bartowski/gemma-2-2b-it-GGUF",
        "model_file": "gemma-2-2b-it-Q8_0.gguf"
    }
}

set_temperature = {
    "entities": 0.1,
    "sentiment": 0.3,
    "summary": 0.3,
    "translation": 0.2
}

# Prompts
system_prompt = """
You are a highly proficient assistant that strictly follows instructions and provides only the requested output. Do not include interpretations, comments, or acknowledgments unless explicitly asked. Avoid using confirmation phrases such as "Sure, here it comes:", "Got it.", "Here is the translation:", or similar expressions. Responses should be generated without any markdown formatting unless specified otherwise. All outputs must be in {language}.

Instruction:
{instruction}

Output: 
"""

entities_prompt = """
You are an expert in Named Entity Recognition (NER). Your task is to analyze the following transcript and extract all named entities that belong to the specified categories: "Person", "Organization", "Location", "Event", "Date/Time", "Phone Number", "Email Address", "Website", "Weapon".

**Instructions:**
1. **Entity Extraction:** Identify and extract all entities that match the specified categories.
2. **Categorization:** For each extracted entity, assign the correct category from the provided list.
3. **Formatting:** Present the results in a structured JSON format as demonstrated in the example below.
4. **No Additional Information:** Do not include any additional information, explanations, or comments.
5. **No Extraneous Information:** Do not include any Markdown code blocks, additional formatting, or extraneous information. Output only the JSON.

**Example Format:**
[
    {{
        "text": "OpenAI",
        "category": "Organization"
    }},
    {{
        "text": "San Francisco",
        "category": "Location"
    }},
    {{
        "text": "GPT-4",
        "category": "Product"
    }}
]
**Transcript:**
"{text}"
"""

sentiment_prompt = """
You are an expert in sentiment analysis. Your task is to analyze the following text and provide a single sentiment score that represents the overall sentiment.

**Instructions:**
1. **Scoring:** Analyze the text and assign a sentiment score on a scale from -1 to 1:
   - **-1** indicates highly negative sentiment.
   - **0** indicates neutral sentiment.
   - **1** indicates highly positive sentiment.
2. **Output Format:** Return only a single float number between -1 and 1 representing the sentiment score, with no additional text or formatting.

**Text to Analyze:**
"{text}"
"""

summarization_prompt = """
You are an expert summarizer. Create a concise and coherent summary of the following text, capturing all key points and essential information.

**Instructions:**
1. **Content Coverage:** Ensure that the summary includes all main ideas and important details from the original text.
2. **Brevity:** The summary should be concise, ideally between 100 to 200 words unless specified otherwise.
3. **Clarity:** Use clear and straightforward language.
4. **No Additional Information:** Do not include personal opinions, interpretations, or external information.
5. **No Extraneous Information:** Do not include any Markdown code blocks, additional formatting, or extraneous information.

**Text to Summarize:**
"{text}"
"""

translation_prompt = """
You are a professional translator. Translate the following text accurately and fluently into {target_language}.

**Instructions:**
1. **Accuracy:** Ensure that the translation faithfully represents the original text's meaning.
2. **Fluency:** The translated text should read naturally and be grammatically correct in {target_language}.
3. **Preserve Formatting:** Maintain any original formatting, such as bullet points, numbering, or special characters.
4. **Contextual Appropriateness:** Use appropriate terminology and phrasing suitable for the context.

**Text to Translate:**
"{text}"
"""

topic_label_sum_prompt = """
I have a topic that is described by the following keywords: "{keywords}"
The topic contains the following documents: \n"{docs}"
Based on the above information, create a short and descriptive title for the topic.
"""

topic_docs_sum_prompt = """
I have a topic that is described by the following title: "{title}"
The topic is described by the following keywords: "{keywords}"
The topic contains the following documents: \n"{docs}"
Based on the above information, create a summary of the topic.
"""
