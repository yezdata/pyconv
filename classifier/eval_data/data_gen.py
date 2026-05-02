from openai import OpenAI
import json
import uuid
import os

SYSTEM_PROMPTS: dict[str, str] = {
    "private": (
        "You are an expert data generator for text classification. "
        "Generate a conversation for the label 'PRIVATE_CONVERSATION'.\n\n"
        "STRICT REQUIREMENTS:\n"
        "1. High Personality: Include specific names , relative references (e.g., my sister, our doctor), and personal dates/addresses.\n"
        "2. Emotional Tone: The dialogue must convey personal feelings, vulnerabilities, trust, or conflict (e.g., 'I'm worried about...', 'I trust you with this...').\n"
        "3. High Specificity: Focus on the individual impact of the topic rather than general facts.\n"
        "4. Informal Style: Use natural, conversational language with frequent use of first and second-person pronouns (I, you, my, your)."
        
    ),
    "topic_based": (
        "You are an expert data generator for text classification. "
        "Generate a conversation for the label 'TOPIC_BASED_CONVERSATION'.\n\n"
        "STRICT REQUIREMENTS:\n"
        "1. High Topic Cohesion: The discussion must follow a clear agenda, goal, or technical objective.\n"
        "2. Low Personality: Avoid personal names, emotional outbursts, or private life details. Speakers act as professional entities or roles.\n"
        "3. Domain Keywords: Use industry-specific terminology, technical parameters, and professional jargon.\n"
        "4. Objective Tone: The language must be professional, factual, and neutral, focusing on solving a task or exchanging knowledge."
    ),
}

# PRIVATE_TOPICS = [
#     "negotiating a debt settlement with a family member",
#     "discussing sensitive medical test results with a spouse",
#     "confessing a personal mistake to a close friend",
#     "planning a private legal strategy for a divorce"
# ]

# TOPIC_BASED_TOPICS = [
#     "troubleshooting a Kubernetes pod scheduling error",
#     "reviewing Q3 EBITDA and financial compliance requirements",
#     "discussing API integration documentation for a payment gateway",
#     "analyzing labor law amendments regarding remote work contracts"
# ]

        

def generate_data(label):
    client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    system_prompt = SYSTEM_PROMPTS[label]

    response = client.chat.completions.create(
        model="qwen/qwen3.5-9b",
        messages=[
            {"role": "system", "content": f"/no_think Write only raw text, don't use any tags or markdown. Write only text of the conversation and dont identify or write name of the speakers. Dont write just monologue, but real conversation. Realistic turn length: Mix short reactions ('Really?', 'Oh god.') with longer explanations. Not every turn should be a paragraph.Single topic focus: Pick ONE specific problem per conversation. {system_prompt} "},
            {"role": "user", "content": f"Generate a detailed {label} conversation."}
        ],
        temperature=0.8,
    )

    conversation = response.choices[0].message.content
    conversation = " ".join(conversation.strip().split())

    return conversation


def write_to_file(data, filename="generated_data.json"):
    if os.path.exists(filename):

        with open(filename, "r", encoding="utf-8") as fr:
            file_data = json.load(fr)

        file_data.extend(data)
    else:
        file_data = data

    with open(filename, "w", encoding="utf-8") as fw:
        json.dump(file_data, fw, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    for i in range(5000):
        print(f"Private conv: {i+1}")
        conversation = generate_data("private")
        print(conversation)
        data_entry = [{
            "id": str(uuid.uuid4()),
            "label": "PRIVATE_CONVERSATION",
            "conversation": conversation
        }]
        write_to_file(data_entry, "private_set.json")
        
        print(f"Topic-based conv: {i+1}")
        conversation = generate_data("topic_based")
        data_entry = [{
            "id": str(uuid.uuid4()),
            "label": "TOPIC_BASED_CONVERSATION",
            "conversation": conversation
        }]
        write_to_file(data_entry, "topic_based_set.json")