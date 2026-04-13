from re import S

from ollama import AsyncClient
from loguru import logger
from collections import deque

from services.models import OllamaResponse


class OllamaClassifier:
    def __init__(
        self, host_url: str, model_name: str, max_context_batch_count: int
    ) -> None:
        self.client = AsyncClient(host=host_url)
        self.model_name = model_name

        self.context_history = deque(maxlen=max_context_batch_count)

    def update_context_history(self, new_text: str) -> None:
        self.context_history.append(new_text)

    async def classify_text(self, text: str) -> OllamaResponse | None:
        messages = [
            {
                "role": "system",
                "content": """
                    # You are a conversation classifier. Your task is to analyze the provided text and determine whether it represents a private conversation or a topic-based conversation.
                    
                    ### PRIVATE CONVERSATION
                    **Personal, sensitive, or confidential exchange of information between specific individuals.**

                    * **Indicators:** Presence of personal names, numbers, addresses.
                    * **Tone:** Emotional (concerns, trust, disputes).
                    * **Characteristics:** Low topic diversity, high personalization.
                    * **Examples:** Family calls, medical consultations.

                    ---

                    ### TOPIC-BASED CONVERSATION
                    **Structured discussion focused on a specific subject or agenda.**

                    * **Indicators:** Clear subject, agenda, or goal of the discussion.
                    * **Tone:** Professional/Formal.
                    * **Characteristics:** Low personalization, high topical cohesion, presence of domain-specific keywords.
                    * **Examples:** Meetings, lectures, tech support.


                    ### RETURN ONLY THIS JSON OUTPUT
                    ### IMPORTANT: You must ALWAYS provide ALL fields in the JSON. Never skip any field even if you are unsure.
                    {
                        "classification": "private" or "topic_based",
                        "confidence": float (0.0 - 1.0),
                        "topics": [list of top 3 identified topics],
                        "dominant_topic": "the most dominant topic",
                        "privacy_signals": [list of identified privacy signals, if classification == "private"],
                        "sentiment": "neutral" or "positive" or "negative",
                        "participants_count": int,
                    }
                    """,
            },
            {
                "role": "user",
                "content": f"History: {" ".join(self.context_history)}\nCurrent message to classify: {text}"
            },
        ]

        response = await self.client.chat(
            model=self.model_name, messages=messages, format="json"
        )
        return OllamaResponse.model_validate_json(response.message.content)
