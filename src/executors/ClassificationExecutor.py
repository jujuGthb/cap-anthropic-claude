"""
Anthropic Claude Executor: Sends data to Anthropic's API and returns AI analysis.
"""

import os
import sys
import base64
import json
import requests
import anthropic
from anthropic import NOT_GIVEN
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.AnthropicClaude.src.utils.response import build_response_classification
from capsules.AnthropicClaude.src.models.PackageModel import PackageModel

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class ClassificationExecutor(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.classes = self.request.get_param("inputClasses")
        self.api_provider = self.request.get_param("apiProvider")
        self.api_key = self.request.get_param("inputApiKey")
        print(f"[DEBUG] Full api_key received: '{self.api_key}'")
        print(f"[DEBUG] api_key length: {len(self.api_key) if self.api_key else 0}")
        self.model_version = self.request.get_param("inputModelVersion")
        self.extended_thinking = self.request.get_param("extendedThinking")
        self.thinking_budget_tokens = self.request.get_param("thinkingBudgetTokens")
        self.temperature = self.request.get_param("inputTemperature")
        self.max_tokens = self.request.get_param("maxTokens")
        self.max_concurrent_requests = self.request.get_param("maxConcurrentRequests")
        self.image_selector = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _build_payload(self, base64_image):
        serialised_classes = ", ".join(self.classes) if isinstance(self.classes, list) else (self.classes or "")
        payload = {
            "model": self.model_version,
            "max_tokens": self.max_tokens,
            "system": (
                'You act as a single-class classification model. You must provide reasonable predictions. '
                'You are only allowed to produce a JSON document. '
                'Expected structure: {"class_name": "class-name", "confidence": 0.9}. '
                '`class-name` must be one of the classes defined by the user. '
                'Return a single JSON object only, no list.'
            ),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": f"List of classes: {serialised_classes}"},
                    ],
                }
            ],
        }

        if self.extended_thinking:
            budget = self.thinking_budget_tokens if self.thinking_budget_tokens and self.thinking_budget_tokens >= 1024 else 1024
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        elif self.temperature is not None:
            payload["temperature"] = self.temperature

        return payload

    def run(self):
        img = Image.get_frame(img=self.image_selector, redis_db=self.redis_db)

        success, encoded_image = cv2.imencode('.jpg', img.value)
        if not success:
            raise RuntimeError("Failed to encode image for API")

        base64_image = base64.b64encode(encoded_image).decode('utf-8')
        payload = self._build_payload(base64_image)

        try:
            if self.api_provider == "NovaVision":
                url = f"{self.environment.web_api}/apiproxy/anthropic?access-token={self.api_key}"
                response = requests.post(url, json=payload)

                print(f"[DEBUG] Status: {response.status_code}")
                print(f"[DEBUG] Response: {response.text[:500]}")

                response.raise_for_status()
                data = response.json()

                stop_reason = data.get("stop_reason")
                if stop_reason == "max_tokens":
                    raise ValueError(
                        "Claude API stopped generation because the max_tokens limit was reached. "
                        "Please increase the maxTokens parameter to allow for a complete response."
                    )
                if stop_reason not in ["end_turn", "stop_sequence", None]:
                    raise ValueError(
                        f"Claude API stopped generation with unexpected stop reason: {stop_reason}."
                    )

                content = data.get("content", [])
                if not content:
                    raise ValueError("Claude API returned no content in response.")
                self.claude_text = next(
                    (block["text"] for block in content if block.get("type") == "text"), ""
                )
                if not self.claude_text:
                    raise ValueError("Claude API returned no text content in response.")

            else:
                client = anthropic.Anthropic(api_key=self.api_key)
                temperature = NOT_GIVEN if (self.temperature is None or self.extended_thinking) else self.temperature
                system = payload.get("system", NOT_GIVEN)

                result = client.messages.create(
                    model=payload["model"],
                    max_tokens=payload["max_tokens"],
                    system=system,
                    messages=payload["messages"],
                    temperature=temperature,
                )

                stop_reason = result.stop_reason
                if stop_reason == "max_tokens":
                    raise ValueError(
                        "Claude API stopped generation because the max_tokens limit was reached. "
                        "Please increase the maxTokens parameter to allow for a complete response."
                    )
                if stop_reason not in ["end_turn", "stop_sequence"]:
                    raise ValueError(
                        f"Claude API stopped generation with unexpected stop reason: {stop_reason}."
                    )

                self.claude_text = next(
                    (block.text for block in result.content if block.type == "text"), ""
                )
                if not self.claude_text:
                    raise ValueError("Claude API returned no text content in response.")

            try:
                parsed = json.loads(self.claude_text)
                class_name = parsed.get("class_name", "")
                self.claude_classes = [class_name] if class_name else []
            except (json.JSONDecodeError, KeyError):
                self.claude_classes = []

        except requests.exceptions.HTTPError as e:
            self.claude_text = f"HTTP Error {response.status_code}: {response.text}"
            self.claude_classes = []
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.claude_text = f"API Error: {str(e)}"
            self.claude_classes = []

        return build_response_classification(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()