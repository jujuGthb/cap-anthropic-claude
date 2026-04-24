"""
Anthropic Claude Executor: Sends data to Anthropic's API and returns AI analysis.
"""

import os
import sys
import requests
import anthropic
from anthropic import NOT_GIVEN

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.AnthropicClaude.src.utils.response import build_response_text_prompt
from capsules.AnthropicClaude.src.models.PackageModel import PackageModel

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class TextPromptExecutor(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.prompt = self.request.get_param("inputPrompt")
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

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _build_payload(self):
        payload = {
            "model": self.model_version,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "user", "content": self.prompt}
            ],
        }

        if self.extended_thinking:
            budget = self.thinking_budget_tokens if self.thinking_budget_tokens and self.thinking_budget_tokens >= 1024 else 1024
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        elif self.temperature is not None:
            payload["temperature"] = self.temperature

        return payload

    def run(self):
        payload = self._build_payload()

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

            self.claude_classes = []

        except requests.exceptions.HTTPError as e:
            self.claude_text = f"HTTP Error {response.status_code}: {response.text}"
            self.claude_classes = []
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.claude_text = f"API Error: {str(e)}"
            self.claude_classes = []

        return build_response_text_prompt(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()