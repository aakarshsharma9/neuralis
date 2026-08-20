import json
import logging
import re
import urllib3
import requests

# Suppress SSL insecure request warnings for internal endpoints
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================================================================
# CONFIGURATION & LOGGING SETUP
# ==============================================================================
BASE_URL = "https://aimodels.jadeglobal.com:8082/ollama/api"
MODEL_A = "llama3.1:8b"
MODEL_B = "deepseek-coder:6.7b"
VERIFY_SSL = False

# Configure logging to record prompts and raw outputs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# ==============================================================================
# 1. MODULAR API CLIENT
# ==============================================================================


class LLMClient:
    """Handles communications with the company-hosted Ollama API."""

    def __init__(
        self, base_url: str, model_name: str, verify_ssl: bool = False
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.verify_ssl = verify_ssl

    def generate_chat_response(
        self, system_prompt: str, user_prompt: str
    ) -> str:
        """Sends chat prompt payload to Ollama API and returns raw text response."""
        url = f"{self.base_url}/chat"
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        logging.info(f"[{self.model_name}] Sending Request...")
        logging.debug(f"[{self.model_name}] Prompt Payload: {payload}")

        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                verify=self.verify_ssl,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            raw_content = result.get("message", {}).get("content", "")

            logging.info(f"[{self.model_name}] Response Received.")
            logging.debug(f"[{self.model_name}] Raw Output: {raw_content}")

            return raw_content

        except requests.exceptions.RequestException as e:
            logging.error(f"[{self.model_name}] API Call Failed: {e}")
            raise RuntimeError(
                f"API Failure for model '{self.model_name}': {str(e)}"
            )


# ==============================================================================
# 2. PROMPT BUILDER
# ==============================================================================


class PromptBuilder:
    """Constructs prompts for each step in the multi-turn interaction workflow."""

    @staticmethod
    def model_a_initial(topic: str) -> tuple[str, str]:
        sys = (
            "You are a knowledgeable AI debate participant (Model A). "
            "Provide a well-structured, clear, and persuasive position on the topic provided."
        )
        usr = f"Topic: '{topic}'\n\nPlease provide your initial position and core arguments on this topic."
        return sys, usr

    @staticmethod
    def model_b_critique(topic: str, model_a_response: str) -> tuple[str, str]:
        sys = (
            "You are an analytical AI debate participant (Model B). "
            "Critique, question, and expand upon the position provided by Model A."
        )
        usr = (
            f"Topic: '{topic}'\n\n"
            f"Model A's Position:\n{model_a_response}\n\n"
            f"Please critique, question, or expand upon Model A's stance on this topic."
        )
        return sys, usr

    @staticmethod
    def model_a_reply(
        topic: str, model_a_initial: str, model_b_critique: str
    ) -> tuple[str, str]:
        sys = (
            "You are Model A. Review Model B's critique of your initial position. "
            "Provide a thoughtful rebuttal or refined perspective."
        )
        usr = (
            f"Topic: '{topic}'\n\n"
            f"Your Initial Position:\n{model_a_initial}\n\n"
            f"Model B's Critique:\n{model_b_critique}\n\n"
            f"Provide your final reply addressing Model B's points."
        )
        return sys, usr

    @staticmethod
    def synthesize_conclusion(
        topic: str, initial_a: str, critique_b: str, final_a: str
    ) -> tuple[str, str]:
        sys = (
            "You are an impartial discussion synthesizer. Provide a concise, balanced 2-3 sentence summary "
            "synthesizing the key outcomes of the debate."
        )
        usr = (
            f"Topic: '{topic}'\n\n"
            f"Model A Initial Stance: {initial_a}\n\n"
            f"Model B Critique: {critique_b}\n\n"
            f"Model A Final Reply: {final_a}\n\n"
            f"Provide a short synthesized conclusion summarizing the discussion."
        )
        return sys, usr


# ==============================================================================
# 3. RESPONSE VALIDATOR
# ==============================================================================


class ResponseValidator:
    """Validates responses for non-empty content and topic relevance."""

    @staticmethod
    def validate_response(response_text: str, topic: str) -> bool:
        if not response_text or not response_text.strip():
            raise ValueError("Validation Error: Model returned empty response.")

        # Key topic token check for relevance
        topic_words = [
            w.lower()
            for w in re.findall(r"\b\w+\b", topic)
            if len(w) > 3  # Ignore trivial stop words
        ]

        if topic_words:
            has_relevance = any(
                word in response_text.lower() for word in topic_words
            )
            if not has_relevance:
                logging.warning(
                    f"Relevance Warning: Response may not contain core keywords from topic '{topic}'."
                )

        return True


# ==============================================================================
# 4. INTERACTION ORCHESTRATOR
# ==============================================================================


class MultiModelOrchestrator:
    """Orchestrates structured 3-turn discussion (A -> B -> A) and generates final JSON output."""

    def __init__(self, client_a: LLMClient, client_b: LLMClient):
        self.client_a = client_a
        self.client_b = client_b
        self.validator = ResponseValidator()

    def run_discussion(self, topic: str) -> str:
        logging.info(f"--- Starting Multi-Model Interaction on Topic: '{topic}' ---")

        # --- Turn 1: Model A Initial Stance ---
        sys_a1, usr_a1 = PromptBuilder.model_a_initial(topic)
        resp_a1 = self.client_a.generate_chat_response(sys_a1, usr_a1)
        self.validator.validate_response(resp_a1, topic)

        # --- Turn 2: Model B Critique ---
        sys_b1, usr_b1 = PromptBuilder.model_b_critique(topic, resp_a1)
        resp_b1 = self.client_b.generate_chat_response(sys_b1, usr_b1)
        self.validator.validate_response(resp_b1, topic)

        # --- Turn 3: Model A Final Reply ---
        sys_a2, usr_a2 = PromptBuilder.model_a_reply(topic, resp_a1, resp_b1)
        resp_a2 = self.client_a.generate_chat_response(sys_a2, usr_a2)
        self.validator.validate_response(resp_a2, topic)

        # --- Synthesis Step: Final Conclusion ---
        sys_synth, usr_synth = PromptBuilder.synthesize_conclusion(
            topic, resp_a1, resp_b1, resp_a2
        )
        synthesized_conclusion = self.client_a.generate_chat_response(
            sys_synth, usr_synth
        )

        # --- Construct Output JSON Structure ---
        final_output = {
            "topic": topic,
            "model_a_initial": resp_a1.strip(),
            "model_b_critique": resp_b1.strip(),
            "model_a_final_reply": resp_a2.strip(),
            "synthesized_conclusion": synthesized_conclusion.strip(),
        }

        # Return strictly formatted JSON string with no additional surrounding prose
        return json.dumps(final_output, indent=4)


# ==============================================================================
# MAIN DRIVER EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Initialize Clients for Model A and Model B
    client_model_a = LLMClient(
        base_url=BASE_URL, model_name=MODEL_A, verify_ssl=VERIFY_SSL
    )
    client_model_b = LLMClient(
        base_url=BASE_URL, model_name=MODEL_B, verify_ssl=VERIFY_SSL
    )

    orchestrator = MultiModelOrchestrator(client_model_a, client_model_b)

    user_topic = "Should artificial intelligence replace traditional code reviews in software delivery pipelines?"

    try:
        json_result = orchestrator.run_discussion(user_topic)

        # Output result strictly as pure JSON
        print("\n" + "=" * 80)
        print("                         FINAL JSON OUTPUT                              ")
        print("=" * 80)
        print(json_result)

    except Exception as error:
        logging.error(f"Execution Failed: {error}")