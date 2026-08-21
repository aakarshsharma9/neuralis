import json
import logging
import re
import time
import urllib3
import requests

# Suppress SSL warnings for internal self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_URL = "https://aimodels.jadeglobal.com:8082/ollama/api"
MODEL_A = "llama3.1:8b"
MODEL_B = "deepseek-coder:6.7b"
VERIFY_SSL = False
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ==============================================================================
# LOGGING MODULE
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# ==============================================================================
# 1. API CLIENT WITH RETRY LOGIC
# ==============================================================================


class ResilientLLMClient:
    """Handles API requests to company-hosted Ollama servers with exponential retries."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        verify_ssl: bool = False,
        retries: int = MAX_RETRIES,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.verify_ssl = verify_ssl
        self.retries = retries

    def generate_chat_response(
        self, system_prompt: str, user_prompt: str
    ) -> str:
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

        for attempt in range(1, self.retries + 1):
            logging.info(
                f"[{self.model_name}] Attempt {attempt}/{self.retries} sending request..."
            )
            logging.debug(f"[{self.model_name}] System Prompt: {system_prompt}")
            logging.debug(f"[{self.model_name}] User Prompt: {user_prompt}")

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

                logging.info(f"[{self.model_name}] Response received.")
                logging.debug(f"[{self.model_name}] Raw Output: {raw_content}")
                return raw_content

            except requests.exceptions.RequestException as e:
                logging.warning(
                    f"[{self.model_name}] Attempt {attempt} failed: {e}"
                )
                if attempt == self.retries:
                    logging.error(f"[{self.model_name}] Exceeded max retries.")
                    raise RuntimeError(
                        f"API Failure for {self.model_name}: {e}"
                    )
                time.sleep(RETRY_DELAY * attempt)


# ==============================================================================
# 2. PROMPT CONSTRUCTION MODULE
# ==============================================================================


class AdversarialPromptBuilder:
    """Constructs prompts for adversarial reasoning workflow."""

    @staticmethod
    def initial_proposal(scenario: str) -> tuple[str, str]:
        sys = (
            "You are Model A, a strategic problem-solver. Provide a comprehensive, structured proposal "
            "or position to address the user's scenario. Include reasoning, core architecture/steps, and justification."
        )
        usr = f"Scenario / Problem Statement:\n{scenario}\n\nProvide your detailed initial proposal and reasoning."
        return sys, usr

    @staticmethod
    def stress_test_critique(
        scenario: str, initial_proposal: str
    ) -> tuple[str, str]:
        sys = (
            "You are Model B, a rigorous red-team critic. Your job is to stress-test Model A's proposal. "
            "Identify critical weaknesses, operational risks, unhandled edge cases, failure modes, and counterarguments."
        )
        usr = (
            f"Original Scenario:\n{scenario}\n\n"
            f"Model A's Proposed Solution:\n{initial_proposal}\n\n"
            f"Perform a severe stress-test critique of this proposal. Highlight all weaknesses, risks, and edge cases."
        )
        return sys, usr

    @staticmethod
    def revised_defense(
        scenario: str, initial_proposal: str, critique: str
    ) -> tuple[str, str]:
        sys = (
            "You are Model A. Review Model B's critique of your proposal. Defend valid design choices "
            "and revise your proposal to explicitly mitigate identified risks and weaknesses."
        )
        usr = (
            f"Original Scenario:\n{scenario}\n\n"
            f"Your Initial Proposal:\n{initial_proposal}\n\n"
            f"Model B's Critique:\n{critique}\n\n"
            f"Provide your revised proposal, explicitly addressing how you mitigate the risks raised."
        )
        return sys, usr

    @staticmethod
    def final_evaluation(
        scenario: str, initial_p: str, critique: str, revised_p: str
    ) -> tuple[str, str]:
        sys = (
            "You are an objective auditor. Summarize the overall robustness of the final proposal "
            "and list any residual risks that remain."
        )
        usr = (
            f"Scenario: {scenario}\n\n"
            f"Initial Proposal: {initial_p}\n\n"
            f"Critique: {critique}\n\n"
            f"Revised Proposal: {revised_p}\n\n"
            f"Provide a concise final evaluation of solution robustness and remaining risks."
        )
        return sys, usr


# ==============================================================================
# 3. VALIDATION MODULE
# ==============================================================================


class TopicValidator:
    """Validates response content and topic relevance."""

    @staticmethod
    def validate(response_text: str, scenario: str) -> None:
        if not response_text or not response_text.strip():
            raise ValueError("Validation Failed: Empty response received.")

        # Extract meaningful keywords (>3 letters) from scenario to verify relevance
        keywords = [
            w.lower()
            for w in re.findall(r"\b\w+\b", scenario)
            if len(w) > 3
        ]
        if keywords:
            matched = any(kw in response_text.lower() for kw in keywords)
            if not matched:
                logging.warning(
                    f"Topic Relevance Warning: Response might lack alignment with scenario keywords: {keywords[:3]}"
                )


# ==============================================================================
# 4. ORCHESTRATION & JSON FORMATTING MODULE
# ==============================================================================


class AdversarialOrchestrator:
    """Orchestrates 3-turn interaction (A -> B -> A) and returns strictly validated JSON."""

    def __init__(
        self, client_a: ResilientLLMClient, client_b: ResilientLLMClient
    ):
        self.client_a = client_a
        self.client_b = client_b
        self.validator = TopicValidator()

    def process_scenario(self, scenario: str) -> str:
        logging.info("--- Starting Multi-Model Adversarial Reasoning Session ---")

        # Turn 1: Model A Initial Proposal
        sys1, usr1 = AdversarialPromptBuilder.initial_proposal(scenario)
        initial_proposal = self.client_a.generate_chat_response(sys1, usr1)
        self.validator.validate(initial_proposal, scenario)

        # Turn 2: Model B Stress-Test Critique
        sys2, usr2 = AdversarialPromptBuilder.stress_test_critique(
            scenario, initial_proposal
        )
        critique = self.client_b.generate_chat_response(sys2, usr2)
        self.validator.validate(critique, scenario)

        # Turn 3: Model A Revision / Defense
        sys3, usr3 = AdversarialPromptBuilder.revised_defense(
            scenario, initial_proposal, critique
        )
        revised_response = self.client_a.generate_chat_response(sys3, usr3)
        self.validator.validate(revised_response, scenario)

        # Final Synthesis / Evaluation
        sys4, usr4 = AdversarialPromptBuilder.final_evaluation(
            scenario, initial_proposal, critique, revised_response
        )
        final_eval = self.client_b.generate_chat_response(sys4, usr4)

        # Output JSON Payload Structure
        result_payload = {
            "scenario": scenario.strip(),
            "model_a_initial_proposal": initial_proposal.strip(),
            "model_b_critique": critique.strip(),
            "model_a_revised_response": revised_response.strip(),
            "final_evaluation": final_eval.strip(),
        }

        return json.dumps(result_payload, indent=4)


# ==============================================================================
# DRIVER EXECUTION
# ==============================================================================
if __name__ == "__main__":
    client_a = ResilientLLMClient(
        BASE_URL, MODEL_A, verify_ssl=VERIFY_SSL, retries=MAX_RETRIES
    )
    client_b = ResilientLLMClient(
        BASE_URL, MODEL_B, verify_ssl=VERIFY_SSL, retries=MAX_RETRIES
    )

    orchestrator = AdversarialOrchestrator(client_a, client_b)

    sample_scenario = (
        "Our company plans to migrate all core enterprise customer databases "
        "to a fully automated, serverless cloud architecture within 30 days to reduce infrastructure costs."
    )

    try:
        json_output = orchestrator.process_scenario(sample_scenario)

        print("\n" + "=" * 80)
        print("                   FINAL ADVERSARIAL REASONING JSON                      ")
        print("=" * 80)
        print(json_output)

    except Exception as e:
        logging.error(f"Adversarial Reasoning Workflow Failed: {e}")