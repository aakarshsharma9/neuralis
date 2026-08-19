import json
import re
import urllib3
import requests

# Suppress SSL insecure request warnings (since verify=False is used for company endpoint)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# Configuration (Company LLM)
# ==========================================
LLAMA_BASE_URL = "https://aimodels.jadeglobal.com:8082/ollama/api"
LLAMA_MODEL = "llama3.1:8b"  # or "deepseek-coder:6.7b"
LLAMA_VERIFY_SSL = False  # Set to True if SSL cert is valid


# ==========================================
# 1. API Call Function
# ==========================================
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Sends prompt payload to the company Ollama API chat endpoint."""
    url = f"{LLAMA_BASE_URL}/chat"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",  # Instructs Ollama to enforce valid JSON output
        "stream": False,
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            verify=LLAMA_VERIFY_SSL,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        # Extract assistant response content
        return result["message"]["content"]

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API Failure: Unable to reach LLM API. {str(e)}")


# ==========================================
# 2. Article Analysis System
# ==========================================
def analyze_article(article_text: str) -> dict:
    """Constructs prompt, sends article to LLM, cleans response,

    and validates constraints.
    """

    system_prompt = (
        "You are an expert article analyst. Your task is to analyze the provided article "
        "and return a structured analysis strictly as a raw JSON object.\n\n"
        "STRICT CONSTRAINTS:\n"
        "1. Output MUST be valid JSON only. Do NOT include markdown code blocks (e.g., ```json), "
        "preamble, or postscript text.\n"
        "2. Required JSON Fields:\n"
        '   - "summary": string (concise summary, strictly under 150 words)\n'
        '   - "important_points": array of strings (must contain between 5 and 10 clearly written points)\n'
        '   - "key_themes": array of strings (must contain between 3 and 5 short phrases, not full sentences)\n'
        '   - "target_audience": string (brief identification of the relevant target audience)\n'
    )

    user_prompt = f"Analyze the following article:\n\n---\n{article_text}\n---"

    # Step A: Query LLM
    raw_response = call_llm(system_prompt, user_prompt)

    # Step B: Clean response (Extract JSON if surrounded by markdown or commentary)
    cleaned_response = raw_response.strip()

    # Regex to extract JSON block if wrapped in markdown ```json ... ```
    json_match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
    if json_match:
        cleaned_response = json_match.group(0)

    # Step C: Parse JSON
    try:
        data = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Malformed JSON response received from LLM: {str(e)}\nRaw Response: {raw_response}"
        )

    # Step D: Validate Fields & Constraints
    required_keys = [
        "summary",
        "important_points",
        "key_themes",
        "target_audience",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required field in JSON response: '{key}'")

    summary = str(data["summary"])
    important_points = data["important_points"]
    key_themes = data["key_themes"]
    target_audience = str(data["target_audience"])

    # Enforce Word Count Constraint on Summary (< 150 words)
    summary_word_count = len(summary.split())
    if summary_word_count > 150:
        # Truncate gracefully if constraint violated slightly
        words = summary.split()[:150]
        data["summary"] = " ".join(words) + "..."
        print(
            f"[Warning] Summary exceeded 150 words ({summary_word_count} words). Automatically truncated."
        )

    # Enforce Constraints on Important Points (5 to 10 points)
    if not isinstance(important_points, list):
        raise TypeError("'important_points' must be an array of strings.")
    if not (5 <= len(important_points) <= 10):
        print(
            f"[Warning] Constraint Violation: Expected 5-10 important points, got {len(important_points)}."
        )

    # Enforce Constraints on Key Themes (3 to 5 themes)
    if not isinstance(key_themes, list):
        raise TypeError("'key_themes' must be an array of strings.")
    if not (3 <= len(key_themes) <= 5):
        print(
            f"[Warning] Constraint Violation: Expected 3-5 key themes, got {len(key_themes)}."
        )

    return data


# ==========================================
# 3. Execution Example
# ==========================================
if __name__ == "__main__":
    sample_article = """
    Artificial Intelligence is transforming renewable energy management by predicting energy demand, 
    optimizing grid performance, and integrating solar and wind sources efficiently. As power grids 
    transition from centralized fossil-fuel systems to distributed renewable networks, unpredictability 
    becomes a primary challenge. Solar power varies with cloud coverage, and wind energy fluctuates with 
    weather shifts. AI models leverage historical weather data, real-time satellite imagery, and sensor 
    telemetry to forecast output hours and days in advance. 

    Furthermore, smart grids powered by machine learning algorithms automatically adjust energy distribution 
    in response to real-time consumer usage, preventing blackouts and reducing operational costs for utility 
    providers. Energy storage systems, such as large-scale lithium-ion batteries, also rely on AI to determine 
    optimal charge and discharge cycles, maximizing battery lifespan and profitability. While initial 
    capital investments in AI infrastructure remain significant, energy executives estimate a 20-30% 
    reduction in operational inefficiencies over the next decade.
    """

    print("Analyzing Article...")
    try:
        structured_analysis = analyze_article(sample_article)

        print("\n" + "=" * 60)
        print("          STRUCTURED ARTICLE ANALYSIS RESULT           ")
        print("=" * 60)
        print(json.dumps(structured_analysis, indent=4))

    except Exception as err:
        print(f"\n[ERROR] System Error: {err}")