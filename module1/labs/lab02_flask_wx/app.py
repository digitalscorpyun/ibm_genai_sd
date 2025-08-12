import os
from flask import Flask, request, jsonify
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models.inference import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

app = Flask(__name__)

# --- Config via environment variables ---
WATSONX_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY = os.environ["WATSONX_API_KEY"]  # required
WATSONX_PROJECT_ID = os.environ["WATSONX_PROJECT_ID"]  # required (Studio Project ID)
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/llama-2-70b-chat")  # default

# Default generation params (tuned for “precision mode” feel)
DEFAULT_PARAMS = {
    GenParams.DECODING_METHOD: "sample",  # or "greedy"
    GenParams.TEMPERATURE: 0.3,
    GenParams.TOP_P: 0.9,
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.RETURN_OPTIONS: {
        "input_text": True,
        "generated_tokens": True,
        "input_tokens": True
    }
}

# --- watsonx.ai client ---
credentials = Credentials(
    url=WATSONX_URL,
    api_key=WATSONX_API_KEY,
)

model = ModelInference(
    model_id=MODEL_ID,
    credentials=credentials,
    params=DEFAULT_PARAMS,
    project_id=WATSONX_PROJECT_ID,
)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON body:
      {
        "prompt": "string",                # required
        "params": { ... }                  # optional, overrides DEFAULT_PARAMS keys
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    prompt = data.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return jsonify({"error": "Missing or invalid 'prompt' (string required)."}), 400

    # Allow per-request overrides of generation params
    user_params = data.get("params") or {}
    params = {**DEFAULT_PARAMS, **user_params}

    try:
        # generate_text() returns just the text; generate() returns full dict.
        text = model.generate_text(prompt=prompt, params=params)
        return jsonify({
            "model_id": MODEL_ID,
            "prompt": prompt,
            "generated_text": text
        }), 200
    except Exception as e:
        # Keep errors surfaced for quick triage in Code Engine logs
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Code Engine sets PORT; default to 8080
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
