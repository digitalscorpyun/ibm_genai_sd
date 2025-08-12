import os
from flask import Flask, request, jsonify
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models.inference import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

app = Flask(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/llama-2-70b-chat")
WATSONX_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

DEFAULT_PARAMS = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.TEMPERATURE: 0.3,
    GenParams.TOP_P: 0.9,
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.RETURN_OPTIONS: {"input_text": True, "generated_tokens": True, "input_tokens": True},
}

_model = None
def get_model():
    global _model
    if _model is not None:
        return _model
    api_key = os.environ.get("WATSONX_API_KEY")
    project_id = os.environ.get("WATSONX_PROJECT_ID")
    if not api_key or not project_id:
        raise RuntimeError("Missing WATSONX_API_KEY or WATSONX_PROJECT_ID")
    creds = Credentials(url=WATSONX_URL, api_key=api_key)
    _model = ModelInference(
        model_id=MODEL_ID,
        credentials=creds,
        params=DEFAULT_PARAMS,
        project_id=project_id,
    )
    return _model

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return jsonify({"error": "Missing or invalid 'prompt' (string required)."}), 400
    params = {**DEFAULT_PARAMS, **(data.get("params") or {})}
    try:
        model = get_model()
    except Exception as e:
        return jsonify({"error": f"Model init failed: {str(e)}"}), 503
    try:
        text = model.generate_text(prompt=prompt, params=params)
        return jsonify({"model_id": MODEL_ID, "generated_text": text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 502

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
