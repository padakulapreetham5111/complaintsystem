"""Flask web app + JSON API.

Run:  python app.py   ->  http://127.0.0.1:5000
"""
from flask import Flask, jsonify, render_template, request

import train
from src.feedback import feedback_stats, record_feedback
from src.pipeline import ComplaintPipeline

app = Flask(__name__)
pipe = ComplaintPipeline()


@app.get("/")
def index():
    return render_template("index.html", taxonomy=pipe.taxonomy)


@app.post("/api/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(pipe.analyze(data.get("text", "")))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/feedback")
def feedback():
    d = request.get_json(silent=True) or {}
    helpful = bool(d.get("helpful"))
    cat, intent = d.get("true_category"), d.get("true_intent")
    if not helpful:
        if cat not in pipe.taxonomy or intent not in pipe.taxonomy[cat]:
            return jsonify({"error": "Provide a valid true_category and true_intent."}), 400
    record_feedback(d.get("ticket_id", ""), d.get("text", ""), d.get("category"),
                    d.get("intent"), helpful, cat, intent)
    return jsonify({"status": "saved", "stats": feedback_stats()})


@app.get("/api/stats")
def stats():
    return jsonify(feedback_stats())


@app.post("/api/retrain")
def retrain():
    metrics = train.main(use_feedback=True)
    pipe.reload()
    return jsonify({"status": "retrained", "metrics": metrics})


if __name__ == "__main__":
    app.run(debug=True)
