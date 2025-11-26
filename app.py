import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# === CONFIG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # This will help you debug on Render if the env var isn't set
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
# Allow your website to call this API (you can restrict origins later)
CORS(app, resources={r"/api/*": {"origins": "*"}})


SYSTEM_PROMPT = """
You are a property development feasibility assistant for NSW, Australia.
You help pre-fill assumptions for a feasibility model.

Constraints:
- You CANNOT live-browse the web. Use only your training and general knowledge.
- You may suggest likely official sources/URLs (e.g. NSW Planning Portal, council websites),
  but you must clearly state that the user must verify them.

Focus areas:
- Section 7.11 and 7.12 development contributions (explain which is likely to apply).
- Housing and Productivity Contribution (HPC) where relevant.
- Long Service Levy.
- BASIX fees (rough magnitude and where to check).
- Typical construction cost per m² for this type of product.
- Typical marketing % of gross realisation.
- Other common state/local contributions (SIC, state infrastructure charges, bonds, etc.)

Output:
Return ONLY valid JSON in this structure:

{
  "assumptions": {
    "section_711_per_dwelling": {
      "value": <number or null>,
      "unit": "AUD per dwelling",
      "notes": "string",
      "source_links": ["https://...", "..."]
    },
    "section_712_percent": {
      "value": <number or null>,
      "unit": "% of estimated development cost",
      "notes": "string",
      "source_links": ["https://..."]
    },
    "hpc_per_dwelling": {
      "value": <number or null>,
      "unit": "AUD per dwelling",
      "notes": "string",
      "source_links": ["https://..."]
    },
    "construction_cost_per_m2": {
      "value": <number or null>,
      "unit": "AUD per m2",
      "notes": "string",
      "source_links": ["https://..."]
    },
    "marketing_percent_of_gross": {
      "value": <number or null>,
      "unit": "% of gross realisation",
      "notes": "string",
      "source_links": ["https://..."]
    },
    "long_service_levy_percent": {
      "value": <number or null>,
      "unit": "% of construction cost",
      "notes": "string",
      "source_links": ["https://..."]
    },
    "basix_fee_estimate": {
      "value": <number or null>,
      "unit": "AUD per project or dwelling",
      "notes": "string",
      "source_links": ["https://..."]
    },
    "other_state_charges_estimate": {
      "value": <number or null>,
      "unit": "AUD total",
      "notes": "string",
      "source_links": ["https://..."]
    }
  },
  "computed": {
    "estimated_total_construction_cost": <number or null>,
    "estimated_total_contributions": <number or null>,
    "notes": "string"
  }
}
"""


def call_chatgpt_prefill(project_data: dict) -> dict:
    """
    Call OpenAI to prefill feasibility assumptions for an NSW project.
    """

    user_prompt = f"""
Project details:
- Address: {project_data.get('address')}
- LGA: {project_data.get('lga')}
- Project type: {project_data.get('project_type')}
- Number of dwellings: {project_data.get('dwellings')}
- Gross floor area (m2): {project_data.get('gfa_m2')}
- Brief description: {project_data.get('description')}

Task:
Based on NSW practice, estimate:
- Section 7.11 OR 7.12 contributions relevant to this project (and explain which is more likely).
- HPC per dwelling if applicable, or note if unlikely.
- Long Service Levy %.
- Typical construction cost per m² for this product type in NSW.
- Typical marketing % of gross realisation.
- Rough BASIX fee magnitude.
- Any other common state or local charges.

Then compute:
- estimated_total_construction_cost = construction_cost_per_m2 * gfa_m2 (if both known)
- estimated_total_contributions = contributions + HPC + other charges (approx).

Return data in the exact JSON format described in the system prompt.
If you are unsure, set value to null and explain in notes.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    content = completion.choices[0].message.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"error": "Model did not return valid JSON", "raw": content}

    return data


@app.route("/api/feasibility/prefill", methods=["POST"])
def feasibility_prefill():
    """
    API endpoint: POST /api/feasibility/prefill
    Expects JSON with at least: address, lga, project_type
    Optional: dwellings, gfa_m2, description
    """
    project_data = request.json or {}

    required = ["address", "lga", "project_type"]
    missing = [k for k in required if not project_data.get(k)]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    ai_result = call_chatgpt_prefill(project_data)

    return jsonify({
        "project": project_data,
        "ai_result": ai_result
    })


@app.route("/")
def root():
    """
    Simple health check endpoint.
    """
    return jsonify({"status": "ok", "message": "NSW feasibility API is running"})


if __name__ == "__main__":
    # For local testing; Render will use gunicorn with app:app
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
