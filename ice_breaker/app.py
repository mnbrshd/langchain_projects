from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from ice_breaker import ice_break_with

load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    company = request.form["company"]
    summary, profile_pic_url = ice_break_with(name=name, company=company)
    return jsonify(
        {
            "summary_and_facts": summary.to_dict(),
            "picture_url": profile_pic_url,
            "ice_breakers": "",
            "topics-of-interest": ""
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)