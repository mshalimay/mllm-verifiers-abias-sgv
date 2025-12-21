import argparse

from flask import Flask, render_template

app = Flask(__name__)
parser = argparse.ArgumentParser(description="Run Flask app with custom port")
parser.add_argument("process_id", type=int, help="Instance ID of the homepage")
parser.add_argument("--port", type=int, default=4399, help="Port to run the Flask app on")
args = parser.parse_args()


@app.route("/process_id")
def process_id() -> str:
    return str(args.process_id)


@app.route("/")
def index() -> str:
    template_file_name = f"index-{args.process_id}.html"
    return render_template(template_file_name)


@app.route("/scratchpad.html")
def scratchpad() -> str:
    return render_template("scratchpad.html")


@app.route("/calculator.html")
def calculator() -> str:
    return render_template("calculator.html")


@app.route("/password.html")
def password() -> str:
    return render_template("password.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port)
