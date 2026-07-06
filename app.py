from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    return "Dataset uploaded successfully! Backend integration coming next."

if __name__ == "__main__":
    app.run(debug=True)