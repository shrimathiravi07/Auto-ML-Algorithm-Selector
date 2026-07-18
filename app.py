<<<<<<< HEAD
from flask import Flask, render_template, request, send_file
import os
import pandas as pd

from src.preprocessing import preprocess_data
from src.problem_detector import detect_problem_type
from src.model_trainer import train_models
from src.evaluator import evaluate_models
from src.model_selector import select_best_model
from src.model_saver import save_model
from src.graph_generator import generate_graph

app = Flask(__name__)

# =====================================
# Dataset Folder Configuration
# =====================================

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "Dataset"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Store uploaded dataset path
current_dataset_path = None


# =====================================
# Home Page
# =====================================

@app.route("/")
def home():
    return render_template(
        "index.html"
    )


# =====================================
# Upload Dataset
# =====================================

@app.route("/upload", methods=["POST"])
def upload_file():

    global current_dataset_path

    # Check file selected
    if "dataset" not in request.files:
        return "No file selected."

    file = request.files["dataset"]

    # Check filename
    if file.filename == "":
        return "Please select a CSV file."

    # Save file
    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(file_path)

    current_dataset_path = file_path

    # Read dataset
    df = pd.read_csv(file_path)

    rows = df.shape[0]
    cols = df.shape[1]
    missing_values = df.isnull().sum().sum()

    columns = df.columns.tolist()

    return render_template(
        "dashboard.html",
        rows=rows,
        cols=cols,
        missing_values=missing_values,
        columns=columns
    )


# =====================================
# Train Models
# =====================================

@app.route("/train", methods=["POST"])
def train():

    global current_dataset_path

    if current_dataset_path is None:
        return "Please upload a dataset first."

    # Selected target column
    target_column = request.form["target_column"]

    # Invalid target columns
    invalid_columns = [
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]

    # Validation
    if target_column in invalid_columns:
        return f"""
        <h2>Invalid Target Column Selected</h2>

        <p>
        '{target_column}' cannot be used as a target column.
        Please choose:
        Survived, Fare or Pclass.
        </p>
        """

    # Read dataset
    df = pd.read_csv(
        current_dataset_path
    )

    # Check target exists
    if target_column not in df.columns:
        return f"""
        <h2>Error</h2>

        <p>
        Target column '{target_column}'
        does not exist in dataset.
        </p>
        """

    # Preprocess data
    df, label_encoders = preprocess_data(
        df
    )

    # Detect problem type
    problem_type = detect_problem_type(
        df,
        target_column
    )

    # Train models
    trained_models, X_test, y_test = train_models(
        df,
        target_column,
        problem_type
    )

    # Evaluate models
    scores = evaluate_models(
        trained_models,
        X_test,
        y_test,
        problem_type
    )

    # Select best model
    best_model_name, best_score = select_best_model(
        scores
    )

    # Save best model
    saved_model_path = save_model(
        trained_models[best_model_name],
        best_model_name
    )

    # Generate graph
    graph_path = generate_graph(
        scores
    )

    return render_template(
        "result.html",
        target_column=target_column,
        problem_type=problem_type,
        scores=scores,
        best_model=best_model_name,
        best_score=round(best_score, 4),
        saved_model_path=saved_model_path,
        graph_path=graph_path
    )


# =====================================
# Download Best Model
# =====================================

@app.route("/download-model")
def download_model():

    model_path = request.args.get(
        "path"
    )

    if (
        model_path and
        os.path.exists(model_path)
    ):
        return send_file(
            model_path,
            as_attachment=True
        )

    return "Model file not found."


# =====================================
# Run Flask Application
# =====================================

if __name__ == "__main__":
    app.run(
        debug=True
    )
=======
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
>>>>>>> 6b3085b5a9ea23bc02c6846f3dc92283c91e9aad

