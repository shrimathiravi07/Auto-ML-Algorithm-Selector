"""Flask application for the SmartML Model Selector dashboard.

This module is responsible for:
- creating the Flask application instance
- defining a route that renders the dashboard page
- preparing dummy data for the current phase
- passing the generated chart filename into the HTML template
"""

from __future__ import annotations

from flask import Flask, render_template

from visualization import generate_accuracy_chart

app = Flask(__name__)


def get_dashboard_context() -> dict:
    """Prepare the dashboard data for the current phase.

    The backend team can later replace the dummy values with real model data
    while keeping the route and template logic unchanged.
    """
    algorithms = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]
    accuracies = [88.5, 91.2, 94.0, 86.7]
    chart_filename = "model_accuracy.png"

    generate_accuracy_chart(
        algorithms=algorithms,
        accuracies=accuracies,
        output_filename=chart_filename,
    )

    comparison_rows = [
        {"algorithm": name, "accuracy": score}
        for name, score in zip(algorithms, accuracies)
    ]

    return {
        "title": "SmartML Model Selector Dashboard",
        "chart_filename": chart_filename,
        "comparison_rows": comparison_rows,
        "recommended_model": "Random Forest",
        "recommended_accuracy": 94.0,
    }


@app.route("/")
@app.route("/dashboard")
def dashboard() -> str:
    """Render the dashboard page with dummy data and the chart filename."""
    context = get_dashboard_context()
    return render_template("dashboard.html", **context)


if __name__ == "__main__":
    app.run(debug=True)
