from flask import Flask, render_template, request
import pandas as pd
from pathlib import Path
from predictor import predict_vehicle

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "co2_emissions.csv")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    if request.method == "POST":
        try:
            values = {
                "Make": request.form["make"],
                "Model": request.form["model"],
                "Vehicle Class": request.form["vehicle_class"],
                "Engine Size(L)": float(request.form["engine"]),
                "Cylinders": int(request.form["cylinders"]),
                "Transmission": request.form["transmission"],
                "Fuel Type": request.form["fuel"],
                "Fuel Consumption City (L/100 km)": float(request.form["city"]),
                "Fuel Consumption Hwy (L/100 km)": float(request.form["highway"]),
                "Fuel Consumption Comb (L/100 km)": float(request.form["combined"]),
                "Fuel Consumption Comb (mpg)": int(request.form["mpg"])
            }
            prediction, tier = predict_vehicle(values)
            result = {"prediction": round(prediction, 1), "tier": tier}
        except Exception as exc:
            error = str(exc)
    return render_template(
        "index.html", makes=sorted(df["Make"].dropna().unique()),
        models=sorted(df["Model"].dropna().unique()),
        classes=sorted(df["Vehicle Class"].dropna().unique()),
        transmissions=sorted(df["Transmission"].dropna().unique()),
        fuels=sorted(df["Fuel Type"].dropna().unique()),
        result=result, error=error
    )

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
