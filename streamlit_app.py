import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from predictor import predict_vehicle

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "co2_emissions.csv"

st.set_page_config(page_title="Vehicle CO₂ Analysis", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle CO₂ Emissions Analysis")
st.caption("Explore vehicle emissions and estimate CO₂ output using machine learning.")

df = pd.read_csv(DATA)
c1, c2, c3 = st.columns(3)
c1.metric("Vehicle records", f"{len(df):,}")
c2.metric("Vehicle makes", f"{df['Make'].nunique():,}")
c3.metric("Average CO₂ emissions", f"{df['CO2 Emissions(g/km)'].mean():.1f} g/km")

tab1, tab2, tab3 = st.tabs(["CO₂ Predictor", "Dataset Dashboard", "About"])
with tab1:
    st.subheader("Predict CO₂ emissions")
    st.write("Enter vehicle specifications below. The model estimates emissions in grams per kilometre.")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            make = st.selectbox("Vehicle make", sorted(df["Make"].dropna().unique()))
            model_options = sorted(df.loc[df["Make"] == make, "Model"].dropna().unique())
            model = st.selectbox("Vehicle model", model_options)
            vehicle_class = st.selectbox("Vehicle class", sorted(df["Vehicle Class"].dropna().unique()))
            engine = st.number_input("Engine size (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
            cylinders = st.selectbox("Cylinders", sorted(int(x) for x in df["Cylinders"].dropna().unique()))
            transmission = st.selectbox("Transmission", sorted(df["Transmission"].dropna().unique()))
        with right:
            fuel = st.selectbox("Fuel type", sorted(df["Fuel Type"].dropna().unique()))
            city = st.number_input("City fuel consumption (L/100 km)", min_value=1.0, max_value=40.0, value=9.0, step=0.1)
            highway = st.number_input("Highway fuel consumption (L/100 km)", min_value=1.0, max_value=35.0, value=7.0, step=0.1)
            combined = st.number_input("Combined fuel consumption (L/100 km)", min_value=1.0, max_value=40.0, value=8.1, step=0.1)
            mpg = st.number_input("Combined fuel economy (mpg)", min_value=1, max_value=100, value=35)
        submitted = st.form_submit_button("Predict CO₂ emissions", type="primary")
    if submitted:
        values = {
            "Make": make, "Model": model, "Vehicle Class": vehicle_class,
            "Engine Size(L)": engine, "Cylinders": cylinders,
            "Transmission": transmission, "Fuel Type": fuel,
            "Fuel Consumption City (L/100 km)": city,
            "Fuel Consumption Hwy (L/100 km)": highway,
            "Fuel Consumption Comb (L/100 km)": combined,
            "Fuel Consumption Comb (mpg)": mpg
        }
        try:
            prediction, tier = predict_vehicle(values)
            st.success("Prediction completed")
            a, b = st.columns(2)
            a.metric("Estimated CO₂ emissions", f"{prediction:.1f} g/km")
            b.metric("Predicted emission category", tier)
            st.caption("This is a model estimate based on the supplied dataset, not an official vehicle certification result.")
        except Exception as e:
            st.error(str(e))

with tab2:
    st.subheader("Dataset overview")
    st.dataframe(df.head(20), use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.write("**Average CO₂ emissions by vehicle class (top 10)**")
        chart = df.groupby("Vehicle Class")["CO2 Emissions(g/km)"].mean().sort_values(ascending=False).head(10)
        st.bar_chart(chart)
    with b:
        st.write("**Average CO₂ emissions by fuel type**")
        chart2 = df.groupby("Fuel Type")["CO2 Emissions(g/km)"].mean().sort_values(ascending=False)
        st.bar_chart(chart2)
    st.write("**Fuel consumption and CO₂ emissions**")
    fig, ax = plt.subplots()
    ax.scatter(df["Fuel Consumption Comb (L/100 km)"], df["CO2 Emissions(g/km)"], alpha=0.25)
    ax.set_xlabel("Combined fuel consumption (L/100 km)")
    ax.set_ylabel("CO₂ emissions (g/km)")
    ax.set_title("Fuel consumption vs CO₂ emissions")
    st.pyplot(fig)

with tab3:
    st.subheader("About this project")
    st.markdown("""
    This project analyzes vehicle CO₂ emissions using the supplied Canada vehicle dataset.
    It includes a Random Forest regression model for numerical prediction and a
    Random Forest classifier for Low, Medium and High emission categories.

    **Category boundaries used:** Low ≤ 220 g/km; Medium 221–273 g/km; High ≥ 274 g/km.

    **Limitations:** Predictions depend on the dataset and entered specifications.
    Correlation does not prove causation, and predictions are not official emissions tests.
    """)
