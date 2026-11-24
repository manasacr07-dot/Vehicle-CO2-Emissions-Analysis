# Vehicle CO2 Emissions Analysis

A student data-science project using the supplied Canada vehicle CO2 emissions dataset.

## Included applications
- **Streamlit dashboard**: prediction, dataset overview, charts and project information.
- **Flask website**: responsive web page with a vehicle prediction form and About page.
- **Machine learning**: Random Forest regression for numerical CO2 prediction and Random Forest classification for Low/Medium/High tiers.

## Dataset
Place the supplied CSV at `data/co2_emissions.csv`. The dataset has 7,385 rows and 12 columns.

## Setup (Mac)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
```

## Run Streamlit
```bash
streamlit run streamlit_app.py
```
Open the local URL shown in the terminal (usually http://localhost:8501).

## Run Flask website
Open a second terminal, activate the environment, then:
```bash
python app.py
```
Open http://127.0.0.1:5000.

## Emission category boundaries
- Low: <= 220 g/km
- Medium: 221–273 g/km
- High: >= 274 g/km

## Notes and limitations
The models are trained from the included dataset after removing exact duplicate rows. Predictions are estimates, not official emissions certification. Lab/test fuel consumption can differ from real-world driving; correlation does not prove causation.
