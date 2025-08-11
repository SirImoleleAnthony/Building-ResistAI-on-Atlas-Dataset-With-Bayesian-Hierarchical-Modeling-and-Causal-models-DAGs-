# Building-ResistAI-on-Atlas-Dataset-With-Bayesian-Hierarchical-Modeling-and-Causal-models-DAGs

# 🧬 ResistAI – AMR Stewardship Platform

**ResistAI** is a Streamlit-powered dashboard that transforms Pfizer’s ATLAS antimicrobial resistance (AMR) data into actionable insights for stewardship programs, policy-makers, and stakeholders. Explore data trends, perform statistical analysis, build predictive models, and forecast resistance dynamics through an interactive web interface.

Access app [here](https://resistaiplus.streamlit.app/)
---

## 📚 Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Methodology & Workflow](#methodology--workflow)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Access](#data-access)
- [Usage](#usage)
  - [App Overview](#app-overview)
  - [Pages & Functionality](#pages--functionality)
- [Results & Impact](#results--impact)
- [Roadmap & Future Work](#roadmap--future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🧠 Introduction

Antimicrobial resistance is a critical global health threat. **ResistAI** aims to empower decision-makers by transforming surveillance data into insights. By integrating statistical modeling, machine learning, forecasting, and visualization, it supports targeted interventions, data-driven policies, and stakeholder collaboration.

---

## 🚀 Key Features

- **Interactive Data Exploration**  
  Visualize demographic, species, and antibiotic-specific resistance patterns.

- **Bayesian Hierarchical Modeling (BHM)**  
  Map baseline resistance trends, hotspots, and MIC creep over time.

- **Machine Learning + Causal Inference**  
  Train classifiers and estimate causal effects via DAGs.

- **Time-Series Forecasting**  
  Leverage Prophet with domain regressors to project future resistance signals.

- **User-Friendly Streamlit Frontend**  
  Intuitive navigation, interactive visuals, and model export capabilities.

---

## 🔄 Methodology & Workflow

The ResistAI pipeline transforms raw AMR data into stewardship-ready insights through the following stages:
```
┌───────────────────────────────┐
│     ATLAS AMR Data Ingestion  │
└──────────────┬────────────────┘
               ↓
┌──────────────┴───────────────┐
│  Data Cleaning & Antibiotic  │
│     Subsetting (MIC-based)   │
└──────────────┬───────────────┘
               ↓
┌──────────────┴───────────────┐
│  Analytical Modules (via     │
│ Plotly & Bayesian Models)    │
│ – Demographics & Species     │
│ – BHM Statistical Mapping    │
└──────────────┬───────────────┘
               ↓
┌──────────────┴───────────────┐
│ ML Model Training & Causal   │
│ Modeling (DAGs, Classification) │
└──────────────┬───────────────┘
               ↓
┌──────────────┴───────────────┐
│ Time-Series Forecasting      │
│ (Prophet + Regressors)       │
└──────────────┬───────────────┘
               ↓
┌──────────────┴───────────────┐
│ Interactive Streamlit UI     │
│ – Visual Insights            │
│ – Forecasts & Predictions    │
│ – Downloadable Models        │
└──────────────────────────────┘
```

Each module is designed to support transparency, reproducibility, and stakeholder relevance.

---

## 🛠️ Getting Started

### 📦 Prerequisites
- Python 3.9 or later
- Recommended: set up a virtual environment (`venv`, `conda`)

### ⚙️ Installation

```bash
git clone https://github.com/SirImoleleAnthony/Building-ResistAI-on-Atlas-Dataset-With-Bayesian-Hierarchical-Modeling-and-Causal-models-DAGs-.git
cd Building-ResistAI-on-Atlas-Dataset-With-Bayesian-Hierarchical-Modeling-and-Causal-models-DAGs
pip install -r requirements.txt
```
### Run app
streamlit run app.py

## 📄 Pages & Functionality

- **Home** - Project overview, methodology, and call-to-action navigation
- **Data Analysis** - Demographic, species, and antibiotic resistance visualizations with insights
- **Statistical Analysis** - BHM-based visualizations including trends, hotspots, and MIC creep
- **Train Model** - Interface to train classification models with causal analysis and export
- **Make a Forecast** - Forecast MIC trajectories with Prophet, offering trend visualizations
- **Make Prediction**	- Predict susceptibility status, estimate causal effects, and share results
- **About** - Acknowledge contributors, data sources, and challenge details

## 📈 Results & Impact

**ResistAI delivers**:
- Data-driven AMR insights for effective stewardship.
- Geospatial and temporal resistance mapping to inform policy.
- Predictive modeling capabilities tailored to stakeholders’ needs.
- A collaborative platform for transparent analysis and decision-making.

**Bayesian Modelling of Antimicrobial Resistance: Change‑points, MIC Creep, and Policy‑Ready Signals**
Results from some selected antibiotics (Levofloxacin, Linezolid, and Meropenem)
- *Change-points*: The Bayesian Hierarchical Model (BHM) identified significant change-points in resistance trends.

  Levofloxacin Change-Points ![Levofloxacin Change-Points](Results/country_small_multiples_levofloxacin.png)

  Linezolid Change-Points ![Linezolid Change-Points](Results/country_small_multiples_linezolid.png)

  Meropenem Change-Points ![Meropenem Change-Points](Results/country_small_multiples_meropenem.png)

- *MIC Creep*: The BHM also detected MIC creep, indicating gradual increases in resistance levels over time.

  Levofloxacin MIC Creep for Top Hotspot Countries ![Levofloxacin MIC Creep for Top Hotspot Countries](Results/Levofloxacin_temporal_trends.png)

  Levofloxacin MIC Creep by Continents ![Levofloxacin MIC Creep by Continents](Results/Levofloxacin_temporal_trends_continent.png)

  Linezolid MIC Creep for Top Hotspot Countries ![Linezolid MIC Creep for Top Hotspot Countries](Results/Linezolid_temporal_trends.png)

  Linezolid MIC Creep by Continent ![Linezolid MIC Creep by Continent](Results/Linezolid_temporal_trends_continent.png)

  Meropenem MIC Creep for Top Hotspot Countries ![Meropenem MIC Creep for Top Hotspot Countries](Results/Meropenem_temporal_trends.png)

  Meropenem MIC Creep by Continent ![Meropenem MIC Creep by Continent](Results/Meropenem_temporal_trends_continent.png)

  - *Policy-Ready Signals*: The BHM results provide actionable insights for policymakers, highlighting areas of concern and potential interventions.
 
  Global-Positive Heatmap for Selected Organisms by Mechanism ![Global-Positive Heatmap for Selected Organisms by Mechanism](Results/heatmap_global_positive.png)

  Country Level Policy Priority for Selected Antibiotics ![Country Level Policy Priority for Selected Antibiotics](Results/policy_priority_heatmap_country.png)

  - *NS by Year for Top 12 Countries*: The BHM results also include a summary of the number of samples (NS) by year for the top 12 countries, providing a comprehensive overview of resistance trends in _E. coli_.
 
  Levofloxacin NS by Year for _E. coli_ ![Levofloxacin NS by Year for E. coli](Results/species_countries_Escherichia_coli_levofloxacin.png)

  Meropenem NS by Year for _E. coli_ ![Meropenem NS by Year for E. coli](Results/species_countries_Escherichia_coli_meropenem.png)


## 🧭 Roadmap & Future Work
- Incorporate genomic features and improve genetic data handling.
- Add multi-drug resistance forecasting and ensemble modeling.
- Enhance UI with dashboards, user onboarding, and multilingual support.
- Automate model retraining and deployment via CI/CD pipelines.

## 🤝 Contributing
This project is a submission for the 2025 Vivli AMR Surveillance Data Challenge. As such, external contributions are not required at this time. However, suggestions for enhancement, UI improvements, or stewardship-focused extensions are welcome via issues or discussions.

## 📄 License
This project is licensed under the MIT License.

## 🙏 Acknowledgments
- Pfizer ATLAS dataset – foundational AMR surveillance data
- Vivli AMR Surveillance Data Challenge 2025 – for inspiration and support
- Team members and contributors
    - Anthony Godswill Imolele - [LinkedIn](https://www.linkedin.com/in/godswill-anthony-850639199/)
    - Teye Richard Gamah - [LinkedIn](https://www.linkedin.com/in/gamah/)
    - Afolabi Owoloye - [LinkedIn](https://www.linkedin.com/in/afolabi-owoloye-a1b8a5b5/)
    - Kehinde Temitope Olubanjo - [LinkedIn](https://www.linkedin.com/in/temitope-kehinde/)
- Open-source libraries – Streamlit, Plotly, Prophet, XGBoost, PyMC, scikit-learn, pandas, etc.

**ResistAI**: _powering collaborative, actionable insights for antimicrobial resistance mitigation._
