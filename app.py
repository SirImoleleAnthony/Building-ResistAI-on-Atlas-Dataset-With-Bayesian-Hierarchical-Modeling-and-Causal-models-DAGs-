import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
from plotly import graph_objs as go 


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
from dowhy import CausalModel
import networkx as nx
import numpy as np
import pickle
import time
from io import BytesIO
import utils

#-----------Web page setting-------------------#
page_title = "ResistAI"
page_icon = "🦠🧬💊"
picker_icon = "👇"
#layout = "centered"

#--------------------Page configuration------------------#
st.set_page_config(page_title = page_title, page_icon = page_icon, layout = "wide")

#--------------------Web App Design----------------------#

selected = option_menu(
    menu_title = page_title + " " + page_icon,
    options = ['Home', 'Data Analysis', 'Statistical Analysis', 'Train Model', 'Make a Forecast', 'Make Prediction', 'About'],
    icons = ["house-fill", "book-half", "book", "gear", "activity", "robot", "envelope-fill"],
    default_index = 0,
    orientation = "horizontal"
)

@st.cache_data
# Load data function
def load_data(antibiotic):
    link = antibiotic + "_subset_clean.csv"
    data = pd.read_csv(link)
    return data

# Some lists
# 1. Antibiotic List
antiotics_list = ['Amikacin', 'Amoxycillin clavulanate', 'Ampicillin', 'Azithromycin', 'Cefepime', 'Cefoxitin', 'Ceftazidime', 'Ceftriaxone', 'Clarithromycin', 'Clindamycin', 'Erythromycin', 'Imipenem', 'Levofloxacin', 'Linezolid', 'Meropenem', 'Metronidazole', 'Minocycline', 'Penicillin', 'Piperacillin tazobactam', 'Tigecycline', 'Vancomycin',
'Ampicillin sulbactam', 'Aztreonam', 'Cefixime', 'Ceftaroline', 'Ceftazidime avibactam', 'Ciprofloxacin', 'Colistin', 'Daptomycin', 'Doripenem', 'Ertapenem', 'Gentamicin', 'Moxifloxacin', 'Oxacillin', 'Quinupristin dalfopristin', 'Teicoplanin',
'Tetracycline', 'Trimethoprim sulfa', 'Ceftolozane tazobactam', 'Meropenem vaborbactam', 'Cefpodoxime', 'Ceftibuten']    


if selected == 'Home':
    st.title("Welcome to ResistAI")
    st.subheader("Your AI-powered Antibiotic Resistance Analysis Tool")
    st.markdown(""" Welcome to ResistAI, a Streamlit-powered interactive platform that synthesizes cutting‐edge analytics and AI to tackle antimicrobial resistance (AMR). By leveraging global surveillance data such as Pfizer’s ATLAS, ResistAI empowers users—scientists, policymakers, and healthcare stakeholders—to drive antimicrobial stewardship, shape informed policies, and engage key players in combating AMR.
    """, unsafe_allow_html=True)
    st.markdown("""
    ## Methodology
    Our workflow brings together data processing, statistical modeling, machine learning, forecasting, and interactive visualization. Here’s a high-level diagram of the methodology:
    """)
    st.image("work_flow.png", caption="ResistAI Methodology Diagram", use_column_width=True)
    st.markdown("""
    ## Features
    - **Data Analysis**: Explore and visualize antibiotic resistance data.
    - **Statistical Analysis**: Explore the results from statistical tests done using Bayesian Hierarchical Model (BHM) to understand data distributions and relationships.
    - **Train Model**: Train a causal machine learning models on antibiotic resistance data.
    - **Make a Forecast**: Use Prophet algorithm and its regressors to forecast antibiotic resistance trends.
    - **Make Prediction**: Predict antibiotic resistance using trained models and perform causal effect estimation.
    """)
    st.markdown("""
    ## Results
    - **Robust Visual Analytics**: 
        - Interactive charts illustrating MIC distributions, resistance trends, and species- or antibiotic-specific insights, complete with narrative observations, implications, and stewardship actions.
    -  **Statistical Insight via Bayesian Hierarchical Modeling**: 
        - Reliable mapping of baseline resistance and temporal trends, including MIC "creep," with credible intervals visualized across countries and continents.
    -  **Machine Learning & Causal Frameworks**: 
        - Users can train classification models to predict resistance status, supported by causal inference through Directed Acyclic Graphs. Model performance metrics, feature importance, and causal effect plots are delivered interactively.
    - **Forecasting Capabilities**: 
        - Time-series MIC forecasting using Prophet, with key regressors and trend lines enabling future-focused decision-making.    
    - **Accessible Tools for Stakeholders**: 
        - From downloadable trained models to intuitive dashboards, _ResistAI_ bridges data and action—delivering real-world utility for stewardship, policy, and stakeholder collaboration.
    """)
    st.markdown("""
    ## Implications
    - **Informed Stewardship**:
        - By providing a comprehensive view of antibiotic resistance patterns, ResistAI enables healthcare professionals to make informed decisions about antibiotic use, ultimately contributing to better patient outcomes and reduced resistance rates.
    - **Policy Development**: 
        - The insights generated from ResistAI can inform policymakers about the current state of antibiotic resistance, helping them to develop effective policies and guidelines for antibiotic use and stewardship.
    - **Global Collaboration**:
        - ResistAI serves as a platform for global collaboration, allowing researchers and healthcare professionals to share data, insights, and best practices in the fight against antimicrobial resistance.
    - **Educational Resource**:
        - ResistAI can be used as an educational tool to raise awareness about antibiotic resistance among healthcare professionals, students, and the general public, fostering a culture of responsible antibiotic use.
    """)
    st.subheader("Experience ResistAI in action! Navigate seamlessly through the app using the menu above.")
    st.markdown("""
    - **Data Analysis** — Explore the landscape of demographics, species, and antibiotic resistance.
    - **Statistical Analysis** — Dive deep into Bayesian mappings of resistance trends and hotspots.
    - **Train Model** — Build and interpret predictive resistance models tailored to your context.
    - **Make a Forecast** — Project MIC trends forward for proactive strategy planning.
    - **Make Prediction** — Input your own data to predict resistance outcomes instantly.
    """)
 

elif selected == 'Data Analysis':
    with st.sidebar:
        st.header("🧪 Data Analysis Guide")
        st.markdown("This page helps you analyze antimicrobial resistance across demographics, bacteria, and antibiotics.")

        with st.expander("Step-by-Step Instructions"):
            st.markdown("""
            1. **Select an Antibiotic** from the dropdown list.
            2. **Choose an Analysis Type**:
                - **Demographical Analysis**: Understand how resistance varies by age, gender, etc.
                - **Bacterial Analysis**: Explore resistance across different bacterial species.
                - **Antibiotic Resistance Analysis**: See how resistance patterns evolve over time for the selected antibiotic.
            3. **Interpret Results**:
                - Hover over the interactive charts for more details.
                - Below each chart, read:
                    - 🧐 **Observations**
                    - 💡 **Implications**
                    - ✅ **Recommendations**
            """)

        st.success("Go ahead, dive into the data and uncover resistance patterns like a true scientist! 🧬")

    # Anaysis starts here
    st.title("Data Analysis")
    st.subheader("Explore and visualize antibiotic resistance data")
    # Select Antibiotic
    antibiotic = st.selectbox("Select Antibiotic", antiotics_list)

    # Load data
    data = load_data(f"{antibiotic}/{antibiotic}")

    # Analysis options
    analysis_type = st.selectbox("Select Analysis Type", ["Demographical Analysis", "Bacterial Species Analysis", "Antibiotic Resistance Analysis"])

    if analysis_type == "Demographical Analysis":
        st.subheader("Demographical Analysis")
        st.write("This section allows you to analyze the demographic data of patients")
        
        #Plot Gender Distribution
        utils.gender_distribution(data)

        #Plot Age Distribution
        utils.age_distribution(data)

        #Country Analysis
        utils.country_analysis(data)
        
        # Continent Analysis
        utils.continent_analysis(data)

        # Patient type Analysis
        utils.patient_type_analysis(data)

    elif analysis_type == "Bacterial Species Analysis":
        st.subheader("Bacteria (Species) Analysis")
        st.write("This section allows you to analyze the bacterial species data")
        
        # Top 10 Species Analysis
        utils.top_10_species_analysis(data, antibiotic)
        
        # Species per Country Analysis
        utils.organism_distribution_by_country(data)

        # Gender Distribution
        utils.organism_distribution_by_gender(data)

        # Bacteria Distribution and age
        utils.organism_distribution_by_age(data)

        # Yearly Bacteria Distribution
        utils.yearly_organism_distribution(data)

        # Bacteria distribution by family
        utils.organism_distribution_by_family(data)

        # Bacteria Distribution by resistance status
        utils.species_by_resistance_status(data, antibiotic)
        
        # Bacteria Distribution to MIC Value
        utils.species_distribution_by_mic(data, antibiotic)


    elif analysis_type == "Antibiotic Resistance Analysis":
        st.subheader("Antibiotic Resistance Analysis")
        st.write("This section allows you to analyze antibiotic resistance data")

        # Antibiotic Resistance Distribution
        utils.resistance_distribution(data, antibiotic)

        # Plot MIC Distribution
        utils.mic_distribution(data, antibiotic)

        # Plot Yearly Resistance Status
        utils.yearly_resistance_status(data, antibiotic)

        # Plot Bacteria Resistance Status
        utils.bacteria_resistance_status(data, antibiotic)
        

elif selected == 'Statistical Analysis':
    with st.sidebar:
        st.header("📐 Statistical Analysis Guide")
        st.markdown("View statistical charts and highlights based on advanced Bayesian modeling.")

        with st.expander("Step-by-Step Instructions"):
            st.markdown("""
            1. **Select an Antibiotic** to analyze.
            2. **Review BHM Results**:
                - BHM-generated plots and charts will load automatically.
            3. **Interpret Each Chart**:
                - Read the following provided for every chart:
                    - 🔍 **Observations**
                    - 📌 **Key Insights**
                    - 🧠 **Implications**
            """)

        st.success("Statistical power unlocked! Let the Bayesian magic guide your decisions 🧙‍♂️📊")

    # Statistical Analysis starts here
    st.title("Statistical Analysis")

    # Select Antibiotic
    antibiotic = st.selectbox("Select Antibiotic", antiotics_list)

    # Show statistical analysis
    utils.statistic_analysis(antibiotic)

elif selected == 'Train Model':
    with st.sidebar:
        st.header("🤖 Train a Model Guide")
        st.markdown("Train classification models and evaluate them with interactive metrics and charts.")

        with st.expander("Step-by-Step Instructions"):
            st.markdown("""
            1. **Select Inputs**:
                - Choose an **Antibiotic**.
                - Select a **Bacteria (Species)**.
                - Pick one of the 9 available **ML Algorithms**.
            2. **Training Conditions**:
                - If the selected bacteria only has one resistance status, a message will notify you — no training needed.
            3. **Click the 'Train Model' Button** to:
                - Train the model and make predictions.
                - View:
                    - 📈 **Accuracy, Precision, Recall, F1-Score**
                    - 📊 **Interactive Confusion Matrix**
                    - ⭐ **Feature Importance Chart** (if available)
                    - 🧪 **Causal Effects Chart** showing the influence of Phenotype, Source, and Country.
            4. **Download the Trained Model** as a `.pkl` file using the provided button.
            """)

        st.success("Model trained? Now go unleash your inner ML ninja! 🤺🧠")

    # Train Model starts here
    st.title("Train Model")
    st.subheader("Train machine learning models on antibiotic resistance data")
    # Select Antibiotic
    antibiotic = st.selectbox("Select Antibiotic", antiotics_list)

    # Load data
    data = load_data(f"{antibiotic}/{antibiotic}")
    # Fill all `NaN` values with mode in all columns that have NaN values
    for col in data.columns:
        if data[col].dtype == 'object' and data[col].isnull().any():
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Prediction
    utils.train_model(data, antibiotic)
    
# Make a Forecast page
elif selected == 'Make a Forecast':
    with st.sidebar:
        st.header("📅 Forecasting Guide")
        st.markdown("Predict future resistance trends using time-series modeling.")

        with st.expander("Step-by-Step Instructions"):
            st.markdown("""
            1. **Select Inputs**:
                - Choose an **Antibiotic**.
                - Select a **Bacteria (Species)**.
                - Enter the **Number of Years** you want to forecast.
            2. **Check for Data Availability**:
                - If not enough historical data is available, you’ll be notified.
            3. **Click 'Make Forecast' Button**:
                - See an **interactive line plot** of future trends.
                - Review:
                    - 🔍 **Observations**
                    - 🧠 **Implications**
                    - ✅ **Recommendations**
            """)

        st.success("Time travel complete! Now steer the future of AMR with your forecasts 🕒🔮")

    # Make a Forecast starts here
    st.title("Make a Forecast")

    # Select Antibiotic
    antibiotic = st.selectbox("Select Antibiotic", antiotics_list)
    # Load data
    data = load_data(f"{antibiotic}/{antibiotic}")

    # Forecast
    utils.forecast(data, antibiotic)
    

elif selected == 'Make Prediction':
    with st.sidebar:
        st.header("🔍 Make a Prediction Guide")
        st.markdown("Simulate conditions and predict bacteria resistance to selected antibiotics.")

        with st.expander("Step-by-Step Instructions"):
            st.markdown("""
            1. **Select Prediction Inputs**:
                - Choose an **Antibiotic**.
                - Select a **Bacteria (Species)**.
                - Pick additional conditions like **Year**, **Type of Study**, etc.
            2. **Click 'Make Prediction' Button**:
                - You’ll receive a message indicating the predicted resistance status:
                    - 🚦 **Resistant**, **Intermediate**, or **Susceptible**.
            3. **Explore Causal Effects**:
                - Choose a **Bacteria (Species)**.
                - Select **Treatment Factors** (e.g., Source, Phenotype).
                - Click **'Show Causal Effects'** to:
                    - View a **summary table**.
                    - See an **interactive horizontal bar chart**.
                    - Read accompanying highlights:
                        - 🔍 **Observations**
                        - 🧠 **Implications**
            """)

        st.success("Just one click and you’re predicting the future of resistance! 🧠🔬")

    # Make Prediction starts here
    st.title("Make Prediction")
    st.subheader("Predict antibiotic resistance using trained models")
    # Select Antibiotic
    antibiotic = st.selectbox("Select Antibiotic", antiotics_list)
    # Load data
    data = load_data(f"{antibiotic}/{antibiotic}")

    # Prediction
    utils.make_prediction(data, antibiotic)

    # Show causal effect estimation
    st.subheader("Causal Effect Estimation")
    utils.check_causal_effect(data, antibiotic)
    
    st.write("This section allows you to make predictions on antibiotic resistance data using the trained machine learning models. You can input new data and get predictions based on the models.")

elif selected == 'About':
    st.title("About ResistAI")
    st.subheader("About the Competition")
    st.markdown(
        """
    The 2025 Vivli AMR Surveillance Data Challenge, funded by GARDP, Paratek, Pfizer, and Vivli, is a groundbreaking initiative aimed at harnessing the power of the Vivli AMR Register to combat antimicrobial resistance (AMR). 
    
    This challenge seeks to drive critical research, foster collaboration and innovation, and push the boundaries of AMR research. 
    By leveraging the Vivli AMR Register's comprehensive datasets, participants can contribute meaningfully to reshaping our understanding and approach to AMR.

    Read more about the 2025 Vivli AMR Surveillance Data Challenge [here](https://amr.vivli.org/data-challenge/data-challenge-overview/).
        """,
        unsafe_allow_html=True
    )

    st.subheader("About the Dataset")
    st.markdown("""
    ResistAI is powered by Pfizer’s ATLAS (Antimicrobial Testing Leadership and Surveillance) data. 
    Pfizer’s ATLAS (Antimicrobial Testing Leadership and Surveillance) is a large-scale global program that aggregates AMR data from three surveillance initiatives (TEST, AWARE, INFORM), spanning more than 14 years across over 60–73 countries
    The dataset includes cumulative data on more than 556,000 bacterial isolates, with updates released approximately every 6 to 12 months via the ATLAS website and mobile app

    Specifically, the dataset accessible through the Vivli AMR Register includes:
        
    - Around 917,049 antibiotic isolates and 21,631 antifungal isolates as of June 2024.
    - It covers both pediatric data and limited genotypic information, such as the presence or absence of β-lactamase genes.
    - MIC (minimal inhibitory concentration) data, along with metadata including country, specimen type, year, organism, antimicrobial used, and basic demographics (e.g., age range, gender) are included
    
    This data empowers ResistAI users to make data-driven decisions, predict AMR dynamics, and stay informed about the latest trends in antimicrobial resistance.
    
    Read more about the Pfizer's ATLAS Program dataset [here](https://amr.vivli.org/members/research-programs/).
    """, unsafe_allow_html=True)
    

    st.subheader("About the Team")
    st.markdown(
        """
    The ResistAI team is a group of dedicated individuals committed to advancing the fight against antimicrobial resistance through innovative data analysis and machine learning techniques. Our team consists of data scientists, bioinformaticians, and healthcare professionals who are passionate about leveraging technology to improve patient outcomes and public health.
    The team members include:
    - **Anthony Godswill Imolele**
        - Research Scientist at Genomics Unit, Helix Biogen Institute, Nigeria. 
        - Affiliation: Helix Biogen Institute, Ogbomosho, Nigeria.
        - Computational Biologist, AI for Healthcare Researcher.
        - [LinkedIn](https://www.linkedin.com/in/godswill-anthony-850639199/)

    - **Teye Richard Gamah**
        - Affiliation: Valley View University, Accra, Ghana.
        - Bioinformatician, DataCamp Certified Data Scientist, and Machine Learning Engineer.
        - [LinkedIn](https://www.linkedin.com/in/gamah/)
    - **Afolabi Owoloye**
        - Research Fellow, PhD Student.
        - Affiliation: Centre for Genomic Research in Biomedicine, Mountain Top University, Nigeria.
        - Bioinformatician, AI/ML Engineer, Computational Biologist, and Data Analyst.
        - [LinkedIn](https://www.linkedin.com/in/afolabi-owoloye-a1b8a5b5/)
    - **Kehinde Temitope Olubanjo**
        - PhD student at the Hong Kong Polytechnic University, Hong Kong
        - Affiliation: Department of Industrial and Systems Engineering, the Hong Kong Polytechnic University, Hung Hom, Hong Kong. 
        - Industrial Systems Engineering, Data Envelopment Analyst.
        - [LinkedIn](https://www.linkedin.com/in/temitope-kehinde/)
    
    We believe that by combining our expertise in data science and healthcare, we can make a significant impact in understanding and combating antimicrobial resistance.
    """,
        unsafe_allow_html=True
    )


