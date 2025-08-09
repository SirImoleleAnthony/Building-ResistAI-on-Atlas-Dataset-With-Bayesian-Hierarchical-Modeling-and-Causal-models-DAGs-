import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
from plotly import graph_objs as go 
from prophet.plot import plot_plotly
import plotly.offline as py


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

# Plot Gender Distribution
def gender_distribution(df):
    fig = px.histogram(df, 
        x="Gender", 
        color="Gender",  
        title="Distribution of Gender",
        labels={'Gender': 'Patient Gender'},
        color_discrete_map={
            "Male": "#1f77b4",     
            "Female": "#ff7f0e",   
        }
    )
    fig.update_layout(
        yaxis_title="Frequency",
        xaxis_title="Patient Gender",
        showlegend=False
    )
    st.plotly_chart(fig)
    st.markdown("""
    - **Observation**: This bar chart shows the frequency of male, female, and possibly other gender identities among patients in the dataset. A skewed distribution may reflect sampling bias or demographic trends in healthcare access.
    - **Implications**: Understanding gender distribution is foundational for interpreting resistance trends by sex. If one gender dominates the dataset, it may influence downstream analyses and policy recommendations.
    - **Recommendations**:
        - **AMR Stewardship**: Ensure balanced sampling and reporting across genders to avoid biased conclusions.
        - **Policy**: Promote inclusive data collection practices in clinical settings.
        - **Stakeholder Engagement**: Work with gender health advocates to interpret and act on sex-disaggregated data responsibly.
                
                """)

# Plot age distribution
def age_distribution(df):
    age_order = [
        '0 to 2 Years', '3 to 12 Years', '13 to 18 Years',
        '19 to 64 Years', '65 to 84 Years', '85 and Over', 'Unknown'
    ]
    
    fig = px.histogram(
        df, 
        x="Age Group", 
        color="Age Group",  
        title='Age Distribution',
        category_orders={"Age Group": age_order},
        color_discrete_map={
            '0 to 2 Years': '#1f77b4',   
            '3 to 12 Years': '#ff7f0e',  
            '13 to 18 Years': '#2ca02c',  
            '19 to 64 Years': '#d62728',  
            '65 to 84 Years': '#9467bd',   
            '85 and Over': '#8c564b',      
            'Unknown': '#7f7f7f'           
        }
    )
    
    fig.update_layout(
        yaxis_title="Frequency",
        xaxis_title="Age Group",
        showlegend=False
    )
    
    st.plotly_chart(fig)
    st.markdown("""
    - **Observation**: This bar chart reveals how patients are distributed across age groups. A concentration in certain brackets (e.g., 19–64 or 65+) may reflect healthcare utilization patterns or disease prevalence.
    - **Implications**: Age distribution informs resistance trend analysis. For example, high representation in older age groups may correlate with increased resistance due to comorbidities or frequent antibiotic exposure.
    - **Recommendations**:
        - **AMR Stewardship**: Customize antibiotic guidelines for age-specific vulnerabilities.
        - **Policy**: Allocate resources to age groups with higher resistance rates and fund age-targeted surveillance and education programs.
        - **Stakeholder Engagement**: Collaborate with geriatric and pediatric health experts to interpret age-related resistance trends.""")

# Country Analysis
def country_analysis(df):
    top_10_countries = df['Country'].value_counts().head(10).reset_index()
    top_10_countries.columns = ['Country', 'Count']

    fig = px.bar(
        top_10_countries, 
        x='Country', 
        y='Count', 
        color='Country',  
        title='Top 10 Countries of the Study',
        labels={'Country': 'Country', 'Count': 'Frequency'},
        color_discrete_sequence=px.colors.qualitative.Set2 
    )

    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Frequency",
        showlegend=False
    )
    st.plotly_chart(fig)
    st.markdown("""
    - **Observation**: This bar chart highlights the top 10 countries contributing data to the study. A dominant country may indicate regional bias or stronger surveillance infrastructure.
    - **Implications**: Country distribution can reveal regional AMR patterns, healthcare access disparities, and surveillance gaps. Countries with high representation may have more robust data collection systems or higher disease burdens.
    - **Recommendations**:
        - **AMR Stewardship**: Tailor interventions to country-specific resistance trends and healthcare practices.
        - **Policy**: Prioritize countries with high resistance rates for targeted AMR policies and resource allocation.
        - **Stakeholder Engagement**: Collaborate with national health authorities to interpret country-level resistance data and inform local AMR strategies.
                """)

country_to_continent = {'France': 'Europe','Spain': 'Europe','Belgium': 'Europe','Italy': 'Europe','Germany': 'Europe',
    'Canada': 'North America','Ireland': 'Europe','Portugal': 'Europe','Israel': 'Asia','Greece': 'Europe','China': 'Asia',
    'United Kingdom': 'Europe','Kuwait': 'Asia','Poland': 'Europe','Switzerland': 'Europe','Hungary': 'Europe',
    'Austria': 'Europe','Colombia': 'South America','Chile': 'South America','Finland': 'Europe','Australia': 'Oceania',
    'Mexico': 'North America','Denmark': 'Europe','Sweden': 'Europe','Hong Kong': 'Asia','Japan': 'Asia','Croatia': 'Europe',
    'Malaysia': 'Asia','Nigeria': 'Africa','Kenya': 'Africa','Czech Republic': 'Europe','Netherlands': 'Europe',
    'Russia': 'Europe','Romania': 'Europe','Venezuela': 'South America','Thailand': 'Asia','Philippines': 'Asia',
    'Turkey': 'Asia','Korea, South': 'Asia','South Africa': 'Africa','Argentina': 'South America','Taiwan': 'Asia',
    'Brazil': 'South America','Panama': 'North America','Jordan': 'Asia','Saudi Arabia': 'Asia','Pakistan': 'Asia',
    'Guatemala': 'North America','Morocco': 'Africa','India': 'Asia','Singapore': 'Asia','Vietnam': 'Asia',
    'Latvia': 'Europe','Lithuania': 'Europe','Serbia': 'Europe','Dominican Republic': 'North America',
    'Costa Rica': 'North America','Ukraine': 'Europe','Ivory Coast': 'Africa','Lebanon': 'Asia','New Zealand': 'Oceania',
    'Qatar': 'Asia','Slovenia': 'Europe','Cameroon': 'Africa','Jamaica': 'North America','Bulgaria': 'Europe',
    'Norway': 'Europe','Honduras': 'North America','Puerto Rico': 'North America','Nicaragua': 'North America',
    'Slovak Republic': 'Europe','Oman': 'Asia','Malawi': 'Africa','Ghana': 'Africa','Uganda': 'Africa','Namibia': 'Africa',
    'Indonesia': 'Asia','Mauritius': 'Africa','Estonia': 'Europe','El Salvador': 'North America','Tunisia': 'Africa',
    'Egypt': 'Africa'
}

def continent_analysis(df):
    df['Continent'] = df['Country'].map(country_to_continent)
    continent_counts = df['Continent'].value_counts().reset_index()
    continent_counts.columns = ['Continent', 'Count']

    fig = px.bar(
        continent_counts, 
        x='Continent', 
        y='Count', 
        color='Continent',  
        title='Continents of the Study',
        labels={'Continent': 'Continent', 'Count': 'Frequency'},
        color_discrete_sequence=px.colors.qualitative.Set3  
    )

    fig.update_layout(
        xaxis_title="Continent",
        yaxis_title="Frequency",
        showlegend=False
    )

    st.plotly_chart(fig)
    st.markdown("""
    - **Observation**: This bar chart shows the distribution of data across continents. A skewed distribution may indicate regional biases in data collection or healthcare access.
    - **Implications**: Understanding continent-level representation helps identify global AMR patterns, healthcare disparities, and surveillance gaps. Regions with high representation may have stronger healthcare systems or more robust data collection practices. Underrepresentation from Africa or Oceania, for example, may obscure region-specific AMR challenges.
    - **Recommendations**:
        - **AMR Stewardship**: Tailor interventions to continent-specific resistance trends and healthcare practices.
        - **Policy**: Prioritize continents with high resistance rates for targeted AMR policies and resource allocation.
        - **Stakeholder Engagement**: Collaborate with regional health authorities to interpret continent-level resistance data and inform local AMR strategies.
                """)

# Patient type analysis
def patient_type_analysis(df):
    pat_type = df['In / Out Patient'].value_counts().reset_index()
    pat_type.columns = ['Patient Type', 'Count']

    fig = px.bar(
        pat_type, 
        x='Patient Type', 
        y='Count', 
        color='Patient Type',  
        title='Patient Type Analysis',
        labels={'Patient Type': 'Patient Type', 'Count': 'Frequency'},
        color_discrete_map={
            'Inpatient': '#d62728',   
            'Outpatient': '#1f77b4',  
            'None Given': '#ff7f0e',  
            'Unknown': '#7f7f7f'      
        }
    )

    fig.update_layout(
        xaxis_title="Patient Type",
        yaxis_title="Frequency",
        showlegend=False
    )

    st.plotly_chart(fig)
    st.markdown("""
    - **Observation**: This bar chart shows the distribution of patients by type (inpatient, outpatient, etc.). A higher frequency of inpatients may reflect more severe cases or hospital-based sampling.
    - **Implications**: Understanding patient type distribution is crucial for interpreting resistance trends. Patient type influences resistance dynamics. Inpatients often face higher antibiotic exposure and nosocomial infections, which can drive resistance. Outpatients may reflect community-acquired resistance trends.
    - **Recommendations**:
        - **AMR Stewardship**: Differentiate antibiotic protocols for inpatient vs outpatient settings.
        - **Policy**: Mandate patient-type tagging in AMR surveillance systems and allocate resources to patient types with higher resistance rates and fund targeted surveillance and education programs.
        - **Stakeholder Engagement**: Collaborate with hospital administrators and primary care providers to tailor stewardship efforts based on patient context.
        """)

# 2. Bacteria (Species) Analysis
# Top 10 Species
def top_10_species_analysis(df, antibiotic):
    top_10_organisms = df['Species'].value_counts().head(10).reset_index()
    top_10_organisms.columns = ['Species', 'Count']

    fig = px.bar(
        top_10_organisms, 
        x='Species', 
        y='Count', 
        color='Species',  
        title=f'Top 10 Species in the Study of {antibiotic}',
        labels={'Species': 'Bacterial Species', 'Count': 'Frequency'},
        color_discrete_sequence=px.colors.qualitative.Set1  
    )

    fig.update_layout(
        xaxis_title="Bacterial Species",
        yaxis_title="Frequency",
        showlegend=False
    )

    st.plotly_chart(fig)
    st.markdown("""
    - **Observation**: This bar chart displays the top 10 bacterial species isolated in the dataset. Dominant species like _Escherichia coli_, _Klebsiella pneumoniae_, or _Staphylococcus aureus_ often reflect common infection sources and clinical priorities.
    - **Implications**: Species prevalence guides empirical treatment decisions and diagnostic focus. High-frequency pathogens may also be linked to specific resistance mechanisms or hospital-acquired infections.
    - **Recommendations**:
        - **AMR Stewardship**: Prioritize species-specific antibiotic protocols based on prevalence.
        - **Policy**: Allocate resources to species with high resistance rates and fund targeted surveillance and education programs.
        - **Stakeholder Engagement**: Collaborate with microbiologists and infectious disease specialists to interpret species-level resistance data and inform local AMR strategies.
                """)
def organism_distribution_by_country(df):
    # Identify top species
    species_counts = df['Species'].value_counts()
    top_species = species_counts.head(10).index.tolist()

    # Multiselect for user to add more species
    selected_species = st.multiselect(
        "Select additional species to display (optional)",
        options=species_counts.index.tolist()[10:],
        default=[],
        help="Top 10 species are shown by default. Add more if needed.",
        key='species_country_multiselect'
    )

    # Combine default top species with user-selected ones
    display_species = top_species + selected_species
    filtered_df = df[df['Species'].isin(display_species)]

    # Plot
    fig = px.histogram(
        filtered_df,
        x="Country",
        color="Species",
        title="Distribution of Bacterial Species per Country of Study",
        labels={"Country": "Country", "Species": "Bacterial Species"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        yaxis_title="Number of Species",
        xaxis_title="Country",
        barmode='stack',
        showlegend=True,
        height=600,
        margin=dict(t=60, b=150)
    )

    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig)
    st.markdown("""
    - **Observation**: This stacked bar chart shows how different bacterial species are distributed across countries. A country with a wide variety of species may have robust surveillance or diverse clinical cases.
    - **Implications**: Understanding species distribution by country helps identify regional infection patterns, healthcare access disparities, and surveillance gaps. Countries with high counts of resistant species may need targeted interventions.
    - **Recommendations**:
        - **AMR Stewardship**: Tailor interventions to country-specific species profiles and healthcare practices.
        - **Policy**: Prioritize countries with high species diversity for targeted AMR policies and resource allocation.
        - **Stakeholder Engagement**: Collaborate with national labs and health ministries to interpret species distribution and guide public health responses.
                """)

def organism_distribution_by_gender(df):
    # Count species occurrences
    species_counts = df['Species'].value_counts()
    top_species = species_counts.head(10).index.tolist()

    # Slider to add more species
    additional_species_count = st.slider(
        "Add more species to display",
        min_value=0,
        max_value=len(species_counts) - 10,
        value=0,
        help="Top 10 species are shown by default. Use the slider to add more.",
        key='species_slider'
    )

    selected_species = species_counts.head(10 + additional_species_count).index.tolist()
    filtered_df = df[df['Species'].isin(selected_species)]

    # Plot
    fig = px.histogram(
        filtered_df,
        x="Gender",
        color="Species",
        title="Distribution of Bacterial Species by Gender",
        labels={'Gender': 'Gender', 'Species': 'Bacterial Species'},
        color_discrete_sequence=px.colors.qualitative.Set3,
        barmode='stack'
    )

    fig.update_layout(
        yaxis_title="Number of Bacteria",
        xaxis_title="Gender",
        showlegend=True,
    )

    st.plotly_chart(fig)
    st.markdown("""
    - **Observation**: This stacked bar chart shows how bacterial species are distributed across male, and female. Differences in species prevalence may reflect biological, behavioral, or healthcare access factors.
    - **Implications**: Gender-based species trends can influence diagnostics and treatment. For example, higher prevalence of _E. coli_ in females may relate to urinary tract infections, while _Staphylococcus aureus_ may be more common in males due to skin-related infections.
    - **Recommendations**:
        - **AMR Stewardship**: Tailor diagnostic and treatment protocols to gender-specific infection patterns.
        - **Policy**: Promote sex-disaggregated data collection in AMR surveillance.
        - **Stakeholder Engagement**: CWork with gender health experts and clinicians to interpret and act on these trends responsibly.
                """)

# Organisms and age
def organism_distribution_by_age(df):
    age_order = [
        '0 to 2 Years', '3 to 12 Years', '13 to 18 Years',
        '19 to 64 Years', '65 to 84 Years', '85 and Over', 'Unknown'
    ]

    # Count species occurrences
    species_counts = df['Species'].value_counts()
    top_species = species_counts.head(10).index.tolist()

    # Slider to add more species
    additional_species_count = st.slider(
        "Add more species to display",
        min_value=0,
        max_value=len(species_counts) - 10,
        value=0,
        help="Top 10 species are shown by default. Use the slider to add more.",
        key="species_age_slider"
    )

    selected_species = species_counts.head(10 + additional_species_count).index.tolist()
    filtered_df = df[df['Species'].isin(selected_species)]

    # Plot
    fig = px.histogram(
        filtered_df,
        x="Age Group",
        color="Species",
        title="Distribution of Bacterial Species by Age Group",
        category_orders={"Age Group": age_order},
        labels={'Age Group': 'Age Group', 'Species': 'Bacterial Species'},
        color_discrete_sequence=px.colors.qualitative.Set1,
        barmode='stack'
    )

    fig.update_layout(
        yaxis_title="Number of Bacterial Species",
        xaxis_title="Age Group",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    - **Observation**: This stacked bar chart shows how bacterial species are distributed across age groups. Certain species may dominate in pediatric or geriatric populations due to immune system differences or exposure patterns.
    - **Implications**: Age-based species trends can inform diagnostics and treatment. For example, higher prevalence of _Streptococcus pneumoniae_ in children may relate to respiratory infections, while _Staphylococcus aureus_ may be more common in older adults due to skin-related infections.
    - **Recommendations**:
        - **AMR Stewardship**: Tailor diagnostic and treatment protocols to age-specific infection patterns.
        - **Policy**: Promote age-disaggregated data collection in AMR surveillance.
        - **Stakeholder Engagement**: Collaborate with pediatric and geriatric health experts to interpret and act on these trends responsibly.
                """)

# Yearly distribution of organisms
def yearly_organism_distribution(df):
    # Aggregate total counts per species
    species_counts = df['Species'].value_counts()
    top_species = species_counts.head(10).index.tolist()

    # Multiselect for user to add more species
    additional_species = st.multiselect(
        "Add more species to display:",
        options=[sp for sp in species_counts.index if sp not in top_species],
        default=[],
        help="Top 10 species are shown by default. Add more if needed.",
        key='yearly_species_multiselect'
    )

    selected_species = top_species + additional_species
    filtered_df = df[df['Species'].isin(selected_species)]

    # Group by Year and Species
    organism_distribution = (
        filtered_df.groupby(['Year', 'Species'])['Species']
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Plot
    fig = px.bar(
        organism_distribution,
        x='Year',
        y=organism_distribution.columns[1:],  
        title='Distribution of Bacterial Species Over the Years',
        labels={'value': 'Number of Bacterial Species', 'variable': 'Species'},
        color_discrete_sequence=px.colors.qualitative.Set2,
        barmode='stack'
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Bacterial Species",
        showlegend=True,
    )

    st.plotly_chart(fig)
    st.markdown("""
    - **Observation**: This stacked bar chart shows how the frequency of different bacterial species changes annually. A rise in certain species may signal outbreaks, shifts in diagnostic focus, or evolving resistance.
    - **Implications**: Yearly species trends can inform public health responses and resource allocation. For example, a rise in _Klebsiella pneumoniae_ may indicate increased nosocomial infections, while a decline in _Streptococcus pneumoniae_ may reflect successful vaccination campaigns.
    - **Recommendations**:
        - **AMR Stewardship**: Tailor interventions to species trends over time.
        - **Policy**: Allocate resources to species with rising resistance rates and fund targeted surveillance and education programs.
        - **Stakeholder Engagement**: Collaborate with epidemiologists and public health officials to interpret yearly species trends and inform local AMR strategies.
                """)
# Distribution of species by family
def organism_distribution_by_family(df):
    # Count species occurrences
    species_counts = df['Species'].value_counts()
    top_species = species_counts.head(10).index.tolist()

    # Slider to add more species
    additional_species_count = st.slider(
        "Add more species to display",
        min_value=0,
        max_value=len(species_counts) - 10,
        value=0,
        help="Top 10 species are shown by default. Use the slider to add more.",
        key='species_family_slider'
    )

    selected_species = species_counts.head(10 + additional_species_count).index.tolist()
    filtered_df = df[df['Species'].isin(selected_species)]

    # Plot
    fig = px.histogram(
        filtered_df,
        x="Family",
        color="Species",
        title="Distribution of Bacterial Species by Their Family",
        labels={"Family": "Bacterial Family", "Species": "Bacterial Species"},
        color_discrete_sequence=px.colors.qualitative.Set3,
        barmode='stack'
    )

    fig.update_layout(
        yaxis_title="Number of Bacterial Species",
        xaxis_title="Bacterial Family",
        showlegend=True,
        margin=dict(t=60, b=150)
    )

    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig)
    st.markdown("""
    - **Observation**: This stacked bar chart shows how bacterial species are distributed across families like _Enterobacteriaceae_, _Staphylococcaceae_, or _Pseudomonadaceae_. Dominance of certain families may reflect clinical relevance or shared resistance traits.
    - **Implications**: Understanding species distribution by family helps identify regional infection patterns, healthcare access disparities, and surveillance gaps. Family-level trends help identify clusters of resistance. For example, _Enterobacteriaceae_ often harbor extended-spectrum beta-lactamases (ESBLs), making them critical targets for stewardship.
    - **Recommendations**:
        - **AMR Stewardship**: Tailor interventions to family-specific species profiles and healthcare practices.
        - **Policy**: Prioritize families with high species diversity for targeted AMR policies and resource allocation.
        - **Stakeholder Engagement**: Collaborate with microbiologists and infectious disease specialists to interpret family-level resistance data and inform local AMR strategies.
                """)

# Distribution of species per antibiotic resistance status
def species_by_resistance_status(df, anti):
    resistance_col = anti + "_I"

    # Count species occurrences
    species_counts = df['Species'].value_counts()
    top_species = species_counts.head(10).index.tolist()

    # Slider to add more species
    additional_species_count = st.slider(
        "Add more species to display",
        min_value=0,
        max_value=len(species_counts) - 10,
        value=0,
        help="Top 10 species are shown by default. Use the slider to add more.",
        key="species_resistance_slider"
    )

    selected_species = species_counts.head(10 + additional_species_count).index.tolist()
    filtered_df = df[df['Species'].isin(selected_species)]

    # Plot
    fig = px.histogram(
        filtered_df,
        x=resistance_col,
        color="Species",
        title=f"Distribution of Bacterial Species per {anti} Resistance Status",
        labels={
            resistance_col: f"{anti} Resistance Status",
            "Species": "Bacterial Species"
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
        barmode='stack'
    )

    fig.update_layout(
        yaxis_title="Number of Bacterial Species",
        xaxis_title=f"{anti} Resistance Status",
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""
    - **Observation**: This stacked bar chart shows how bacterial species are distributed across resistance categories—Resistant, Intermediate, and Susceptible—for **{anti}** antibiotic. A dominance of red (Resistant) bars for certain species signals clinical concern. Empty bars for certain resistance statuses indicate no isolates were found in those categories.
    - **Implications**: Understanding species resistance profiles helps identify treatment challenges and inform empirical therapy decisions. For example, a high prevalence of _Escherichia coli_ with resistance to **{anti}** may indicate treatment failure risks.
    - **Recommendations**:
        - **AMR Stewardship**: Tailor empirical therapy protocols based on species resistance profiles.
        - **Policy**: Allocate resources to species with high resistance rates and fund targeted surveillance and education programs.
        - **Stakeholder Engagement**: Collaborate with infectious disease experts to interpret resistance dynamics and update treatment guidelines.
                """)
# Bacteria Distribution to MIC Value
def species_distribution_by_mic(df, antibiotic):
    # Count species occurrences
    species_counts = df['Species'].value_counts()
    top_species = species_counts.head(10).index.tolist()

    # Slider to add more species
    additional_species_count = st.slider(
        "Add more species to display",
        min_value=0,
        max_value=len(species_counts) - 10,
        value=0,
        help="Top 10 species are shown by default. Use the slider to add more.",
        key="species_mic_slider"
    )

    selected_species = species_counts.head(10 + additional_species_count).index.tolist()
    filtered_df = df[df['Species'].isin(selected_species)]

    # Plot
    fig = px.histogram(
        filtered_df,
        x=antibiotic,
        color="Species",
        title=f"Distribution of {antibiotic} MIC Resistance Values per Bacterial Species",
        labels={antibiotic: "MIC Value", "Species": "Bacterial Species"},
        color_discrete_sequence=px.colors.qualitative.Set3,
        barmode='stack',
        #category_orders={antibiotic: sorted(df[antibiotic].unique())}
    )

    fig.update_layout(
        yaxis_title="Frequency",
        xaxis_title=f"{antibiotic} MIC Value",
        showlegend=True,
        xaxis_type='category',
    )

    st.plotly_chart(fig)
    st.markdown(f"""
    - **Observation**: This stacked bar chart shows how bacterial species are distributed across MIC values for **{antibiotic}**. Species clustering at higher MICs (e.g., >32, >64 or above) suggests reduced susceptibility and potential resistance. Empty bars for certain MIC values indicate no isolates were found at those levels.
    - **Implications**: Understanding MIC distribution helps assess treatment challenges and inform empirical therapy decisions. For example, a high prevalence of _Klebsiella pneumoniae_ with MIC values > 32 for **{antibiotic}** may indicate treatment failure risks.
    - **Recommendations**:
        - **AMR Stewardship**: Tailor empirical therapy protocols based on species MIC profiles.
        - **Policy**: Allocate resources to species with high MIC values and fund targeted surveillance and education programs.
        - **Stakeholder Engagement**: Collaborate with microbiologists to interpret MIC dynamics and update treatment guidelines.
                """)
 

# Antibiotic Analysis
# 1. Plot Resistance Status Distribution
def resistance_distribution(df, anti):
    # Ensure categorical ordering for interpretability
    status_col = f'{anti}_I'
    df[status_col] = pd.Categorical(df[status_col], categories=["Susceptible", "Intermediate", "Resistant"], ordered=True)

    # Define color mapping for stewardship clarity
    color_map = {"Susceptible": "green", "Intermediate": "orange", "Resistant": "red"}

    # Create histogram with color coding and category order
    fig = px.histogram(
        df,
        x=status_col,
        color=status_col,
        category_orders={status_col: ["Susceptible", "Intermediate", "Resistant"]},
        color_discrete_map=color_map,
        title=f"Resistance Status Distribution for {anti}",
        labels={status_col: "Resistance Status"},
        text_auto=True
    )

    # Update layout for better readability
    fig.update_layout(
        yaxis_title="Number of Isolates",
        xaxis_title="Resistance Category",
        legend_title="Status",
        bargap=0.2,
        template="plotly_white"
    )

    # Calculate percentages for annotation
    res_counts = df[status_col].value_counts(normalize=True).mul(100).round(1)
    total_counts = df[status_col].value_counts()

    # Display plot
    st.plotly_chart(fig)
    st.markdown(f"""
    - **Observation**: This bar chart shows the distribution of resistance status for **{anti}**. A high percentage of resistant isolates (red) indicates potential treatment challenges. - **{res_counts.get('Resistant', 0)}%** of isolates show resistance to **{anti}**, indicating potential concern. 
    - **Implications**: Understanding resistance distribution is crucial for guiding empirical therapy decisions. High resistance rates may signal treatment failure risks and necessitate alternative strategies.
    - **Recommendations**:
        - **AMR Stewardship**: Prioritize stewardship interventions for antibiotics with high resistance rates.
        - **Policy**: Allocate resources to address resistance challenges and fund targeted surveillance and education programs.
        - **Stakeholder Engagement**: Collaborate with clinicians and pharmacists to interpret resistance data and update treatment guidelines.
    """)

# 2. Plot MIC Distribution
def mic_distribution(df, anti):

    # Convert MIC values to string for categorical binning
    df[anti] = df[anti].astype(str)

    fig = px.histogram(
        df,
        x=anti,
        color="Species",
        title=f"Distribution of {anti} MIC Values by Species",
        #category_orders={anti: mic_order},
        labels={anti: "MIC Value", "Species": "Bacterial Species"},
        color_discrete_sequence=px.colors.qualitative.Set1 
    )

    fig.update_layout(
        xaxis_title="MIC Value",
        yaxis_title="Number of Isolates",
        barmode='stack',
        showlegend=True,
        xaxis_type='category'
    )

    st.plotly_chart(fig)
    st.markdown(f"""
    - **Observation**: This bar chart shows the distribution of MIC values for **{anti}**. A concentration of isolates at higher MICs (e.g., >32) suggests reduced susceptibility and potential resistance.
    - **Implications**: Understanding MIC distribution helps assess treatment challenges and inform empirical therapy decisions. For example, a high prevalence of isolates with MIC > 32 for **{anti}** may indicate treatment failure risks.
    - **Recommendations**:
        - **AMR Stewardship**: Tailor empirical therapy protocols based on MIC profiles.
        - **Policy**: Allocate resources to address high MIC challenges and fund targeted surveillance and education programs.
        - **Stakeholder Engagement**: Collaborate with microbiologists to interpret MIC dynamics and update treatment guidelines.
    """)

# 3. Yearly Antibiotic Resistance Status
def yearly_resistance_status(df, anti):
    anti_MIC = anti + "_I"
    resistance = df.groupby(['Year', anti_MIC])[anti_MIC].count().unstack().fillna(0)
    fig = px.bar(resistance, 
                        x=resistance.index, 
                        y=resistance.columns, 
                        title=f'Distribution of {anti} Resistance Status over the Years',
                        labels={'x': 'Year', 'value': 'Frequency'},
                        color_discrete_map={'Resistant': 'red', 'Intermediate': 'orange', 'Susceptible': 'green'})
    fig.update_layout(legend_title_text='Resistance Status' )
    st.plotly_chart(fig)
    st.markdown(f"""
    - **Observation**: The bar chart shows the yearly distribution of resistance status—Resistant (red), Intermediate (orange), and Susceptible (green)—for {anti} across multiple years. A visible shift toward red bars over time would indicate increasing resistance, while dominance of green suggests sustained effectiveness.
    - **Implications**: Temporal resistance patterns reflect the evolving impact of antibiotic usage, resistance gene spread, and clinical outcomes. A rise in red (Resistant) bars signals growing treatment challenges, while orange (Intermediate) bars highlight borderline cases that complicate dosing and efficacy decisions.
    - **Recommendations**:
        - **AMR Stewardship**: Strengthen AMR stewardship by integrating resistance trend monitoring into clinical workflows.
        - **Policy**: Inform policy by enforcing time-sensitive antibiotic usage regulations and funding longitudinal surveillance.
        - **Stakeholder Engagement**: Engage stakeholders—clinicians, public health officials, and researchers—to respond to resistance trajectories and promote sustainable antibiotic practices.
    """)

# 4. Bacteria Resistance Status
def bacteria_resistance_status(df, anti):
    anti_MIC = anti + "_I"

    # Get top species for selection
    species = df['Species'].value_counts().head(15).index.tolist()
    target_species = st.selectbox("Select Species", species)

    # Filter for selected species
    species_df = df[df["Species"] == target_species]

    # Check if the resistance column exists and is non-empty
    if anti_MIC not in species_df.columns or species_df[anti_MIC].dropna().empty:
        st.warning(f"No resistance data available for _{target_species}_ and {anti}.")
        return

    # Group and reshape
    resistance = species_df.groupby(['Year', anti_MIC])[anti_MIC].count().unstack().fillna(0)

    if resistance.empty:
        st.warning(f"No resistance records found for _{target_species}_ over the years.")
        return

    resistance_melted = resistance.reset_index().melt(
        id_vars='Year', var_name='Resistance Status', value_name='Count'
    )

    # Plot
    fig = px.bar(
        resistance_melted,
        x="Year",
        y="Count",
        color="Resistance Status",
        title=f'{target_species} Resistance to {anti} over the Years',
        labels={'Year': 'Year', 'Resistance Status': 'Status', 'Count': 'Frequency'},
        color_discrete_map={
            'Resistant': 'red',
            'Intermediate': 'orange',
            'Susceptible': 'green'
        }
    )

    fig.update_layout(legend_title_text='Resistance Status')
    st.plotly_chart(fig)
    st.markdown(f"""
    - **Observation**: The bar chart shows the yearly distribution of resistance status—Resistant (red), Intermediate (orange), and Susceptible (green)—for _{target_species}_ against **{anti}**. A visible shift toward red bars over time would indicate increasing resistance, while dominance of green suggests sustained effectiveness.
    - **Implications**: Temporal resistance patterns reflect the evolving impact of antibiotic usage, resistance gene spread, and clinical outcomes. A rise in red (Resistant) bars signals growing treatment challenges, while orange (Intermediate) bars highlight borderline cases that complicate dosing and efficacy decisions.
    - **Recommendations**:
        - **AMR Stewardship**: Strengthen AMR stewardship by integrating resistance trend monitoring into clinical workflows.
        - **Policy**: Inform policy by enforcing time-sensitive antibiotic usage regulations and funding longitudinal surveillance.
        - **Stakeholder Engagement**: Engage stakeholders—clinicians, public health officials, and researchers—to respond to resistance trajectories and promote sustainable antibiotic practices.
    """)

# Statistic Analysis
def statistic_analysis(anti):

    st.subheader(f"Statistical Analysis for {anti} Resistance")
    st.markdown(f"""
                Using Bayesian Hierarchical Modelling (BHM) with the {anti} AMR dataset, a model was fitted to estimate country-specific resistance rates, accounting for continent and year effects, followed by clustering to identify spatial hotspots of high resistance. The model parameterized country-level baseline resistance (`α`) and temporal trends (`β`) with normal priors, using a Binomial likelihood for resistance counts and incorporating continent-level random effects with normal hyperpriors.
                """)

    st.subheader("Trends and Patterns over time")
    st.image(f"{anti}/{anti}_trend_pattern.png", caption=f"Country-specific trends in {anti} resistance highlight critical targets for AMR stewardship interventions and policy actions, enabling informed decision-making by stakeholders.", use_column_width=True)
    st.markdown("""
    - **Observations**: Each country’s slope (`β`) shows whether resistance is rising (`β>0`) or falling (`β<0`) over time; the blue band is the 95% Highest Density Intervals (HDI) around that estimate. This single value effectively summarizes the resistance trend for that country.
    - **Key Insights**:
        - **Rising Resistance**: Rising slopes mean resistance is growing (antibiotics are becoming less effective), whereas declining slopes mean resistance is improving or stable. Such increases have been recognized as serious threats to infection management and public health, so a positive slope flags where the problem is intensifying.
        - **Country-Specific Trends**: The analysis reveals significant variations in antimicrobial resistance patterns across different countries, indicating the need for tailored interventions.
        - **Temporal Patterns**: The trends over time highlight the dynamic nature of antimicrobial resistance, emphasizing the importance of continuous monitoring and adaptive strategies in such countries.
        - **Implications**: Identifying countries with rising resistance trends supports strategic policymaking, allows prioritization of early intervention, and encourages coordinated stakeholder responses to address emerging AMR challenges.
    """)
    st.image(f"{anti}/{anti}_mean_resistance_beta_map.png", caption=f"A global heatmap of {anti} mean resistance slopes pinpoints countries with accelerating AMR trends and missing data, guiding evidence-based stewardship, policy targeting, and stakeholder mobilization.", use_column_width=True)
    st.markdown("""
    - **Observations**: _Bright yellow_ country signals a high mean slope of resistance – i.e. resistance is _growing quickly_ there. A _deep purple_ country signals a low or negative slope – i.e. resistance is _stable_ or _falling_. Intermediate colors (_blue_, _green_) are _moderate increases_. _Gray/hatched_ means “_no data_” for such countries.
    - **Key Insights**:
        - **Global Patterns**: The map reveals global patterns in antimicrobial resistance, with certain regions showing higher levels of resistance. This highlights the need for targeted interventions in specific areas.
        - **Data Gaps**: Countries with insufficient data are marked in gray, indicating the need for improved data collection and reporting to better understand and address antimicrobial resistance.
        - **Implications**: The map serves as a valuable tool for policymakers and public health officials to identify priority areas for intervention and resource allocation.
    """)

    # Country-Specific Differences
    st.subheader("Country-Specific Differences")
    st.image(f"{anti}/{anti}_baseline_resistance_alpha.png", caption=f"Country-specific differences in {anti} resistance highlight the need for tailored interventions and policies to address unique challenges in each region.", use_column_width=True)
    st.markdown("""
    - **Observations**: Each country’s baseline resistance (intercept, `α`) varies, with confidence intervals (HDI) indicating uncertainty with some countries showing higher initial resistance.
    - **Key Insights**:
        - **Baseline Resistance**: The analysis reveals significant differences in baseline resistance levels across countries, indicating the need for tailored interventions and policies.
        - **Confidence Intervals**: The confidence intervals provide insights into the uncertainty of the estimates, highlighting areas where further research is needed.
        - **Implications**:  Identifying countries with elevated baseline resistance supports strategic policymaking, allows prioritization of early intervention, and encourages coordinated stakeholder responses to address entrenched AMR challenges
    """) 
    st.image(f"{anti}/{anti}_mean_baseline_resistance_alpha_map.png", caption=f"Country-specific {anti} resistance patterns reveal critical targets for AMR stewardship interventions and policy actions, enabling informed decision-making by stakeholders.", use_column_width=True)
    st.markdown("""
    - **Observations**: _Bright yellow_ country signals a high mean baseline resistance – i.e. resistance is _high_ there. A _deep purple_ country signals a low or negative baseline resistance – i.e. resistance is _low_. Intermediate colors (_blue_, _green_) are _moderate levels_. _Gray/hatched_ means “_no data_” for such countries.
    - **Key Insights**:
        - **Global Patterns**: The map reveals global patterns in antimicrobial resistance, with certain regions showing higher levels of resistance. These geographic differences highlight where AMR has historically been more prevalent or where surveillance is lacking, helping to distinguish high-burden contexts from low-risk or under-monitored areas.
        - **Data Gaps**: Countries with insufficient data are marked in gray, indicating the need for improved data collection and reporting to better understand and address antimicrobial resistance.
        - **Implications**: The map serves as a valuable tool for policymakers and public health officials to identify priority areas for intervention and resource allocation.
    """)
    # Continent-Specific Differences
    st.subheader("Continent-Specific Differences")
    st.image(f"{anti}/{anti}_baseline_resistance_alpha_continent.png", caption=f"Continent-specific differences in {anti} resistance highlight the need for tailored interventions and policies to address unique challenges in each region.", use_column_width=True)
    st.markdown("""
    - **Observations**: The plot shows variation in the baseline resistance levels (in log-odds) across different continents, with some continents having consistently higher average resistance than others. Each continent’s baseline resistance varies, with confidence intervals (HDI) indicating uncertainty.
    - **Key Insights**:
        - **Baseline Resistance**: The analysis reveals significant differences in baseline resistance levels across continents, indicating the need for tailored interventions and policies. This implies that antimicrobial resistance is not geographically uniform; certain continents may face more entrenched resistance challenges due to varying healthcare practices, antibiotic usage patterns, or surveillance efforts.
        - **Confidence Intervals**: The confidence intervals provide insights into the uncertainty of the estimates, highlighting areas where further research is needed.
        - **Implications**: Recognizing these disparities helps global health organizations and policymakers prioritize regions for enhanced surveillance, funding, and interventions. It also allows stakeholders to tailor region-specific AMR action plans that reflect local realities and global responsibilities.
    """)
    st.image(f"{anti}/{anti}_mean_baseline_resistance_alpha_continent_map.png", caption=f"Continent-specific {anti} resistance patterns reveal critical targets for AMR stewardship interventions and policy actions, enabling informed decision-making by stakeholders.", use_column_width=True)
    st.markdown("""
    - **Observations**: _Bright yellow_ continent signals a high mean baseline resistance – i.e. resistance is _high_ there. A _deep purple_ continent signals a low or negative baseline resistance – i.e. resistance is _low_. Intermediate colors (_blue_, _green_) are _moderate levels_. _Gray/hatched_ means “_no data_” for such continents.
    - **Key Insights**:
        - **Global Patterns**: The map reveals global patterns in antimicrobial resistance, with certain continents showing higher levels of resistance. These geographic differences highlight where AMR has historically been more prevalent or where surveillance is lacking, helping to distinguish high-burden contexts from low-risk or under-monitored areas.
        - **Data Gaps**: Continents with insufficient data are marked in gray, indicating the need for improved data collection and reporting to better understand and address antimicrobial resistance. 
        - **Implications**: Recognizing continent-level AMR patterns supports international cooperation, guides regional policy prioritization, and helps AMR stakeholders tailor interventions that align with each continent's unique resistance profile. The also map serves as a valuable tool for policymakers and public health officials to identify priority areas for intervention and resource allocation.
    """)

    # Hotspots and Emerging Trends
    st.subheader("Hotspots and Emerging Trends")
    st.image(f"{anti}/{anti}_hotspots_map.png", caption=f"Hotspots and emerging trends in {anti} resistance highlight critical targets for AMR stewardship interventions and policy actions, enabling informed decision-making by stakeholders.", use_column_width=True)
    st.markdown(f"""
    - **Observations**: The first map identifies countries with high baseline resistance (`α`), while the second highlights countries with the fastest-growing resistance trends (`β`), revealing that these do not always overlap.
    - **Key Insights**:
        - **Hotspots (High Baseline)**: Countries with high `α` values (blue outline) have elevated resistance levels for {anti}, indicating current public health challenges. These are critical areas for immediate intervention, such as enhanced antibiotic stewardship.
        - **Hotspots (Rapid Increase)**: Countries with high `β
        ` values (blue outline) show rapid increases in resistance over time, signaling emerging threats. These areas may require proactive measures, like surveillance or policy changes, to prevent worsening resistance.
        - **Geographical Patterns**: The maps reveal spatial clustering, that is, if multiple high-resistance countries are in the same region. This suggests shared factors (e.g., healthcare practices, antibiotic usage) that contribute to resistance. Understanding these patterns can help target interventions more effectively.
        - **Implications**: This dual-mapping approach empowers AMR stakeholders to act proactively: countries with high baselines require containment policies and antimicrobial usage reviews. This ensures stewardship efforts are evidence-driven and globally equitable.
    """)

    # MIC “creep” Detection in Non-Resistant Population over Time
    st.subheader("MIC 'Creep' Detection in Non-Resistant Population over Time")
    st.image(f"{anti}/{anti}_mic_creep_trend.png", caption=f"Tracking MIC creep trends in {anti} across countries enables timely AMR stewardship interventions and supports data-driven policymaking for global AMR stakeholders.", use_column_width=True)
    st.markdown("""
    - **Observations**: This plot visualizes the slope of log₂ MIC over time (MIC creep) for each country, showing whether susceptibility is declining (positive slope) or improving (negative slope) in the non-resistant population.
    - **Key Insights**:
        - **MIC Creep**: A positive slope suggests increasing MIC values over time—indicating reduced susceptibility despite being below resistance breakpoints—which is an early warning signal of evolving resistance. The analysis reveals trends in MIC creep, indicating whether susceptibility to the antibiotic is declining or improving over time. This helps identify potential issues with antibiotic effectiveness and guides interventions.
        - **Country-Specific Trends**: The plot highlights country-specific trends in MIC creep, allowing for targeted interventions and resource allocation.
        - **Implications**: Identifying countries with upward MIC creep trends allows AMR stakeholders to prioritize surveillance and interventions, adjust treatment guidelines, and inform early containment policies before full resistance emerges. Tracking MIC creep trends supports data-driven policymaking by providing insights into the effectiveness of antimicrobial stewardship efforts and guiding future interventions.
    """)

    st.image(f"{anti}/{anti}_mic_creep_trend_map.png", caption=f"Global MIC creep trends in {anti} resistance highlight critical targets for AMR stewardship interventions and policy actions, enabling informed decision-making by stakeholders.", use_column_width=True)
    st.markdown("""
    - **Observations**: This map visualizes the global trends in MIC creep for the antibiotic, showing which countries are experiencing increasing or decreasing MIC values over time. 
    - **Key Insights**:
        - **Global Patterns**: The map reveals global patterns in MIC creep, with certain countries showing increasing MIC values over time. Countries shaded with higher slope values indicate increasing MIC creep—suggesting that even non-resistant strains are gradually becoming less susceptible, signaling a potential shift toward resistance. This highlights the need for targeted interventions in specific areas.
        - **Data Gaps**: Countries with insufficient data are marked in gray, indicating the need for improved data collection and reporting to better understand and address antimicrobial resistance.
        - **Implications**: This spatial visualization helps identify geographic hotspots of emerging resistance pressure, allowing policymakers and AMR stakeholders to allocate resources, update national treatment guidelines, and implement region-specific stewardship interventions proactively.
    """)



# Forecasting with Prophet
def forecast(df, anti):
    species = df['Species'].unique().tolist()
    target_species = st.selectbox("Select Species", species)
    df_fc = df[df["Species"] == target_species].copy()

    def parse_mic(value):
        value = str(value).strip()
        try:
            if value.startswith('<='):
                return float(value[2:]) * 0.9
            elif value.startswith('<'):
                return float(value[1:]) * 0.9
            elif value.startswith('>='):
                return float(value[2:]) * 1.1
            elif value.startswith('>'):
                return float(value[1:]) * 1.1
            else:
                return float(value)
        except:
            return np.nan

    df_fc[f'{anti}_numeric'] = df_fc[anti].apply(parse_mic)
    df_fc[f'{anti}_numeric'] = np.log2(df_fc[f'{anti}_numeric'])
    df_fc.dropna(subset=[f'{anti}_numeric'], inplace=True)

    cols_to_use = ['Study', 'Species', 'Family', 'Country', 'Gender', 'Age Group', 'Speciality', 
               'Source', 'In / Out Patient', 'Year','Phenotype', anti, f'{anti}_I', f'{anti}_numeric']
    
    df_fc = df_fc[cols_to_use].copy()

    # Check for data sufficiency before filling NaNs
    min_required_rows = 10 
    if df_fc.shape[0] < min_required_rows:
        st.warning(f"""
    🚨 **Insufficient Data for Forecasting**

    The selected combination of **{anti}** and _{target_species}_ has only **{df_fc.shape[0]}** records, which is below the minimum required (**{min_required_rows}**) for reliable forecasting.

    This data gap is more than a technical limitation—it reflects a broader challenge in antimicrobial resistance (AMR) surveillance.

    📊 **Why This Matters**:
    - Sparse data can obscure resistance trends and delay detection of emerging threats.
    - Forecasting models depend on consistent, high-quality inputs to guide empirical therapy.
    - Data gaps weaken the foundation for evidence-based stewardship and policy decisions.

    🧬 **Call to Action**:
    - Strengthen AMR data collection across healthcare settings.
    - Standardize MIC reporting and phenotype classification.
    - Invest in regional surveillance systems to close the data gap.

    Let’s transform data scarcity into a catalyst for smarter stewardship and stronger policy.
    """)
        st.stop()

    # Fill missing values
    for col in df_fc.columns:
        if df_fc[col].dtype == 'object':
            mode_val = df_fc[col].mode()
            if not mode_val.empty:
                df_fc[col] = df_fc[col].fillna(mode_val[0])
                #df_fc[col].fillna(mode_val[0], inplace=True)
            else:
                df_fc[col] = df_fc[col].fillna("Unknown")
                #df_fc[col].fillna("Unknown", inplace=True)  
        else:
            mean_val = df_fc[col].mean()
            if not np.isnan(mean_val):
                df_fc[col] = df_fc[col].fillna(mean_val)
                #df_fc[col].fillna(mean_val, inplace=True)
            else:
                df_fc[col] = df_fc[col].fillna(0)
                #df_fc[col].fillna(0, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in df_fc.select_dtypes(include=['object', 'category']).columns:
        df_fc[col] = le.fit_transform(df_fc[col])

    # Prepare Prophet-compatible columns
    df_fc['ds'] = pd.to_datetime(df_fc['Year'], format='%Y')
    df_fc['y'] = df_fc[f'{anti}_numeric']

    # Forecasting UI
    num_years = st.number_input("Number of Years to Forecast", min_value=1, max_value=10, value=3)

    if st.button("Make Forecast"):
        with st.spinner(f"Forecasting **{num_years}** years ahead for _{target_species}_ using **{anti}** data..."):
            model = Prophet()

            # Add only valid regressors
            regressors = [col for col in df_fc.columns if col not in ['ds', 'y', 'Year', f'{anti}_numeric']]
            for col in regressors:
                model.add_regressor(col)

            model.fit(df_fc[['ds', 'y'] + regressors])

            future = model.make_future_dataframe(periods=num_years, freq='YE')

            # Extend future with last known regressor values
            for col in regressors:
                future[col] = df_fc[col].iloc[-1]

            forecast = model.predict(future)

            fig = plot_plotly(model, forecast)
            fig.update_layout(
                title=f"Forecast for {target_species} Resistance to {anti}",
                xaxis_title="Year",
                yaxis_title="Resistance Value (Log2)",
                template="plotly_white"
            )
            st.plotly_chart(fig)
            st.markdown(f"""
            - **Observation**: The forecast plot shows the predicted trend in **{anti}** resistance for _{target_species}_ over the next **{num_years}** years. The shaded area represents the uncertainty in the forecast. Rising trends in the forecast indicate potential increases in resistance, while stable or declining trends suggest sustained effectiveness.
            - **Implications**: Understanding future resistance trends helps inform empirical therapy decisions and guide AMR stewardship efforts.
            - **Recommendations**:
                - **AMR Stewardship**: Use forecasted trends to adjust empirical therapy protocols proactively.
                - **Policy**: Allocate resources based on predicted resistance patterns and invest in targeted surveillance.
                - **Stakeholder Engagement**: Collaborate with microbiologists and infectious disease specialists to interpret forecast data and update treatment guidelines.
            """)


# Prediction
def train_model(df, anti):
    #st.subheader(f"Make Prediction for {anti} Resistance")
    species = df['Species'].unique().tolist()
    selected_species = st.selectbox("Select Species", species)
    df_species = df[df["Species"] == selected_species].copy()
    outcome_col = f"{anti}_I"

    base_features = ['Study', 'Speciality', 'Source', 'In / Out Patient', 'Phenotype', 'Country', 'Year', 'Gender', 'Age Group']
    features = [f for f in base_features if f in df_species.columns]

    for col in features:
        if df_species[col].dtype == 'object':
            df_species[col] = df_species[col].fillna('Other').astype(str)
            df_species[col] = LabelEncoder().fit_transform(df_species[col])
        elif col == 'Year':
            df_species[col] = df_species[col].fillna(df_species[col].median())

    df_species = df_species.dropna(subset=['Species', outcome_col])

    resistance_types = df_species[outcome_col].unique()
    if df_species.empty or len(resistance_types) == 1:
        most_common = resistance_types[0] if len(resistance_types) == 1 else "No outcome"
        st.warning(f"_{selected_species}_ is **{most_common}** only to **{anti}**. No further training needed.")
        return

    if df_species[outcome_col].dtype == 'object':
        le_outcome = LabelEncoder()
        df_species['target_encoded'] = le_outcome.fit_transform(df_species[outcome_col])
        outcome_for_causal_model = 'target_encoded'
    else:
        outcome_for_causal_model = outcome_col

    treatments = ['Phenotype', 'Source', 'Country'] if all(t in features for t in ['Phenotype', 'Source', 'Country']) else ['Year']
    confounders = [c for c in base_features if c in features and c not in treatments]

    causal_effects = []
    for treatment in treatments:
        graph = nx.DiGraph()
        graph.add_edge(treatment, outcome_for_causal_model)
        for conf in confounders:
            graph.add_edge(conf, outcome_for_causal_model)
            graph.add_edge(conf, treatment)
        try:
            model = CausalModel(
                data=df_species,
                treatment=[treatment],
                outcome=outcome_for_causal_model,
                graph=graph
            )
            estimand = model.identify_effect()
            estimate = model.estimate_effect(
                estimand,
                method_name="backdoor.linear_regression",
                target_units="ate"
            )
            causal_effects.append({'Treatment': treatment, 'Effect': estimate.value})
        except Exception as e:
            st.warning(f"Error estimating causal effect for {treatment}: {e}")

    model_options = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42),
        'Support Vector Machine (SVM)': SVC(probability=True, random_state=42),
        'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
        'Extreme Gradient Boosting (XGBoost)': XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=42),
        'CatBoost Classifier': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
        'LightGBM Classifier': LGBMClassifier(random_state=42)
    }

    model_name = st.selectbox("Select Model", list(model_options.keys()))

    def train_causal_ml(df, species, antibiotic, model_name, features, outcome_col):
        df_species = df[df['Species'] == species].copy()
        le_outcome = LabelEncoder()
        df_species['target'] = le_outcome.fit_transform(df_species[outcome_col])

        for col in features:
            if df_species[col].dtype == 'object':
                df_species[col] = df_species[col].fillna('Other').astype(str)
                df_species[col] = LabelEncoder().fit_transform(df_species[col])
            elif col == 'Year':
                df_species[col] = df_species[col].fillna(df_species[col].median())

        X = df_species[features]
        y = df_species['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        clf = model_options[model_name]
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        st.subheader(f"{model_name} Performance")
        st.markdown(f"""
        - **Accuracy**: {accuracy_score(y_test, y_pred) * 100:.2f}%
        - **Precision**: {precision_score(y_test, y_pred, average='weighted') * 100:.2f}%
        - **Recall**: {recall_score(y_test, y_pred, average='weighted') * 100:.2f}%
        - **F1 Score**: {f1_score(y_test, y_pred, average='weighted') * 100:.2f}%
        """)

        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, x=le_outcome.classes_, y=le_outcome.classes_,
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           title=f"Confusion Matrix for {species} - {antibiotic}")
        st.plotly_chart(fig_cm)

        if hasattr(clf, 'feature_importances_'):
            fi_df = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_}).sort_values(by='Importance', ascending=False)
            fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                            title=f'Feature Importance: {species} Resistance to {antibiotic}')
            st.plotly_chart(fig_fi)
            st.markdown(f"""
            - **Feature Importance**: The bar chart above shows the importance of each feature in predicting _{species}_ resistance to **{antibiotic}**. Features with higher importance scores have a greater impact on the model's predictions.
            - **Implications**: Understanding feature importance helps identify key factors influencing resistance, guiding targeted interventions and policy decisions.
            """)

        if causal_effects:
            try:
                df_causal = pd.DataFrame(causal_effects)
                fig_ce = px.bar(df_causal, x='Effect', y='Treatment', orientation='h',
                                title=f'Causal Effects on {antibiotic} Resistance')
                fig_ce.update_layout(xaxis_title="Average Treatment Effect (log-odds)")
                st.plotly_chart(fig_ce)
                st.markdown(f"""
                - **Treatments and Confounders**:
                    - **Treatments**: `Phenotype`, `Source` and `Country` are used as primary treatments, as they likely influence resistance directly (Phenotype reflects resistance mechanisms, Source indicates infection context, and Country inculcates geographical factors).
                    - **Confounders**: Other factors (Study, Speciality, Year, Gender, Age Group, etc) are used to control for confounding effects as they attribute to healthcare practices, patient demographics, etc.
                    - **Rationale**: Phenotype captures resistance-related characteristics, and Source reflects clinical context while Country provides geographical information.
                - **Causal Effects**: The bar chart above shows the estimated causal effects of different treatments on _{species}_ resistance to **{antibiotic}**. Each treatment's effect is measured in log-odds, indicating how much the treatment influences the likelihood of resistance.
                - **Implications**: Understanding causal effects helps identify which factors significantly impact resistance, guiding targeted interventions and policy decisions.
                """)
            except Exception as e:
                st.warning(f"Error plotting causal effects: {e}")

        return clf

    if st.button("Train Model"):
        with st.spinner(f"Training model for _{selected_species}_ resistance to **{anti}**..."):
            clf = train_causal_ml(df, selected_species, anti, model_name, features, outcome_col)
            st.subheader("Download Trained Model")
            pickle_buffer = BytesIO()
            pickle.dump(clf, pickle_buffer)
            pickle_buffer.seek(0)
            st.download_button(
                label="Download Model",
                data=pickle_buffer,
                file_name=f"{model_name.replace(' ', '_')}_model.pkl",
                mime="application/octet-stream"
            )

# Make Prediction
def make_prediction(df, anti):
    st.subheader(f"Make Prediction for {anti} Resistance")

    # species selection 
    species = df['Species'].unique().tolist()
    selected_species = st.selectbox("Select Species", species)
    df_species = df[df["Species"] == selected_species].copy()

    # Column definitions
    anti_MIC = anti + "_I"
    base_cols = ['Study', 'Family', 'Country', 'Gender', 'Age Group',
                 'Speciality', 'Source', 'In / Out Patient', anti, 'Year']
    cols_to_use = base_cols + [anti_MIC]

    # Filter and prepare data
    df = df_species[cols_to_use].copy()
    # Check for data sufficiency before filling NaNs
    min_required_rows = 10  
    resistance_types = df[anti_MIC].unique()
   
    if df.shape[0] < min_required_rows or len(resistance_types) == 1:
        st.warning(f"""
    🚨 **Insufficient Data to Make Predictions**

    The selected _{selected_species}_ has **{resistance_types[0]}** resistance status to **{anti}** in the data.

    This data gap is more than a technical limitation—it reflects a broader challenge in antimicrobial resistance (AMR) surveillance.

    📊 **Why This Matters**:
    - Sparse data can obscure resistance trends and delay detection of emerging threats.
    - Prediction models depend on consistent, high-quality inputs to guide empirical therapy.
    - Data gaps weaken the foundation for evidence-based stewardship and policy decisions.

    🧬 **Call to Action**:
    - Strengthen AMR data collection across healthcare settings.
    - Standardize MIC reporting and phenotype classification.
    - Invest in regional surveillance systems to close the data gap.

    Let’s transform data scarcity into a catalyst for smarter stewardship and stronger policy.
    """)
        st.stop()

    # Handle missing values
    for col in cols_to_use:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Other').astype(str)
        elif col == 'Year':
            df[col] = df[col].fillna(df[col].median())

    # Map labels to numeric
    label_mapping = {'Resistant': 2, 'Intermediate': 1, 'Susceptible': 0}
    df[anti_MIC] = df[anti_MIC].map(label_mapping)

    # Drop rows with missing resistance labels
    df = df.dropna(subset=[anti_MIC])

    # Reindex class labels to start from 0
    unique_classes = sorted(df[anti_MIC].dropna().unique())
    class_remap = {old: new for new, old in enumerate(unique_classes)}
    df[anti_MIC] = df[anti_MIC].map(class_remap)

    # Split data
    X = df.drop(anti_MIC, axis=1)
    y = df[anti_MIC]

    # Check if y is empty after filtering
    num_classes = len(np.unique(y))

    if y.empty or num_classes < 2:
        st.warning(f"Not enough class diversity in resistance labels for _{selected_species}_ and *{anti}*. Cannot train model.")
        
        # Implications and call to action
        return st.markdown("""
        🚨 **Insufficient Data for Prediction**
        The selected combination of **{anti}** and _{selected_species}_ has not enough resistance labels in the data for making predictions.
        This data gap is more than a technical limitation—it reflects a broader challenge in antimicrobial resistance (AMR) surveillance.
        📊 **Why This Matters**:
        - Sparse data can obscure resistance trends and delay detection of emerging threats.
        - Prediction models depend on consistent, high-quality inputs to guide empirical therapy.
        - Data gaps weaken the foundation for evidence-based stewardship and policy decisions.
        🧬 **Call to Action**:  
        - Strengthen AMR data collection across healthcare settings.
        - Standardize MIC reporting and phenotype classification.
        - Invest in regional surveillance systems to close the data gap.
        Let’s transform data scarcity into a catalyst for smarter stewardship and stronger policy.
        """)
    

    # Initialize encoders
    encoders = {}
    categorical_cols = [col for col in X.columns if col != "Year"]

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Train model
    classifier = XGBClassifier(objective='multi:softmax',
                            num_class=num_classes,
                            eval_metric='mlogloss',
                            random_state=42)
    classifier.fit(X, y)

    # User input section (unchanged)
    year = st.number_input("Pick year of study", min_value=2000, max_value=2030, value=2010, step=1)
    study_selected = st.selectbox("Select the type of Study done: ", df['Study'].unique())
    family_selected = st.selectbox("Select the family that bacteria (organism) under study belong to: ", df['Family'].unique())
    country_selected = st.selectbox("Select country of study: ", df['Country'].unique())
    gender_selected = st.selectbox("Select gender: ", df['Gender'].unique())
    age_group_selected = st.selectbox("Select the age group: ", df['Age Group'].unique())
    speciality_selected = st.selectbox("Select the kind of Speciality: ", df['Speciality'].unique())
    source_selected = st.selectbox("Select Source: ", df['Source'].unique())
    patient_type_selected = st.selectbox("Select the patient type: ", df['In / Out Patient'].unique())
    anti_selected = st.selectbox("Select antibiotic MIC value: ", df[anti].unique())

    pred_btn = st.button("Make Prediction")

    if pred_btn:
        with st.spinner(f"Making prediction for {selected_species} resistance to {anti}..."):
            # Prepare single test input
            input_dict = {
                'Study': study_selected,
                'Family': family_selected,
                'Country': country_selected,
                'Gender': gender_selected,
                'Age Group': age_group_selected,
                'Speciality': speciality_selected,
                'Source': source_selected,
                'In / Out Patient': patient_type_selected,
                anti: anti_selected,
                'Year': year
            }

            # Encode input (except 'Year')
            input_encoded = []
            for col in X.columns:
                if col == 'Year':
                    input_encoded.append([year])
                else:
                    le = encoders[col]
                    try:
                        encoded_val = le.transform([input_dict[col]])[0]
                    except ValueError:
                        encoded_val = 0
                    input_encoded.append([encoded_val])

            X_test = np.array(input_encoded).T.astype(float)

            # Make prediction
            pred = classifier.predict(X_test)

            # Interpret prediction
            if pred[0] == 0:
                st.success(f"The Bacterial Species _{selected_species}_ would be **Susceptible** to the antibiotic **{anti}** under the selected conditions.")
                st.warning("**Disclaimer:** The predictions provided by this tool are intended for study purposes only. Please cross check with domain experts before making any decisions based on these predictions.")
            elif pred[0] == 1:
                st.info(f"The bacteria (organism) _{selected_species}_ would have **Intermediate Resistance** to the antibiotic **{anti}** under the selected conditions.")
                st.warning("**Disclaimer:** The predictions provided by this tool are intended for study purposes only. Please cross check with domain experts before making any decisions based on these predictions.")
            elif pred[0] == 2:
                st.error(f"The bacteria (organism) _{selected_species}_ would be **Resistant** to the antibiotic **{anti}** under the selected conditions.")
                st.warning("**Disclaimer:** The predictions provided by this tool are intended for study purposes only. Please cross check with domain experts before making any decisions based on these predictions.")

# Check Causal Effect
def check_causal_effect(df, anti):
    outcome_col = f'{anti}_I'

    # Fill all `NaN` values with mode in all columns that have NaN values
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].isnull().any():
            #df[col].fillna(df[col].mode()[0], inplace=True)
            df[col] = df[col].fillna(df[col].mode()[0]).astype(str)

    # Select species
    species = df['Species'].unique().tolist()
    # Select species
    selected_species = st.selectbox("Select Species", species, key ="species_select")
    df_species = df[df['Species'] == selected_species].copy()
    df_species[f'{anti}_MIC'] = df_species[anti]

    # Define features (non-gene columns)
    features = ['Study', 'Speciality', 'Source', 'In / Out Patient', 'Phenotype', 'Country', 'Year', 'Gender', 'Age Group', f'{anti}_MIC']
    # Filter features to those present in the dataset
    features = [f for f in features if f in df_species.columns]

    # Handle missing data
    categorical_cols = [col for col in features if col in ['Study', 'Speciality', 'Source', 'In / Out Patient', 'Phenotype', 'Country', 'Gender', 'Age Group', f'{anti}_MIC']]
    for col in categorical_cols:
        df_species[col] = df_species[col].fillna('Other').astype(str)

    # Drop rows with NaN in outcome or required columns
    required_cols = ['Species', outcome_col]
    df_species = df_species.dropna(subset=required_cols)

    resistance_types = df_species[outcome_col].unique()
    susceptible_value = resistance_types[0]

    # Check if dataset is empty
    if df_species.empty or len(df_species[outcome_col].unique()) == 1:
        st.warning(f"{selected_species} is {susceptible_value} only to {anti}. \nNo further training needed." )
        st.stop()
    
    # Encode categorical predictors
    for col in categorical_cols:
        le = LabelEncoder()
        df_species[col] = le.fit_transform(df_species[col])

    # Handle Year (numeric, impute with median if missing)
    if 'Year' in features:
        df_species['Year'] = df_species['Year'].fillna(df_species['Year'].median())

    # Generate causal graph for a single treatment
    def generate_causal_graph(outcome, treatment, confounders):
        graph = nx.DiGraph()
        graph.add_edge(treatment, outcome)
        for confounder in [c for c in confounders if c in features and c != treatment]:
            graph.add_edge(confounder, outcome)
            graph.add_edge(confounder, treatment)
        return graph
    
    # Define treatments and confounders
    default_treatments = ['Phenotype', 'Source', 'Country']
    selected_factors = st.multiselect(
        "Select Factors for Analysis",
        options=features,
        default=default_treatments
    )
    treatments = selected_factors
    confounders = [c for c in features if c in features and c not in treatments]

    # Compute causal effect
    causal_effects = []

    # Encode the outcome column if it's not already numerical
    if df_species[outcome_col].dtype == 'object':
        le_outcome = LabelEncoder()
        df_species['target_encoded'] = le_outcome.fit_transform(df_species[outcome_col])
        outcome_for_causal_model = 'target_encoded'
    else:
        outcome_for_causal_model = outcome_col

    # Identify columns to encode: treatments and confounders that are in features and are object type
    cols_to_encode = [col for col in treatments + confounders if col in features and df_species[col].dtype == 'object']

    for col in cols_to_encode:
        # Using LabelEncoder for simplicity for all categorical treatments and confounders
        le = LabelEncoder()
        df_species[col] = le.fit_transform(df_species[col].astype(str))
        # Explicitly convert to int to ensure numerical dtype
        df_species[col] = df_species[col].astype(int)

    for treatment in treatments:
        # Ensure the treatment column is explicitly an int type for the model
        if df_species[treatment].dtype != 'int':
            df_species[treatment] = df_species[treatment].astype(int)

        causal_graph = generate_causal_graph(outcome_for_causal_model, treatment, confounders)
        model = CausalModel(
            data=df_species,
            treatment=[treatment],
            outcome=outcome_for_causal_model,
            graph=causal_graph
        )
        identified_estimand = model.identify_effect()
        try:
            causal_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                target_units="ate"
            )

            # For linear regression, the causal effect value is directly available
            causal_effects.append({'Treatment': treatment, 'Effect': causal_estimate.value})
            #st.write(f"Successfully estimated causal effect for {treatment}. Effect Value: {causal_estimate.value}")

        except Exception as e:
            st.warning(f"Error estimating causal effect for {treatment}: {e}")

    if st.button("Show Causal Effects"):
        with st.spinner(f"Showing causal effects of _{selected_species}_ resistance to {anti} prediction..."):
            if not causal_effects:
                st.warning("No causal effects could be estimated. Please check the data and selected factors.")
            else:
                df_causal = pd.DataFrame(causal_effects)
                if df_causal.empty:
                    st.warning("No causal effects found for the selected factors.")
                else:
                    # Print Causal Effects Values
                    st.write("Causal Effects Values:")
                    st.dataframe(df_causal)
                    # Plot Causal Effects
                    fig_ce = px.bar(df_causal, x='Effect', y='Treatment', orientation='h',
                                    title=f'Causal Effects on {anti} Resistance for {selected_species}',
                                    labels={'Effect': 'Average Treatment Effect (log-odds)', 'Treatment': 'Treatment'})
                    fig_ce.update_layout(xaxis_title="Average Treatment Effect (log-odds)")
                    st.plotly_chart(fig_ce)
                    st.markdown(f"""
                    - **Treatments and Confounders**:
                        - The treatments selected are `{', '.join(treatments)}`. These factors are expected to influence resistance directly.
                        - The confounders that were used are `{', '.join(confounders)}`. These factors are used to control for confounding effects as they attribute to healthcare practices, patient demographics, etc.
                    - Experiment with different combinations of treatments and confounders to explore how they affect the causal estimates.
                    - **Causal Effects**: The bar chart above shows the estimated causal effects of different treatments on _{selected_species}_ resistance to **{anti}**. Each treatment's effect is measured in log-odds, indicating how much the treatment influences the likelihood of resistance.
                    - **Implications**: Understanding causal effects helps identify which factors significantly impact resistance, guiding targeted interventions and policy decisions.
                    """)

    

