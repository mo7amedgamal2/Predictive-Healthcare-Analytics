"""
Dashboard Helper Module
Loads dataset and generates Plotly charts for dashboard and visualizations
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Get base directory relative to this file (same directory as app.py)
BASE_DIR = Path(__file__).resolve().parent

# Dataset path - use relative path from base directory
DATASET_PATH = BASE_DIR / "data" / "oral_cancer_prediction_dataset.csv"

# Numeric columns for type coercion
_NUMERIC_COLS = [
    'Age',
    'Cancer Stage',
    'Year_of_Diagnosis',
    'Tumor Size (cm)',
    'Predicted_LOS(Days)',
    'Predicted_Recovery(Days)',
    'Survival Rate (5-Year, %)',
    'Cost of Treatment (USD)',
    'Economic Burden (Lost Workdays per Year)',
    'Cancer_Label',
]

_STR_COLS = [
    'Gender',
    'Country',
    'Treatment Type',
    'Oral Cancer (Diagnosis)',
    'Age_Group',
]

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to correct types"""
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in _STR_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    
    # Create Cancer_Label if not exists
    if 'Cancer_Label' not in df.columns and 'Oral Cancer (Diagnosis)' in df.columns:
        df['Cancer_Label'] = df['Oral Cancer (Diagnosis)'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    # Create Age_Group if not exists
    if 'Age_Group' not in df.columns and 'Age' in df.columns:
        df['Age_Group'] = pd.cut(
            df['Age'], bins=[0, 19, 39, 59, 79, 200],
            labels=['0-19', '20-39', '40-59', '60-79', '80+']
        ).astype(str)
    
    return df

def load_dashboard_data():
    """Load and prepare dataset for dashboard"""
    # Try multiple possible paths - deployment path first, then development paths
    possible_paths = [
        BASE_DIR / "data" / "oral_cancer_prediction_dataset.csv",  # Deployment path
        BASE_DIR / "archive" / "Milestone -2" / "oral_cancer_prediction_dataset (Before Milestone-2).csv",  # Archive path
        BASE_DIR / "archive" / "Milestone -2" / "oral_cancer_prediction_dataset(Pre-model)not_scale.csv",  # Alternative archive path
    ]
    
    for dataset_path in possible_paths:
        try:
            if dataset_path.exists():
                logger.info(f"Loading dataset from: {dataset_path}")
                df = pd.read_csv(dataset_path)
                df = _coerce_types(df)
                logger.info(f"Dataset loaded: {df.shape}")
                return df
        except Exception as e:
            logger.warning(f"Error loading from {dataset_path}: {e}")
            continue
    
    logger.warning(f"Dataset not found in any of the expected locations, using fallback")
    return _generate_fallback_data()

def _generate_fallback_data():
    """Generate fallback data if dataset not found"""
    logger.warning("Generating fallback dataset")
    _raw = {
        'Age': np.random.randint(20, 80, 1000),
        'Gender': np.random.choice(['Male', 'Female'], 1000),
        'Country': np.random.choice(['USA', 'Canada', 'Mexico', 'UK', 'Germany'], 1000),
        'Cancer Stage': np.random.randint(0, 5, 1000),
        'Treatment Type': np.random.choice(['Surgery', 'Radiation', 'Chemotherapy', 'No Treatment', 'Targeted Therapy'], 1000),
        'Year_of_Diagnosis': np.random.randint(2018, 2025, 1000),
        'Tumor Size (cm)': np.random.rand(1000) * 5,
        'Oral Cancer (Diagnosis)': np.random.choice(['Yes', 'No'], 1000),
        'Predicted_LOS(Days)': np.random.randint(1, 20, 1000),
        'Predicted_Recovery(Days)': np.random.randint(10, 200, 1000),
        'Survival Rate (5-Year, %)': np.random.rand(1000) * 100,
        'Cost of Treatment (USD)': np.random.rand(1000) * 100000,
        'Economic Burden (Lost Workdays per Year)': np.random.rand(1000) * 150,
    }
    df = pd.DataFrame(_raw)
    df['Age_Group'] = pd.cut(
        df['Age'], bins=[0, 19, 39, 59, 79, 200],
        labels=['0-19', '20-39', '40-59', '60-79', '80+']
    ).astype(str)
    df['Cancer_Label'] = df['Oral Cancer (Diagnosis)'].map({'Yes': 1, 'No': 0}).astype(int)
    return _coerce_types(df)

def filter_data(df, selected_age_groups=None, selected_genders=None, selected_countries=None,
                selected_stages=None, selected_treatments=None, selected_years=None):
    """Filter dataset based on selected criteria"""
    df = df.copy()
    
    if selected_age_groups:
        df = df[df['Age_Group'].isin(selected_age_groups)]
    if selected_genders:
        df = df[df['Gender'].isin(selected_genders)]
    if selected_countries:
        df = df[df['Country'].isin(selected_countries)]
    if selected_stages:
        df['Cancer Stage'] = pd.to_numeric(df['Cancer Stage'], errors='coerce').astype('Int64')
        df = df[df['Cancer Stage'].isin(pd.Series(selected_stages).astype('Int64'))]
    if selected_treatments:
        df = df[df['Treatment Type'].isin(selected_treatments)]
    if selected_years and len(selected_years) == 2:
        yrs = pd.to_numeric(df['Year_of_Diagnosis'], errors='coerce')
        df = df[(yrs >= selected_years[0]) & (yrs <= selected_years[1])]
    
    return df

def get_dashboard_charts(df):
    """Generate all dashboard charts from filtered data"""
    charts = {}
    
    # Age Distribution
    if not df.empty:
        fig = px.histogram(df, x="Age", color="Oral Cancer (Diagnosis)",
                          barmode="overlay", nbins=30,
                          title="Age Distribution by Cancer Diagnosis")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['age_distribution'] = fig.to_json()
    else:
        charts['age_distribution'] = _empty_chart("Age Distribution (No data)")
    
    # Gender Distribution (Cancer Diagnosed Count)
    if not df.empty:
        counts = df["Oral Cancer (Diagnosis)"].value_counts().reset_index(name="Count")
        counts.columns = ["Oral Cancer (Diagnosis)", "Count"]
        fig = px.bar(counts, x="Oral Cancer (Diagnosis)", y="Count",
                     title="Count of Cancer Diagnosed",
                     color="Oral Cancer (Diagnosis)",
                     hover_data=["Count"])
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['gender_distribution'] = fig.to_json()
    else:
        charts['gender_distribution'] = _empty_chart("Count of Cancer Diagnosed (No data)")
    
    # Country Distribution
    if not df.empty:
        country_counts = df.groupby("Country")["Oral Cancer (Diagnosis)"].value_counts().unstack(fill_value=0).reset_index()
        for col in ['No', 'Yes']:
            if col not in country_counts.columns:
                country_counts[col] = 0
        fig = px.bar(country_counts, x="Country", y=["No", "Yes"],
                     title="Cancer Diagnosis by Country",
                     labels={"value": "Count", "variable": "Oral Cancer (Diagnosis)", "Country": "Country"},
                     barmode="group")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['country_distribution'] = fig.to_json()
    else:
        charts['country_distribution'] = _empty_chart("Cancer Diagnosis by Country (No data)")
    
    # Tumor Size vs Cancer Stage
    if not df.empty:
        fig = px.box(df, x="Cancer Stage", y="Tumor Size (cm)", color="Oral Cancer (Diagnosis)",
                     title="Tumor Size vs Cancer Stage (by Diagnosis)")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['tumor_size_stage'] = fig.to_json()
    else:
        charts['tumor_size_stage'] = _empty_chart("Tumor Size vs Cancer Stage (No data)")
    
    # Stage-Diagnosis Heatmap
    if not df.empty:
        cross_tab = pd.crosstab(df['Cancer Stage'], df['Oral Cancer (Diagnosis)'])
        fig = go.Figure(data=go.Heatmap(
            z=cross_tab.values,
            x=list(cross_tab.columns.astype(str)),
            y=list(cross_tab.index.astype(str)),
            colorscale='Reds',
            text=cross_tab.values,
            texttemplate="%{text}",
            hoverongaps=False
        ))
        fig.update_layout(
            title="Cancer Stage vs. Cancer Diagnosis Heatmap",
            xaxis_title="Cancer Diagnosis",
            yaxis_title="Cancer Stage",
            title_x=0.5,
            template="plotly_white"
        )
        charts['stage_diagnosis_heatmap'] = fig.to_json()
    else:
        charts['stage_diagnosis_heatmap'] = _empty_chart("Cancer Stage vs. Cancer Diagnosis Heatmap (No data)")
    
    # Treatment Stage Bar Chart
    if not df.empty:
        counts = df.groupby(['Cancer Stage', 'Treatment Type']).size().reset_index(name='Count')
        fig = px.bar(counts, x='Cancer Stage', y='Count', color='Treatment Type',
                     barmode='group', title='Distribution of Treatment Type by Cancer Stage')
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['treatment_stage'] = fig.to_json()
    else:
        charts['treatment_stage'] = _empty_chart("Distribution of Treatment Type by Cancer Stage (No data)")
    
    # LOS by Treatment
    if not df.empty:
        fig = px.box(df, x="Treatment Type", y="Predicted_LOS(Days)", color="Treatment Type",
                     title="Predicted Length of Stay (Days) by Treatment Type")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['los_treatment'] = fig.to_json()
    else:
        charts['los_treatment'] = _empty_chart("Predicted Length of Stay (No data)")
    
    # Recovery by Treatment
    if not df.empty:
        fig = px.box(df, x="Treatment Type", y="Predicted_Recovery(Days)", color="Treatment Type",
                     title="Predicted Recovery Time (Days) by Treatment Type")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['recovery_treatment'] = fig.to_json()
    else:
        charts['recovery_treatment'] = _empty_chart("Predicted Recovery Time (No data)")
    
    # Recovery vs Survival Scatter
    if not df.empty:
        fig = px.scatter(df, x="Predicted_Recovery(Days)", y="Survival Rate (5-Year, %)",
                         color="Treatment Type",
                         hover_data=["Cancer Stage", "Tumor Size (cm)"],
                         title="Predicted Recovery Time vs. Survival Rate by Treatment Type",
                         opacity=0.6)
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['recovery_survival'] = fig.to_json()
    else:
        charts['recovery_survival'] = _empty_chart("Predicted Recovery Time vs. Survival Rate (No data)")
    
    # Cost vs Economic Burden Scatter
    if not df.empty:
        fig = px.scatter(df, x="Cost of Treatment (USD)", y="Economic Burden (Lost Workdays per Year)",
                         color="Oral Cancer (Diagnosis)",
                         hover_data=["Cancer Stage", "Age"],
                         title="Cost of Treatment vs. Economic Burden by Cancer Diagnosis")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['cost_economic'] = fig.to_json()
    else:
        charts['cost_economic'] = _empty_chart("Cost of Treatment vs. Economic Burden (No data)")
    
    # Cost by Stage
    if not df.empty:
        fig = px.box(df, x="Cancer Stage", y="Cost of Treatment (USD)",
                     color="Cancer Stage",
                     title="Cost of Treatment by Cancer Stage")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['cost_stage'] = fig.to_json()
    else:
        charts['cost_stage'] = _empty_chart("Cost of Treatment by Cancer Stage (No data)")
    
    # Economic Burden by Stage
    if not df.empty:
        fig = px.box(df, x="Cancer Stage", y="Economic Burden (Lost Workdays per Year)",
                     color="Cancer Stage",
                     title="Economic Burden by Cancer Stage")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['economic_stage'] = fig.to_json()
    else:
        charts['economic_stage'] = _empty_chart("Economic Burden by Cancer Stage (No data)")
    
    return charts

def get_visualization_charts(df):
    """Generate all EDA visualization charts"""
    charts = {}
    
    # 1. Count of Cancer Diagnosed
    if not df.empty:
        diagnosis_counts = df["Oral Cancer (Diagnosis)"].value_counts().reset_index(name="Count")
        diagnosis_counts.columns = ["Oral Cancer (Diagnosis)", "Count"]
        fig = px.bar(diagnosis_counts, x="Oral Cancer (Diagnosis)", y="Count",
                     title="Count of Cancer Diagnosed",
                     color="Oral Cancer (Diagnosis)",
                     color_discrete_sequence=px.colors.qualitative.Plotly,
                     hover_data=["Count"])
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['diagnosis_count'] = fig.to_json()
    
    # 2. Age Group vs Cancer Diagnosis Heatmap
    if not df.empty:
        cross = pd.crosstab(df["Age_Group"], df["Oral Cancer (Diagnosis)"])
        fig = px.imshow(
            cross,
            color_continuous_scale="Reds",
            text_auto=True,
            title="Age Group vs Cancer Diagnosis (Heatmap)"
        )
        fig.update_layout(
            xaxis_title="Cancer Diagnosis",
            yaxis_title="Age Group",
            title_x=0.5,
            font=dict(size=13),
            template="plotly_white"
        )
        charts['age_group_heatmap'] = fig.to_json()
    
    # 3. Cancer Stage vs Age Heatmap
    if not df.empty:
        cross_tab = pd.crosstab(df['Age'], df['Cancer Stage'])
        fig = go.Figure(data=go.Heatmap(
            z=cross_tab.values,
            x=cross_tab.columns,
            y=cross_tab.index,
            colorscale='Reds'
        ))
        fig.update_layout(
            title="Cancer stage vs Age Heatmap",
            xaxis_title="Cancer Stage",
            yaxis_title="Age",
            title_x=0.5,
            font=dict(size=13),
            template="plotly_white"
        )
        charts['stage_age_heatmap'] = fig.to_json()
    
    # 4. Cancer Stage vs Diagnosis Heatmap
    if not df.empty:
        cross_tab_stage_diagnosis = pd.crosstab(df['Cancer Stage'], df['Oral Cancer (Diagnosis)'])
        fig = go.Figure(data=go.Heatmap(
            z=cross_tab_stage_diagnosis.values,
            x=cross_tab_stage_diagnosis.columns,
            y=cross_tab_stage_diagnosis.index,
            colorscale='Reds',
            text=cross_tab_stage_diagnosis.values,
            texttemplate="%{text}",
            textfont={"size":10}
        ))
        fig.update_layout(
            title="Cancer Stage vs. Cancer Diagnosis Heatmap",
            xaxis_title="Cancer Diagnosis",
            yaxis_title="Cancer Stage",
            title_x=0.5,
            font=dict(size=13),
            template="plotly_white"
        )
        charts['stage_diagnosis_heatmap_viz'] = fig.to_json()
    
    # 5. Tumor Size Distribution by Cancer Stage and Diagnosis
    if not df.empty:
        fig = px.box(df, x="Cancer Stage", y="Tumor Size (cm)", color="Cancer_Label",
                     title="Tumor Size Distribution by Cancer Stage and Diagnosis")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['tumor_size_distribution'] = fig.to_json()
    
    # 6. Treatment Distribution by Cancer Diagnosis
    if not df.empty:
        treatment_crosstab = pd.crosstab(df["Treatment Type"], df["Oral Cancer (Diagnosis)"]).reset_index()
        fig = px.bar(treatment_crosstab, x="Treatment Type", y=["No", "Yes"],
                     title="Treatment Distribution by Cancer Diagnosis",
                     labels={"value": "Count", "variable": "Oral Cancer (Diagnosis)"},
                     barmode="group",
                     hover_data={"value": True, "variable": True})
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['treatment_diagnosis'] = fig.to_json()
    
    # 7. Treatment Type by Cancer Stage
    if not df.empty:
        treatment_stage_counts = df.groupby(['Cancer Stage', 'Treatment Type']).size().reset_index(name='Count')
        fig = px.bar(treatment_stage_counts,
                     x='Cancer Stage',
                     y='Count',
                     color='Treatment Type',
                     barmode='group',
                     title='Distribution of Treatment Type by Cancer Stage',
                     labels={'Cancer Stage': 'Cancer Stage', 'Count': 'Number of Patients', 'Treatment Type': 'Treatment Type'},
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['treatment_stage_viz'] = fig.to_json()
    
    # 8. Cancer Diagnosis Over the Years
    if not df.empty:
        yearly = df.groupby("Year_of_Diagnosis")["Oral Cancer (Diagnosis)"].value_counts().unstack().reset_index()
        fig = px.bar(yearly, x="Year_of_Diagnosis", y=["No", "Yes"],
                     title="Cancer Diagnosis Over the Years",
                     labels={"value": "Count", "variable": "Oral Cancer (Diagnosis)", "Year_of_Diagnosis": "Year of Diagnosis"},
                     barmode="group",
                     hover_data={"value": True, "variable": True, "Year_of_Diagnosis": True})
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['diagnosis_years'] = fig.to_json()
    
    # 9. Cancer Diagnosis by Country
    if not df.empty:
        country = df.groupby("Country")["Oral Cancer (Diagnosis)"].value_counts().unstack().reset_index()
        fig = px.bar(country, x="Country", y=["No", "Yes"],
                     title="Cancer Diagnosis by Country",
                     labels={"value": "Count", "variable": "Oral Cancer (Diagnosis)", "Country": "Country"},
                     barmode="group",
                     hover_data={"value": True, "variable": True, "Country": True})
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['diagnosis_country'] = fig.to_json()
    
    # 10. Cost vs Economic Burden Scatter
    if not df.empty:
        fig = px.scatter(df, x="Cost of Treatment (USD)",
                         y="Economic Burden (Lost Workdays per Year)",
                         color="Cancer_Label",
                         size="Tumor Size (cm)",
                         hover_data=["Age", "Cancer Stage"],
                         title="Cost vs Economic Burden (with Tumor Size & Stage)")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['cost_economic_viz'] = fig.to_json()
    
    # 11. LOS by Treatment Type
    if not df.empty:
        fig = px.box(df, x="Treatment Type", y="Predicted_LOS(Days)", color="Treatment Type",
                     title="Predicted Length of Stay (Days) by Treatment Type")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['los_treatment_viz'] = fig.to_json()
    
    # 12. Recovery Time by Cancer Stage
    if not df.empty:
        fig = px.box(df, x="Cancer Stage", y="Predicted_Recovery(Days)",
                     title="Predicted Recovery Time (Days) by Cancer Stage",
                     color="Cancer Stage")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['recovery_stage'] = fig.to_json()
    
    # 13. Recovery Time by Treatment
    if not df.empty:
        fig = px.box(df, x="Treatment Type", y="Predicted_Recovery(Days)", color="Treatment Type",
                     title="Predicted Recovery Time (Days) by Treatment Type")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['recovery_treatment_viz'] = fig.to_json()
    
    # 14. Recovery vs Survival Rate
    if not df.empty:
        fig = px.scatter(df, x="Predicted_Recovery(Days)", y="Survival Rate (5-Year, %)",
                         color="Treatment Type",
                         hover_data=["Cancer Stage", "Tumor Size (cm)"],
                         title="Predicted Recovery Time vs. Survival Rate by Treatment Type")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['recovery_survival_viz'] = fig.to_json()
    
    # 15. Cost by Cancer Stage
    if not df.empty:
        fig = px.box(df, x="Cancer Stage", y="Cost of Treatment (USD)",
                     color="Cancer Stage",
                     title="Cost of Treatment by Cancer Stage")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['cost_stage_viz'] = fig.to_json()
    
    # 16. Economic Burden by Cancer Stage
    if not df.empty:
        fig = px.box(df, x="Cancer Stage", y="Economic Burden (Lost Workdays per Year)",
                     color="Cancer Stage",
                     title="Economic Burden by Cancer Stage")
        fig.update_layout(template="plotly_white", title_x=0.5)
        charts['economic_stage_viz'] = fig.to_json()
    
    # 17. Cost vs Economic Burden for Diagnosed Only
    if not df.empty:
        filtered_data = df[df['Oral Cancer (Diagnosis)'] == 'Yes']
        if not filtered_data.empty:
            fig = px.scatter(filtered_data, x="Cost of Treatment (USD)", y="Economic Burden (Lost Workdays per Year)",
                             title="Cost of Treatment vs. Economic Burden for Diagnosed Individuals")
            fig.update_layout(template="plotly_white", title_x=0.5)
            charts['cost_economic_diagnosed'] = fig.to_json()
    
    return charts

def _empty_chart(title):
    """Create an empty chart with a message"""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )]
    )
    return fig.to_json()

