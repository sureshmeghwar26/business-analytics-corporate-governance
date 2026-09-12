import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Initialize Raw Executive Interview Transcript Segments (Simulating Stage 1 Matrix)
raw_telemetry_data = {
    'Executive_ID': [f"Exec_{i}" for i in range(1, 9)], # 8 Senior ICT/Consultancy Executives interviewed
    'Industry_Sector': ['ICT', 'Consultancy', 'Consultancy', 'ICT', 'Consultancy', 'ICT', 'ICT', 'Consultancy'],
    'Core_Response_Transcript': [
        "We successfully aligned BA with our strategic goals, but high cost of the CDM layer model remains a data governance gate.",
        "Data privacy concerns and algorithmic bias are persistent issues preventing complete migration to real-time analytics pipelines.",
        "Integrating predictive analytics with our PESTEL strategy increased our competitive advantage under the VRIO framework.",
        "Our legacy infrastructure is consistently behind business development, causing major data integration problems across sectors.",
        "Establishing clear goals across all levels allowed us to improve organizational performance and decision accuracy using SPSS frameworks.",
        "High development costs, algorithm fairness issues, and data privacy limits remain fundamental bottlenecks to address.",
        "Real-time analysis enables faster decision-making, but proper information governance is required to minimize processing errors.",
        "Our investments in big data analytics led to significant productivity improvements once we resolved structural data quality issues."
    ],
    'Stated_BA_Maturity_Tier': ['High', 'Moderate', 'High', 'Low', 'High', 'Low', 'Moderate', 'High']
}

df_transcript = pd.DataFrame(raw_telemetry_data)
print(f"--- Stage 1: Qualitative Text Matrix Ingested ({df_transcript.shape[0]} Executive Records) ---")

# 2. Stage 2: Automated Thematic Open & Axial Coding Search Engine
# Define high-impact thematic keyword flags derived from IRJMETS 2024 literature
thematic_keywords = {
    'Strategic_Alignment': ['align', 'strategy', 'goal', 'vrio', 'performance'],
    'Data_Governance_Privacy': ['privacy', 'governance', 'quality', 'cdm', 'administration'],
    'Technical_Barriers': ['bias', 'integration', 'algorithm', 'cost', 'infrastructure']
}

print("\n--- Running Two-Stage Explorative Deduction and Axial Representation Matrix ---")

# Execute row-by-row string parsing to flag explicit keyword themes
for theme, keywords in thematic_keywords.items():
    df_transcript[theme] = df_transcript['Core_Response_Transcript'].str.lower().apply(
        lambda text: 1 if any(kw in text for kw in keywords) else 0
    )

# Display the clean structural coding matrix
print(df_transcript[['Executive_ID', 'Stated_BA_Maturity_Tier', 'Strategic_Alignment', 'Data_Governance_Privacy', 'Technical_Barriers']])

# 3. Maturity Stratification Cross-Tabulation Analysis
print("\n--- Structural Cross-Tabulation Matrix: Maturity Tier vs Identified Technical Barriers ---")
cross_tab = pd.crosstab(df_transcript['Stated_BA_Maturity_Tier'], df_transcript['Technical_Barriers'])
print(cross_tab)

# 4. Generate Corporate Performance Analytics Visualizations for Reviewers
summary_metrics = df_transcript.groupby('Stated_BA_Maturity_Tier')[['Strategic_Alignment', 'Data_Governance_Privacy', 'Technical_Barriers']].sum()

plt.figure(figsize=(10, 5))
summary_metrics.plot(kind='bar', stacked=True, color=['#2ca02c', '#1f77b4', '#d62728'], figsize=(9, 5))
plt.title('Thematic Profile Analysis: Identified Themes across Corporate BA Maturity Tiers', fontsize=11)
plt.xlabel('Organizational Analytics Maturity Level')
plt.ylabel('Aggregated Code Frequencies (Thematic Transcripts)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.legend(['Strategic Alignment', 'Data Governance & Privacy', 'Technical & Algorithmic Barriers'])
plt.tight_layout()
plt.savefig('irjmets_thematic_matrix_output.png')
print("\n[Pipeline Complete]: Thematic reporting chart exported as 'irjmets_thematic_matrix_output.png'.")
