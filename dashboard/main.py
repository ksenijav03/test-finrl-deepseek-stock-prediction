import streamlit as st
import pandas as pd
import altair as alt

# Page configuration
st.set_page_config(page_title="Nvidia Stock Predictions", layout="wide")

# Title
st.markdown(
    "<h1 style='text-align: center; font-weight: bold;'>Nvidia Stock (near) real-time predictions with FinRL and DeepSeek</h1>",
    unsafe_allow_html=True
)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('../finrl/results.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').reset_index()
    
    # Calculate percentage difference between Agent 2 and Agent 1
    df['% Diff'] = 100 * (df["A2C Agent2"] - df["A2C Agent1"]) / df["A2C Agent1"]

    return df

df = load_data()

# Melt the dataframe for Altair
df_melted = df.melt(
    id_vars=['date', '% Diff'], 
    var_name='Agent', 
    value_name='Portfolio Value'
)

# Define CSS colors for each line
custom_colors = {
    "A2C Agent1": "#CD5C5C",
    "A2C Agent2": "#FF8C00",
    "Mean Var": "#5F9EA0",
    "djia": "#9ACD32"
}

# Conditional tooltip: show % difference only for A2C Agent 2
df_melted['Tooltip Diff'] = df_melted.apply(
    lambda row: f"{row['% Diff']:.2f}%" if row['Agent'] == 'A2C Agent2' else "-",
    axis=1
)

# Altair chart
chart = alt.Chart(df_melted).mark_line().encode(
    x=alt.X('date:T', title='Date'),
    y=alt.Y('Portfolio Value:Q', title='Portfolio Value'),
    color=alt.Color('Agent:N', scale=alt.Scale(domain=list(custom_colors.keys()), range=list(custom_colors.values()))),
    tooltip=[
        alt.Tooltip('date:T', title='Date'),
        alt.Tooltip('Agent:N', title='Agent'),
        alt.Tooltip('Portfolio Value:Q', title='Value', format=".2f"),
        alt.Tooltip('Tooltip Diff:N', title='% Diff vs Agent 1')
    ]
).interactive().properties(
    width=1000,
    height=500,
    title='Portfolio Performance Over Time'
)

# Display chart
st.altair_chart(chart, use_container_width=True)
