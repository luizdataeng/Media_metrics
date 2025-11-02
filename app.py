"""
Streamlit Marketing Analytics Dashboard
Comprehensive dashboard for social media metrics analysis
"""

import warnings
import logging

# Suppress Streamlit ScriptRunContext warnings
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from data_loader import MarketingDataLoader

# Page configuration
st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(csv_file):
    """Load and cache the marketing data"""
    return MarketingDataLoader(csv_file)

# Load data
csv_file = "Final Dataset - Social Media Metrics Practice.csv"
data_loader = load_data(csv_file)
df = data_loader.df

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Overview", "📈 Trend Analysis", "📅 Monthly KPI Summary", 
     "🎯 Campaign Performance", "📱 Ad Performance", "🎬 Format Comparison"]
)

# Global Filters in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

# Date range filter
date_range = st.sidebar.date_input(
    "Date Range",
    value=(df['Reporting starts'].min().date(), df['Reporting ends'].max().date()),
    min_value=df['Reporting starts'].min().date(),
    max_value=df['Reporting ends'].max().date()
)

# Format filter
formats = ['All'] + sorted(df['Format'].unique().tolist())
selected_format = st.sidebar.selectbox("Format", formats)

# Campaign filter
campaigns = ['All'] + sorted(df['Campaign'].unique().tolist())
selected_campaign = st.sidebar.selectbox("Campaign", campaigns)

# Apply filters
filtered_df = df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['Reporting starts'].dt.date >= date_range[0]) &
        (filtered_df['Reporting ends'].dt.date <= date_range[1])
    ]

if selected_format != 'All':
    filtered_df = filtered_df[filtered_df['Format'] == selected_format]

if selected_campaign != 'All':
    filtered_df = filtered_df[filtered_df['Campaign'] == selected_campaign]

# Overview Page
if page == "🏠 Overview":
    st.title("🏠 Marketing Analytics Dashboard - Overview")
    
    # Get basic stats
    stats = data_loader.get_basic_stats()
    
    # Dataset Overview
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info(f"**Total Rows:** {stats['total_rows']:,}")
    
    with col2:
        st.info(f"**Date Range:** {stats['date_start'].strftime('%Y-%m-%d')} to {stats['date_end'].strftime('%Y-%m-%d')}")
    
    with col3:
        st.info(f"**Total Ads:** {stats['total_ads']}")
    
    with col4:
        st.info(f"**Unique Months:** {stats['unique_months']}")
    
    # Monthly Performance Summary
    st.header("📅 Monthly Performance Summary")
    monthly_kpis = data_loader.get_monthly_kpis()
    
    # Mini charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_spend = px.bar(
            monthly_kpis,
            x='Month_Display',
            y='Amount spent (CAD)',
            title='Spend by Month',
            color='Amount spent (CAD)',
            color_continuous_scale='Blues'
        )
        fig_spend.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_spend, use_container_width=True)
    
    with col2:
        fig_ctr = px.line(
            monthly_kpis,
            x='Month_Display',
            y='CTR_Calculated',
            title='CTR Trend by Month',
            markers=True
        )
        fig_ctr.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_ctr, use_container_width=True)
    
    # Format Distribution
    st.header("🎬 Format Distribution")
    format_dist = pd.DataFrame(list(stats['format_distribution'].items()), columns=['Format', 'Count'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            format_dist,
            values='Count',
            names='Format',
            title='Format Distribution'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(
            format_dist,
            x='Format',
            y='Count',
            title='Format Count',
            color='Format',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# Trend Analysis Page
elif page == "📈 Trend Analysis":
    st.title("📈 Trend Analysis")
    
    daily_data = data_loader.get_daily_data()
    
    # Impressions vs Spend
    st.header("📊 Impressions vs Spend")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['Reporting starts'],
            y=daily_data['Impressions'],
            name="Impressions",
            line=dict(color='#1f77b4', width=2),
            mode='lines+markers'
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=daily_data['Reporting starts'],
            y=daily_data['Amount spent (CAD)'],
            name="Spend (CAD)",
            line=dict(color='#ff7f0e', width=2),
            mode='lines+markers'
        ),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Impressions", secondary_y=False)
    fig.update_yaxes(title_text="Spend (CAD)", secondary_y=True)
    fig.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Clicks vs Spend
    st.header("🖱️ Clicks vs Spend")
    
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig2.add_trace(
        go.Scatter(
            x=daily_data['Reporting starts'],
            y=daily_data['Link clicks'],
            name="Clicks",
            line=dict(color='#2ca02c', width=2),
            mode='lines+markers'
        ),
        secondary_y=False,
    )
    
    fig2.add_trace(
        go.Scatter(
            x=daily_data['Reporting starts'],
            y=daily_data['Amount spent (CAD)'],
            name="Spend (CAD)",
            line=dict(color='#d62728', width=2),
            mode='lines+markers'
        ),
        secondary_y=True,
    )
    
    fig2.update_xaxes(title_text="Date")
    fig2.update_yaxes(title_text="Clicks", secondary_y=False)
    fig2.update_yaxes(title_text="Spend (CAD)", secondary_y=True)
    fig2.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig2, use_container_width=True)
    
    # Impressions Over Time
    st.header("👁️ Impressions Over Time")
    
    fig3 = px.area(
        daily_data,
        x='Reporting starts',
        y='Impressions',
        title='Impressions Trend',
        labels={'Impressions': 'Impressions', 'Reporting starts': 'Date'}
    )
    fig3.update_traces(fill='tozeroy', line=dict(width=2))
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)
    
    # CTR Over Time
    st.header("📈 CTR Over Time")
    
    monthly_ctr = data_loader.get_monthly_kpis()
    
    fig4 = px.line(
        monthly_ctr,
        x='Month_Display',
        y='CTR_Calculated',
        title='CTR Trend (Monthly)',
        markers=True,
        labels={'CTR_Calculated': 'CTR (%)', 'Month_Display': 'Month'}
    )
    fig4.update_traces(line=dict(width=3), marker=dict(size=10))
    fig4.add_scatter(
        x=monthly_ctr['Month_Display'],
        y=monthly_ctr['CTR_Calculated'],
        mode='markers+lines',
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)'
    )
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

# Monthly KPI Summary Page
elif page == "📅 Monthly KPI Summary":
    st.title("📅 Monthly KPI Summary")
    
    monthly_kpis = data_loader.get_monthly_kpis()
    
    # Display table
    st.header("📊 Monthly KPI Table")
    
    display_cols = {
        'Month_Display': 'Month',
        'Amount spent (CAD)': 'Spend (CAD)',
        'Reach': 'Reach',
        'Impressions': 'Impressions',
        'Link clicks': 'Clicks',
        'CTR_Calculated': 'CTR (%)',
        'CPM (cost per 1,000 impressions) (CAD)': 'CPM (CAD)',
        'CPC (cost per link click) (CAD)': 'CPC (CAD)'
    }
    
    display_df = monthly_kpis[list(display_cols.keys())].copy()
    display_df.columns = list(display_cols.values())
    
    st.dataframe(display_df.style.format({
        'Spend (CAD)': '${:,.2f}',
        'Reach': '{:,.0f}',
        'Impressions': '{:,.0f}',
        'Clicks': '{:,.0f}',
        'CTR (%)': '{:.2f}%',
        'CPM (CAD)': '${:.2f}',
        'CPC (CAD)': '${:.2f}'
    }), use_container_width=True, height=200)
    
    # Visualizations
    st.header("📈 Monthly Metrics Visualization")
    
    metrics_to_plot = ['Spend (CAD)', 'Reach', 'Impressions', 'Clicks', 'CTR (%)', 'CPM (CAD)', 'CPC (CAD)']
    
    selected_metric = st.selectbox("Select Metric to Visualize", metrics_to_plot)
    
    col_name = display_cols[list(display_cols.keys())[list(display_cols.values()).index(selected_metric)]] if selected_metric in display_cols.values() else None
    
    if col_name:
        original_col = [k for k, v in display_cols.items() if v == selected_metric][0]
        fig = px.bar(
            monthly_kpis,
            x='Month_Display',
            y=original_col,
            title=f'{selected_metric} by Month',
            color=original_col,
            color_continuous_scale='Blues',
            labels={original_col: selected_metric, 'Month_Display': 'Month'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # All metrics in one view
    st.header("📊 All Metrics Overview")
    
    tabs = st.tabs(["Spend & Reach", "Impressions & Clicks", "CTR, CPM & CPC"])
    
    with tabs[0]:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=monthly_kpis['Month_Display'], y=monthly_kpis['Amount spent (CAD)'], name="Spend (CAD)"),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=monthly_kpis['Month_Display'], y=monthly_kpis['Reach'], name="Reach", mode='lines+markers'),
            secondary_y=True
        )
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Spend (CAD)", secondary_y=False)
        fig.update_yaxes(title_text="Reach", secondary_y=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        fig = px.scatter(
            monthly_kpis,
            x='Impressions',
            y='Link clicks',
            size='Amount spent (CAD)',
            color='Month_Display',
            hover_data=['Month_Display'],
            title='Impressions vs Clicks (size = Spend)',
            labels={'Impressions': 'Impressions', 'Link clicks': 'Clicks'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_kpis['Month_Display'], y=monthly_kpis['CTR_Calculated'], name='CTR (%)', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=monthly_kpis['Month_Display'], y=monthly_kpis['CPM (cost per 1,000 impressions) (CAD)'], name='CPM (CAD)', mode='lines+markers', yaxis='y2'))
        fig.add_trace(go.Scatter(x=monthly_kpis['Month_Display'], y=monthly_kpis['CPC (cost per link click) (CAD)'], name='CPC (CAD)', mode='lines+markers', yaxis='y3'))
        
        fig.update_layout(
            height=400,
            xaxis=dict(title="Month"),
            yaxis=dict(title="CTR (%)"),
            yaxis2=dict(title="CPM (CAD)", overlaying='y', side='right'),
            yaxis3=dict(title="CPC (CAD)", overlaying='y', side='right', position=0.85)
        )
        st.plotly_chart(fig, use_container_width=True)

# Campaign Performance Page
elif page == "🎯 Campaign Performance":
    st.title("🎯 Campaign Performance")
    
    campaign_perf = data_loader.get_campaign_performance()
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_spend = st.number_input("Minimum Spend (CAD)", min_value=0.0, value=0.0, step=100.0)
    with col2:
        sort_by = st.selectbox("Sort By", ["Spend (CAD)", "CTR (%)", "CPM (CAD)", "CPC (CAD)", "Clicks"], index=0)
    
    # Apply filters
    filtered_campaigns = campaign_perf[campaign_perf['Amount spent (CAD)'] >= min_spend].copy()
    
    if sort_by == "Spend (CAD)":
        filtered_campaigns = filtered_campaigns.sort_values('Amount spent (CAD)', ascending=False)
    elif sort_by == "CTR (%)":
        filtered_campaigns = filtered_campaigns.sort_values('CTR (%)', ascending=False)
    elif sort_by == "CPM (CAD)":
        filtered_campaigns = filtered_campaigns.sort_values('CPM (cost per 1,000 impressions) (CAD)', ascending=False)
    elif sort_by == "CPC (CAD)":
        filtered_campaigns = filtered_campaigns.sort_values('CPC (cost per link click) (CAD)', ascending=False)
    elif sort_by == "Clicks":
        filtered_campaigns = filtered_campaigns.sort_values('Link clicks', ascending=False)
    
    # Table
    st.header("📊 Campaign Performance Table")
    
    display_campaign = filtered_campaigns[[
        'Campaign', 'Amount spent (CAD)', 'Impressions', 'Reach', 'Link clicks',
        'CTR (%)', 'CPM (cost per 1,000 impressions) (CAD)', 'CPC (cost per link click) (CAD)'
    ]].copy()
    
    display_campaign.columns = ['Campaign', 'Spend (CAD)', 'Impressions', 'Reach', 'Clicks',
                               'CTR (%)', 'CPM (CAD)', 'CPC (CAD)']
    
    st.dataframe(display_campaign.style.format({
        'Spend (CAD)': '${:,.2f}',
        'Impressions': '{:,.0f}',
        'Reach': '{:,.0f}',
        'Clicks': '{:,.0f}',
        'CTR (%)': '{:.2f}%',
        'CPM (CAD)': '${:.2f}',
        'CPC (CAD)': '${:.2f}'
    }), use_container_width=True, height=400)
    
    # Top Campaigns Visualizations
    st.header("🏆 Top Campaigns Analysis")
    
    top_n = st.slider("Number of Top Campaigns", 5, 20, 10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_ctr = filtered_campaigns.nlargest(top_n, 'CTR (%)')
        fig = px.bar(
            top_ctr,
            x='CTR (%)',
            y='Campaign',
            orientation='h',
            title=f'Top {top_n} Campaigns by CTR (%)',
            color='CTR (%)',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_cpm = filtered_campaigns.nsmallest(top_n, 'CPM (cost per 1,000 impressions) (CAD)')
        fig = px.bar(
            top_cpm,
            x='CPM (cost per 1,000 impressions) (CAD)',
            y='Campaign',
            orientation='h',
            title=f'Top {top_n} Campaigns by Lowest CPM (CAD) - Best Performing',
            color='CPM (cost per 1,000 impressions) (CAD)',
            color_continuous_scale='Greens_r'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter Plot
    st.header("📈 Spend vs Clicks Analysis")
    
    active_campaigns = filtered_campaigns[filtered_campaigns['Amount spent (CAD)'] > 0].copy()
    
    fig = px.scatter(
        active_campaigns,
        x='Amount spent (CAD)',
        y='Link clicks',
        size='Impressions',
        color='CTR (%)',
        hover_data=['Campaign'],
        title='Spend vs Clicks (size = Impressions, color = CTR)',
        color_continuous_scale='Viridis',
        labels={'Amount spent (CAD)': 'Spend (CAD)', 'Link clicks': 'Clicks'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export button
    if st.button("📥 Export Campaign Performance to CSV"):
        csv = display_campaign.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="campaign_performance.csv",
            mime="text/csv"
        )

# Ad Performance Page
elif page == "📱 Ad Performance":
    st.title("📱 Ad Performance")
    
    ad_perf = data_loader.get_ad_performance()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_spend = st.number_input("Minimum Spend (CAD)", min_value=0.0, value=0.0, step=50.0, key="ad_min_spend")
    with col2:
        format_filter = st.selectbox("Format Filter", ['All'] + sorted(ad_perf['Format'].unique().tolist()), key="ad_format")
    with col3:
        sort_by = st.selectbox("Sort By", ["Spend (CAD)", "CTR (%)", "CPM (CAD)", "CPC (CAD)", "Clicks"], index=0, key="ad_sort")
    
    # Apply filters
    filtered_ads = ad_perf[ad_perf['Amount spent (CAD)'] >= min_spend].copy()
    
    if format_filter != 'All':
        filtered_ads = filtered_ads[filtered_ads['Format'] == format_filter]
    
    if sort_by == "Spend (CAD)":
        filtered_ads = filtered_ads.sort_values('Amount spent (CAD)', ascending=False)
    elif sort_by == "CTR (%)":
        filtered_ads = filtered_ads.sort_values('CTR (%)', ascending=False)
    elif sort_by == "CPM (CAD)":
        filtered_ads = filtered_ads.sort_values('CPM (cost per 1,000 impressions) (CAD)', ascending=False)
    elif sort_by == "CPC (CAD)":
        filtered_ads = filtered_ads.sort_values('CPC (cost per link click) (CAD)', ascending=False)
    elif sort_by == "Clicks":
        filtered_ads = filtered_ads.sort_values('Link clicks', ascending=False)
    
    # Table
    st.header("📊 Ad Performance Table")
    
    display_ads = filtered_ads[[
        'Ad Name', 'Campaign', 'Format', 'Amount spent (CAD)', 'Impressions', 'Reach', 'Link clicks',
        'CTR (%)', 'CPM (cost per 1,000 impressions) (CAD)', 'CPC (cost per link click) (CAD)'
    ]].copy()
    
    display_ads.columns = ['Ad Name', 'Campaign', 'Format', 'Spend (CAD)', 'Impressions', 'Reach', 'Clicks',
                           'CTR (%)', 'CPM (CAD)', 'CPC (CAD)']
    
    st.dataframe(display_ads.style.format({
        'Spend (CAD)': '${:,.2f}',
        'Impressions': '{:,.0f}',
        'Reach': '{:,.0f}',
        'Clicks': '{:,.0f}',
        'CTR (%)': '{:.2f}%',
        'CPM (CAD)': '${:.2f}',
        'CPC (CAD)': '${:.2f}'
    }), use_container_width=True, height=400)
    
    # Top Ads Visualizations
    st.header("🏆 Top Ads Analysis")
    
    top_n = st.slider("Number of Top Ads", 5, 30, 15, key="ad_top_n")
    
    active_ads = filtered_ads[filtered_ads['Amount spent (CAD)'] > 0].copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_ctr_ads = active_ads.nlargest(top_n, 'CTR (%)')
        fig = px.bar(
            top_ctr_ads,
            x='CTR (%)',
            y='Ad Name',
            orientation='h',
            title=f'Top {top_n} Ads by CTR (%)',
            color='CTR (%)',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_cpc_ads = active_ads.nlargest(top_n, 'CPC (cost per link click) (CAD)')
        fig = px.bar(
            top_cpc_ads,
            x='CPC (cost per link click) (CAD)',
            y='Ad Name',
            orientation='h',
            title=f'Top {top_n} Ads by CPC (CAD)',
            color='CPC (cost per link click) (CAD)',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance by Format
    st.header("🎬 Performance by Format")
    
    format_perf = active_ads.groupby('Format').agg({
        'CTR (%)': 'mean',
        'CPM (cost per 1,000 impressions) (CAD)': 'mean',
        'CPC (cost per link click) (CAD)': 'mean'
    }).round(2).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=format_perf['Format'], y=format_perf['CTR (%)'], name='CTR (%)', marker_color='#1f77b4'))
    fig.add_trace(go.Bar(x=format_perf['Format'], y=format_perf['CPM (cost per 1,000 impressions) (CAD)'], name='CPM (CAD)', marker_color='#ff7f0e'))
    fig.add_trace(go.Bar(x=format_perf['Format'], y=format_perf['CPC (cost per link click) (CAD)'], name='CPC (CAD)', marker_color='#2ca02c'))
    
    fig.update_layout(
        barmode='group',
        height=400,
        title='Average Metrics by Format',
        xaxis_title='Format',
        yaxis_title='Value'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Export button
    if st.button("📥 Export Ad Performance to CSV"):
        csv = display_ads.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="ad_performance.csv",
            mime="text/csv"
        )

# Format Comparison Page
elif page == "🎬 Format Comparison":
    st.title("🎬 Format Comparison: Video vs Image")
    
    format_comp = data_loader.get_format_comparison()
    
    # Summary Table
    st.header("📊 Format Comparison Table")
    
    display_format = format_comp[[
        'Format',
        'Amount spent (CAD)_sum', 'Impressions_sum', 'Reach_sum', 'Link clicks_sum',
        'CTR_Calculated', 'CPM (cost per 1,000 impressions) (CAD)_mean',
        'CPC (cost per link click) (CAD)_mean'
    ]].copy()
    
    display_format.columns = ['Format', 'Total Spend (CAD)', 'Total Impressions', 'Total Reach', 'Total Clicks',
                             'CTR (%)', 'Avg CPM (CAD)', 'Avg CPC (CAD)']
    
    st.dataframe(display_format.style.format({
        'Total Spend (CAD)': '${:,.2f}',
        'Total Impressions': '{:,.0f}',
        'Total Reach': '{:,.0f}',
        'Total Clicks': '{:,.0f}',
        'CTR (%)': '{:.2f}%',
        'Avg CPM (CAD)': '${:.2f}',
        'Avg CPC (CAD)': '${:.2f}'
    }), use_container_width=True)
    
    # Visualizations
    st.header("📈 Format Metrics Visualization")
    
    tabs = st.tabs(["Total Metrics", "Average Metrics", "Normalized Comparison"])
    
    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(
                format_comp,
                x='Format',
                y='Amount spent (CAD)_sum',
                title='Total Spend by Format',
                color='Format',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                format_comp,
                x='Format',
                y='Impressions_sum',
                title='Total Impressions by Format',
                color='Format',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.bar(
                format_comp,
                x='Format',
                y='Link clicks_sum',
                title='Total Clicks by Format',
                color='Format',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(
                format_comp,
                x='Format',
                y='CTR_Calculated',
                title='Average CTR by Format',
                color='Format',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                format_comp,
                x='Format',
                y='CPM (cost per 1,000 impressions) (CAD)_mean',
                title='Average CPM by Format',
                color='Format',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.bar(
                format_comp,
                x='Format',
                y='CPC (cost per link click) (CAD)_mean',
                title='Average CPC by Format',
                color='Format',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        # Normalized comparison
        normalized = format_comp.copy()
        
        # Normalize metrics (0-100 scale)
        normalized['CTR_Norm'] = (normalized['CTR_Calculated'] / normalized['CTR_Calculated'].max() * 100).round(2)
        normalized['CPM_Norm'] = (normalized['CPM (cost per 1,000 impressions) (CAD)_mean'] / normalized['CPM (cost per 1,000 impressions) (CAD)_mean'].max() * 100).round(2)
        normalized['CPC_Norm'] = (normalized['CPC (cost per link click) (CAD)_mean'] / normalized['CPC (cost per link click) (CAD)_mean'].max() * 100).round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=normalized['Format'], y=normalized['CTR_Norm'], name='CTR (normalized)', marker_color='#1f77b4'))
        fig.add_trace(go.Bar(x=normalized['Format'], y=normalized['CPM_Norm'], name='CPM (normalized)', marker_color='#ff7f0e'))
        fig.add_trace(go.Bar(x=normalized['Format'], y=normalized['CPC_Norm'], name='CPC (normalized)', marker_color='#2ca02c'))
        
        fig.update_layout(
            barmode='group',
            height=500,
            title='Normalized Metrics Comparison by Format',
            xaxis_title='Format',
            yaxis_title='Normalized Value (%)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.header("💡 Format Insights")
    
    image_format = format_comp[format_comp['Format'] == 'Image'].iloc[0] if 'Image' in format_comp['Format'].values else None
    video_format = format_comp[format_comp['Format'] == 'Video'].iloc[0] if 'Video' in format_comp['Format'].values else None
    
    if image_format is not None and video_format is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Image Format:**
            - Total Spend: ${image_format['Amount spent (CAD)_sum']:,.2f}
            - Total Impressions: {image_format['Impressions_sum']:,.0f}
            - CTR: {image_format['CTR_Calculated']:.2f}%
            - CPM: ${image_format['CPM (cost per 1,000 impressions) (CAD)_mean']:.2f}
            - CPC: ${image_format['CPC (cost per link click) (CAD)_mean']:.2f}
            """)
        
        with col2:
            st.info(f"""
            **Video Format:**
            - Total Spend: ${video_format['Amount spent (CAD)_sum']:,.2f}
            - Total Impressions: {video_format['Impressions_sum']:,.0f}
            - CTR: {video_format['CTR_Calculated']:.2f}%
            - CPM: ${video_format['CPM (cost per 1,000 impressions) (CAD)_mean']:.2f}
            - CPC: ${video_format['CPC (cost per link click) (CAD)_mean']:.2f}
            """)
        
        # Comparison
        if video_format['CTR_Calculated'] > image_format['CTR_Calculated']:
            st.success(f"✅ Video format has {((video_format['CTR_Calculated'] / image_format['CTR_Calculated'] - 1) * 100):.1f}% higher CTR than Image format")
        else:
            st.info(f"📊 Image format has {((image_format['CTR_Calculated'] / video_format['CTR_Calculated'] - 1) * 100):.1f}% higher CTR than Video format")
        
        if video_format['CPC (cost per link click) (CAD)_mean'] < image_format['CPC (cost per link click) (CAD)_mean']:
            st.success(f"✅ Video format has ${(image_format['CPC (cost per link click) (CAD)_mean'] - video_format['CPC (cost per link click) (CAD)_mean']):.2f} lower CPC than Image format")
        else:
            st.info(f"📊 Image format has ${(video_format['CPC (cost per link click) (CAD)_mean'] - image_format['CPC (cost per link click) (CAD)_mean']):.2f} lower CPC than Video format")
    
    # Export button
    if st.button("📥 Export Format Comparison to CSV"):
        csv = display_format.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="format_comparison.csv",
            mime="text/csv"
        )

