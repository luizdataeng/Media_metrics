"""
Marketing Analysis Script for Social Media Metrics
Analyzes campaign performance, trends, and format comparisons
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Color palette
colors = sns.color_palette("husl", 8)

class MarketingAnalyzer:
    def __init__(self, csv_file):
        """Initialize the analyzer with data from CSV file"""
        print(f"Loading data from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        self.prepare_data()
        print(f"Data loaded successfully! Shape: {self.df.shape}")
        
    def prepare_data(self):
        """Clean and prepare data for analysis"""
        # Convert date columns
        self.df['Reporting starts'] = pd.to_datetime(self.df['Reporting starts'])
        self.df['Reporting ends'] = pd.to_datetime(self.df['Reporting ends'])
        
        # Create month column for grouping
        self.df['Month'] = self.df['Reporting starts'].dt.to_period('M')
        self.df['Month_Name'] = self.df['Reporting starts'].dt.strftime('%Y-%m')
        
        # Fill NaN values in numeric columns with 0
        numeric_cols = ['Impressions', 'Reach', 'Link clicks', 'Amount spent (CAD)',
                       'CPM (cost per 1,000 impressions) (CAD)', 
                       'CPC (cost per link click) (CAD)', 
                       'CTR (link click-through rate)']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Clean campaign and ad names
        self.df['Campaign'] = self.df['New Ad set name'].fillna('Unknown')
        self.df['Ad Name'] = self.df['New Ad name'].fillna('Unknown')
        
        # Ensure format column exists
        if 'Ad Video' in self.df.columns:
            self.df['Format'] = self.df['Ad Video'].fillna('Unknown')
        else:
            self.df['Format'] = 'Unknown'
    
    def basic_stats(self):
        """Display basic statistics about the dataset"""
        print("\n" + "="*80)
        print("BASIC DATASET STATISTICS")
        print("="*80)
        
        print(f"\nDataset Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"Date Range: {self.df['Reporting starts'].min()} to {self.df['Reporting ends'].max()}")
        print(f"Total Campaigns: {self.df['Campaign'].nunique()}")
        print(f"Total Ads: {self.df['Ad Name'].nunique()}")
        print(f"Unique Months: {self.df['Month'].nunique()}")
        
        print("\n--- Overall Metrics Summary ---")
        metrics = {
            'Total Spend (CAD)': self.df['Amount spent (CAD)'].sum(),
            'Total Impressions': self.df['Impressions'].sum(),
            'Total Reach': self.df['Reach'].sum(),
            'Total Clicks': self.df['Link clicks'].sum(),
            'Average CTR (%)': self.df['CTR (link click-through rate)'].mean(),
            'Average CPM (CAD)': self.df['CPM (cost per 1,000 impressions) (CAD)'].mean(),
            'Average CPC (CAD)': self.df['CPC (cost per link click) (CAD)'].mean()
        }
        
        for key, value in metrics.items():
            if 'Total' in key or 'Average' in key:
                if 'Spend' in key or 'CPM' in key or 'CPC' in key:
                    print(f"{key}: ${value:,.2f}")
                elif 'CTR' in key:
                    print(f"{key}: {value:.2f}%")
                else:
                    print(f"{key}: {value:,.0f}")
        
        print("\n--- Format Distribution ---")
        print(self.df['Format'].value_counts())
        
        return metrics
    
    def monthly_kpi_summary(self):
        """Calculate and display monthly KPI averages"""
        print("\n" + "="*80)
        print("MONTHLY KPI SUMMARY")
        print("="*80)
        
        # Group by month and calculate averages
        monthly_metrics = self.df.groupby('Month_Name').agg({
            'Amount spent (CAD)': 'sum',
            'Reach': 'sum',
            'Impressions': 'sum',
            'Link clicks': 'sum',
            'CTR (link click-through rate)': 'mean',
            'CPM (cost per 1,000 impressions) (CAD)': 'mean',
            'CPC (cost per link click) (CAD)': 'mean'
        }).round(2)
        
        monthly_metrics.columns = ['Spend (CAD)', 'Reach', 'Impressions', 'Clicks', 
                                   'Avg CTR (%)', 'Avg CPM (CAD)', 'Avg CPC (CAD)']
        
        print("\nMonthly Averages:")
        print(monthly_metrics.to_string())
        
        # Create visual KPI cards/table
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        months = monthly_metrics.index
        metrics_to_plot = ['Spend (CAD)', 'Reach', 'Impressions', 'Clicks', 
                          'Avg CTR (%)', 'Avg CPM (CAD)', 'Avg CPC (CAD)']
        
        for idx, metric in enumerate(metrics_to_plot[:7]):
            ax = axes[idx]
            values = monthly_metrics[metric]
            bars = ax.bar(range(len(months)), values, color=colors[idx % len(colors)])
            ax.set_xticks(range(len(months)))
            ax.set_xticklabels(months, rotation=45, ha='right')
            ax.set_title(f'{metric} by Month', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:,.0f}' if 'CAD' not in metric and 'CTR' not in metric and 'CPC' not in metric and 'CPM' not in metric
                           else f'${height:.2f}' if 'CAD' in metric or 'CPC' in metric or 'CPM' in metric
                           else f'{height:.2f}%',
                           ha='center', va='bottom', fontsize=9)
        
        # Hide the 8th subplot
        axes[7].axis('off')
        
        plt.tight_layout()
        plt.savefig('monthly_kpi_summary.png', dpi=300, bbox_inches='tight')
        print("\nMonthly KPI Summary saved as 'monthly_kpi_summary.png'")
        plt.close()
        
        return monthly_metrics
    
    def trend_lines(self):
        """Create trend line visualizations"""
        print("\n" + "="*80)
        print("TREND LINE VISUALIZATIONS")
        print("="*80)
        
        # Aggregate data by date (Reporting starts)
        daily_data = self.df.groupby('Reporting starts').agg({
            'Impressions': 'sum',
            'Link clicks': 'sum',
            'Amount spent (CAD)': 'sum'
        }).reset_index()
        
        daily_data['CTR'] = (daily_data['Link clicks'] / daily_data['Impressions'] * 100).replace([np.inf, -np.inf], 0)
        daily_data = daily_data.sort_values('Reporting starts')
        
        # 1. Impressions vs Spend
        fig, ax = plt.subplots(figsize=(14, 6))
        ax2 = ax.twinx()
        
        line1 = ax.plot(daily_data['Reporting starts'], daily_data['Impressions'], 
                       color=colors[0], label='Impressions', linewidth=2, marker='o', markersize=4)
        line2 = ax2.plot(daily_data['Reporting starts'], daily_data['Amount spent (CAD)'],
                        color=colors[1], label='Spend (CAD)', linewidth=2, marker='s', markersize=4)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Impressions', fontsize=12, fontweight='bold', color=colors[0])
        ax2.set_ylabel('Spend (CAD)', fontsize=12, fontweight='bold', color=colors[1])
        ax.set_title('Impressions vs Spend Over Time', fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('trend_impressions_vs_spend.png', dpi=300, bbox_inches='tight')
        print("Trend: Impressions vs Spend saved as 'trend_impressions_vs_spend.png'")
        plt.close()
        
        # 2. Clicks vs Spend (dual-axis)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax2 = ax.twinx()
        
        line1 = ax.plot(daily_data['Reporting starts'], daily_data['Link clicks'],
                       color=colors[2], label='Clicks', linewidth=2, marker='o', markersize=4)
        line2 = ax2.plot(daily_data['Reporting starts'], daily_data['Amount spent (CAD)'],
                        color=colors[3], label='Spend (CAD)', linewidth=2, marker='s', markersize=4)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Clicks', fontsize=12, fontweight='bold', color=colors[2])
        ax2.set_ylabel('Spend (CAD)', fontsize=12, fontweight='bold', color=colors[3])
        ax.set_title('Clicks vs Spend Over Time', fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('trend_clicks_vs_spend.png', dpi=300, bbox_inches='tight')
        print("Trend: Clicks vs Spend saved as 'trend_clicks_vs_spend.png'")
        plt.close()
        
        # 3. Impressions over time
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(daily_data['Reporting starts'], daily_data['Impressions'],
               color=colors[0], linewidth=2.5, marker='o', markersize=5)
        ax.fill_between(daily_data['Reporting starts'], daily_data['Impressions'],
                       alpha=0.3, color=colors[0])
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Impressions', fontsize=12, fontweight='bold')
        ax.set_title('Impressions Over Time', fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('trend_impressions_over_time.png', dpi=300, bbox_inches='tight')
        print("Trend: Impressions Over Time saved as 'trend_impressions_over_time.png'")
        plt.close()
        
        # 4. CTR over time
        monthly_ctr = self.df.groupby('Month_Name').agg({
            'Link clicks': 'sum',
            'Impressions': 'sum'
        })
        monthly_ctr['CTR'] = (monthly_ctr['Link clicks'] / monthly_ctr['Impressions'] * 100).replace([np.inf, -np.inf], 0)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(monthly_ctr.index, monthly_ctr['CTR'],
               color=colors[4], linewidth=2.5, marker='o', markersize=8)
        ax.fill_between(monthly_ctr.index, monthly_ctr['CTR'],
                       alpha=0.3, color=colors[4])
        
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('CTR (%)', fontsize=12, fontweight='bold')
        ax.set_title('CTR Over Time (Monthly)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('trend_ctr_over_time.png', dpi=300, bbox_inches='tight')
        print("Trend: CTR Over Time saved as 'trend_ctr_over_time.png'")
        plt.close()
    
    def campaign_performance(self):
        """Create campaign-level performance table and graphics"""
        print("\n" + "="*80)
        print("CAMPAIGN-LEVEL PERFORMANCE")
        print("="*80)
        
        campaign_perf = self.df.groupby('Campaign').agg({
            'Amount spent (CAD)': 'sum',
            'Impressions': 'sum',
            'Reach': 'sum',
            'Link clicks': 'sum',
            'CTR (link click-through rate)': 'mean',
            'CPM (cost per 1,000 impressions) (CAD)': 'mean',
            'CPC (cost per link click) (CAD)': 'mean'
        }).round(2)
        
        campaign_perf['CTR_Calculated'] = (campaign_perf['Link clicks'] / campaign_perf['Impressions'] * 100).replace([np.inf, -np.inf], 0)
        campaign_perf['CTR_Calculated'] = campaign_perf['CTR_Calculated'].round(2)
        
        campaign_perf.columns = ['Spend (CAD)', 'Impressions', 'Reach', 'Clicks',
                                'Avg CTR (%)', 'Avg CPM (CAD)', 'Avg CPC (CAD)', 'CTR (%)']
        
        # Select the calculated CTR for display
        campaign_perf = campaign_perf[['Spend (CAD)', 'Impressions', 'Reach', 'Clicks',
                                      'CTR (%)', 'Avg CPM (CAD)', 'Avg CPC (CAD)']]
        
        campaign_perf = campaign_perf.sort_values('Spend (CAD)', ascending=False)
        
        print("\nCampaign Performance Table:")
        print(campaign_perf.to_string())
        
        # Save as CSV
        campaign_perf.to_csv('campaign_performance.csv')
        print("\nCampaign performance table saved as 'campaign_performance.csv'")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top 10 campaigns by spend
        top_campaigns = campaign_perf.head(10)
        
        # CTR by Campaign
        ax1 = axes[0, 0]
        top_ctr = campaign_perf.nlargest(10, 'CTR (%)')
        bars1 = ax1.barh(range(len(top_ctr)), top_ctr['CTR (%)'], color=colors[0])
        ax1.set_yticks(range(len(top_ctr)))
        ax1.set_yticklabels(top_ctr.index, fontsize=9)
        ax1.set_xlabel('CTR (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Top 10 Campaigns by CTR (%)', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # CPM by Campaign
        ax2 = axes[0, 1]
        top_cpm = campaign_perf.nlargest(10, 'Avg CPM (CAD)')
        bars2 = ax2.barh(range(len(top_cpm)), top_cpm['Avg CPM (CAD)'], color=colors[1])
        ax2.set_yticks(range(len(top_cpm)))
        ax2.set_yticklabels(top_cpm.index, fontsize=9)
        ax2.set_xlabel('CPM (CAD)', fontsize=11, fontweight='bold')
        ax2.set_title('Top 10 Campaigns by CPM (CAD)', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # CPC by Campaign
        ax3 = axes[1, 0]
        top_cpc = campaign_perf.nlargest(10, 'Avg CPC (CAD)')
        bars3 = ax3.barh(range(len(top_cpc)), top_cpc['Avg CPC (CAD)'], color=colors[2])
        ax3.set_yticks(range(len(top_cpc)))
        ax3.set_yticklabels(top_cpc.index, fontsize=9)
        ax3.set_xlabel('CPC (CAD)', fontsize=11, fontweight='bold')
        ax3.set_title('Top 10 Campaigns by CPC (CAD)', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Spend vs Clicks scatter
        ax4 = axes[1, 1]
        scatter = ax4.scatter(campaign_perf['Spend (CAD)'], campaign_perf['Clicks'],
                            s=100, alpha=0.6, c=campaign_perf['CTR (%)'],
                            cmap='viridis', edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Spend (CAD)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Clicks', fontsize=11, fontweight='bold')
        ax4.set_title('Spend vs Clicks (colored by CTR)', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='CTR (%)')
        
        plt.tight_layout()
        plt.savefig('campaign_performance.png', dpi=300, bbox_inches='tight')
        print("Campaign performance visualization saved as 'campaign_performance.png'")
        plt.close()
        
        return campaign_perf
    
    def ad_performance(self):
        """Create ad-level performance table and graphics"""
        print("\n" + "="*80)
        print("AD-LEVEL PERFORMANCE")
        print("="*80)
        
        ad_perf = self.df.groupby('Ad Name').agg({
            'Amount spent (CAD)': 'sum',
            'Impressions': 'sum',
            'Reach': 'sum',
            'Link clicks': 'sum',
            'CTR (link click-through rate)': 'mean',
            'CPM (cost per 1,000 impressions) (CAD)': 'mean',
            'CPC (cost per link click) (CAD)': 'mean',
            'Campaign': 'first',
            'Format': 'first'
        }).round(2)
        
        ad_perf['CTR_Calculated'] = (ad_perf['Link clicks'] / ad_perf['Impressions'] * 100).replace([np.inf, -np.inf], 0)
        ad_perf['CTR_Calculated'] = ad_perf['CTR_Calculated'].round(2)
        
        ad_perf.columns = ['Spend (CAD)', 'Impressions', 'Reach', 'Clicks',
                          'Avg CTR (%)', 'Avg CPM (CAD)', 'Avg CPC (CAD)', 
                          'Campaign', 'Format', 'CTR (%)']
        
        ad_perf = ad_perf[['Campaign', 'Format', 'Spend (CAD)', 'Impressions', 'Reach', 'Clicks',
                          'CTR (%)', 'Avg CPM (CAD)', 'Avg CPC (CAD)']]
        
        ad_perf = ad_perf.sort_values('Spend (CAD)', ascending=False)
        
        print("\nTop 20 Ads Performance Table:")
        print(ad_perf.head(20).to_string())
        
        # Save as CSV
        ad_perf.to_csv('ad_performance.csv')
        print("\nAd performance table saved as 'ad_performance.csv'")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Filter ads with meaningful spend (> 0)
        active_ads = ad_perf[ad_perf['Spend (CAD)'] > 0].copy()
        
        # Top 15 ads by CTR
        ax1 = axes[0, 0]
        top_ctr_ads = active_ads.nlargest(15, 'CTR (%)')
        bars1 = ax1.barh(range(len(top_ctr_ads)), top_ctr_ads['CTR (%)'], color=colors[0])
        ax1.set_yticks(range(len(top_ctr_ads)))
        ax1.set_yticklabels(top_ctr_ads.index, fontsize=8)
        ax1.set_xlabel('CTR (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Top 15 Ads by CTR (%)', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Top 15 ads by CPM
        ax2 = axes[0, 1]
        top_cpm_ads = active_ads.nlargest(15, 'Avg CPM (CAD)')
        bars2 = ax2.barh(range(len(top_cpm_ads)), top_cpm_ads['Avg CPM (CAD)'], color=colors[1])
        ax2.set_yticks(range(len(top_cpm_ads)))
        ax2.set_yticklabels(top_cpm_ads.index, fontsize=8)
        ax2.set_xlabel('CPM (CAD)', fontsize=11, fontweight='bold')
        ax2.set_title('Top 15 Ads by CPM (CAD)', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Top 15 ads by CPC
        ax3 = axes[1, 0]
        top_cpc_ads = active_ads.nlargest(15, 'Avg CPC (CAD)')
        bars3 = ax3.barh(range(len(top_cpc_ads)), top_cpc_ads['Avg CPC (CAD)'], color=colors[2])
        ax3.set_yticks(range(len(top_cpc_ads)))
        ax3.set_yticklabels(top_cpc_ads.index, fontsize=8)
        ax3.set_xlabel('CPC (CAD)', fontsize=11, fontweight='bold')
        ax3.set_title('Top 15 Ads by CPC (CAD)', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Performance by format
        ax4 = axes[1, 1]
        format_perf = active_ads.groupby('Format').agg({
            'CTR (%)': 'mean',
            'Avg CPM (CAD)': 'mean',
            'Avg CPC (CAD)': 'mean'
        }).round(2)
        
        x = np.arange(len(format_perf))
        width = 0.25
        
        ax4.bar(x - width, format_perf['CTR (%)'], width, label='CTR (%)', color=colors[0])
        ax4.bar(x, format_perf['Avg CPM (CAD)'], width, label='CPM (CAD)', color=colors[1])
        ax4.bar(x + width, format_perf['Avg CPC (CAD)'], width, label='CPC (CAD)', color=colors[2])
        
        ax4.set_xlabel('Format', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax4.set_title('Average Metrics by Ad Format', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(format_perf.index)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ad_performance.png', dpi=300, bbox_inches='tight')
        print("Ad performance visualization saved as 'ad_performance.png'")
        plt.close()
        
        return ad_perf
    
    def format_comparison(self):
        """Compare Video vs Image format metrics"""
        print("\n" + "="*80)
        print("VIDEO vs IMAGE FORMAT COMPARISON")
        print("="*80)
        
        format_comparison = self.df.groupby('Format').agg({
            'Amount spent (CAD)': ['sum', 'mean'],
            'Impressions': ['sum', 'mean'],
            'Reach': ['sum', 'mean'],
            'Link clicks': ['sum', 'mean'],
            'CTR (link click-through rate)': 'mean',
            'CPM (cost per 1,000 impressions) (CAD)': 'mean',
            'CPC (cost per link click) (CAD)': 'mean'
        }).round(2)
        
        # Flatten column names
        format_comparison.columns = ['_'.join(col).strip() for col in format_comparison.columns.values]
        
        # Calculate CTR from clicks/impressions
        format_comparison['CTR_Calculated'] = (format_comparison['Link clicks_sum'] / 
                                             format_comparison['Impressions_sum'] * 100).replace([np.inf, -np.inf], 0)
        format_comparison['CTR_Calculated'] = format_comparison['CTR_Calculated'].round(2)
        
        print("\nFormat Comparison Table:")
        print(format_comparison.to_string())
        
        # Save as CSV
        format_comparison.to_csv('format_comparison.csv')
        print("\nFormat comparison table saved as 'format_comparison.csv'")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        formats = format_comparison.index.tolist()
        colors_format = colors[:len(formats)]
        
        # 1. Total Spend by Format
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(formats, format_comparison['Amount spent (CAD)_sum'], color=colors_format)
        ax1.set_ylabel('Total Spend (CAD)', fontsize=11, fontweight='bold')
        ax1.set_title('Total Spend by Format', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Total Impressions by Format
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(formats, format_comparison['Impressions_sum'], color=colors_format)
        ax2.set_ylabel('Total Impressions', fontsize=11, fontweight='bold')
        ax2.set_title('Total Impressions by Format', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Total Clicks by Format
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(formats, format_comparison['Link clicks_sum'], color=colors_format)
        ax3.set_ylabel('Total Clicks', fontsize=11, fontweight='bold')
        ax3.set_title('Total Clicks by Format', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Average CTR by Format
        ax4 = fig.add_subplot(gs[1, 0])
        bars4 = ax4.bar(formats, format_comparison['CTR_Calculated'], color=colors_format)
        ax4.set_ylabel('CTR (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Average CTR by Format', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
        
        # 5. Average CPM by Format
        ax5 = fig.add_subplot(gs[1, 1])
        bars5 = ax5.bar(formats, format_comparison['CPM (cost per 1,000 impressions) (CAD)_mean'], color=colors_format)
        ax5.set_ylabel('CPM (CAD)', fontsize=11, fontweight='bold')
        ax5.set_title('Average CPM by Format', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars5):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Average CPC by Format
        ax6 = fig.add_subplot(gs[1, 2])
        bars6 = ax6.bar(formats, format_comparison['CPC (cost per link click) (CAD)_mean'], color=colors_format)
        ax6.set_ylabel('CPC (CAD)', fontsize=11, fontweight='bold')
        ax6.set_title('Average CPC by Format', fontsize=12, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        for i, bar in enumerate(bars6):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 7. Side-by-side comparison of key metrics
        ax7 = fig.add_subplot(gs[2, :])
        x = np.arange(len(formats))
        width = 0.2
        
        # Normalize metrics for comparison (using min-max scaling)
        ctr_normalized = format_comparison['CTR_Calculated'] / format_comparison['CTR_Calculated'].max() * 100
        cpm_normalized = format_comparison['CPM (cost per 1,000 impressions) (CAD)_mean'] / format_comparison['CPM (cost per 1,000 impressions) (CAD)_mean'].max() * 100
        cpc_normalized = format_comparison['CPC (cost per link click) (CAD)_mean'] / format_comparison['CPC (cost per link click) (CAD)_mean'].max() * 100
        
        ax7.bar(x - width, ctr_normalized, width, label='CTR (normalized)', color=colors[0])
        ax7.bar(x, cpm_normalized, width, label='CPM (normalized)', color=colors[1])
        ax7.bar(x + width, cpc_normalized, width, label='CPC (normalized)', color=colors[2])
        
        ax7.set_xlabel('Format', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Normalized Value (%)', fontsize=12, fontweight='bold')
        ax7.set_title('Normalized Metrics Comparison by Format', fontsize=13, fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(formats)
        ax7.legend()
        ax7.grid(axis='y', alpha=0.3)
        
        plt.savefig('format_comparison.png', dpi=300, bbox_inches='tight')
        print("Format comparison visualization saved as 'format_comparison.png'")
        plt.close()
        
        return format_comparison
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "="*80)
        print("STARTING MARKETING ANALYSIS")
        print("="*80)
        
        # 1. Basic Stats
        self.basic_stats()
        
        # 2. Monthly KPI Summary
        monthly_kpis = self.monthly_kpi_summary()
        
        # 3. Trend Lines
        self.trend_lines()
        
        # 4. Campaign Performance
        campaign_perf = self.campaign_performance()
        
        # 5. Ad Performance
        ad_perf = self.ad_performance()
        
        # 6. Format Comparison
        format_comp = self.format_comparison()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated Files:")
        print("1. monthly_kpi_summary.png")
        print("2. trend_impressions_vs_spend.png")
        print("3. trend_clicks_vs_spend.png")
        print("4. trend_impressions_over_time.png")
        print("5. trend_ctr_over_time.png")
        print("6. campaign_performance.csv")
        print("7. campaign_performance.png")
        print("8. ad_performance.csv")
        print("9. ad_performance.png")
        print("10. format_comparison.csv")
        print("11. format_comparison.png")
        print("\nAll visualizations saved with 300 DPI for high quality printing.")


if __name__ == "__main__":
    # Initialize analyzer
    csv_file = "Final Dataset - Social Media Metrics Practice.csv"
    analyzer = MarketingAnalyzer(csv_file)
    
    # Run full analysis
    analyzer.run_full_analysis()

