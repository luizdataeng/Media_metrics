"""
Data loading and preparation module for Marketing Analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MarketingDataLoader:
    """Load and prepare marketing data for analysis"""
    
    def __init__(self, csv_file):
        """Initialize the data loader with CSV file"""
        self.df = pd.read_csv(csv_file)
        self.prepare_data()
    
    def prepare_data(self):
        """Clean and prepare data for analysis"""
        # Convert date columns
        self.df['Reporting starts'] = pd.to_datetime(self.df['Reporting starts'])
        self.df['Reporting ends'] = pd.to_datetime(self.df['Reporting ends'])
        
        # Create month column for grouping
        self.df['Month'] = self.df['Reporting starts'].dt.to_period('M')
        self.df['Month_Name'] = self.df['Reporting starts'].dt.strftime('%Y-%m')
        self.df['Month_Display'] = self.df['Reporting starts'].dt.strftime('%b %Y')
        
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
    
    def get_basic_stats(self):
        """Get basic statistics about the dataset"""
        return {
            'total_rows': self.df.shape[0],
            'total_columns': self.df.shape[1],
            'date_start': self.df['Reporting starts'].min(),
            'date_end': self.df['Reporting ends'].max(),
            'total_campaigns': self.df['Campaign'].nunique(),
            'total_ads': self.df['Ad Name'].nunique(),
            'unique_months': self.df['Month'].nunique(),
            'total_spend': self.df['Amount spent (CAD)'].sum(),
            'total_impressions': self.df['Impressions'].sum(),
            'total_reach': self.df['Reach'].sum(),
            'total_clicks': self.df['Link clicks'].sum(),
            'avg_ctr': self.df['CTR (link click-through rate)'].mean(),
            'avg_cpm': self.df['CPM (cost per 1,000 impressions) (CAD)'].mean(),
            'avg_cpc': self.df['CPC (cost per link click) (CAD)'].mean(),
            'format_distribution': self.df['Format'].value_counts().to_dict()
        }
    
    def get_monthly_kpis(self):
        """Get monthly KPI summary"""
        monthly = self.df.groupby('Month_Name').agg({
            'Amount spent (CAD)': 'sum',
            'Reach': 'sum',
            'Impressions': 'sum',
            'Link clicks': 'sum',
            'CTR (link click-through rate)': 'mean',
            'CPM (cost per 1,000 impressions) (CAD)': 'mean',
            'CPC (cost per link click) (CAD)': 'mean'
        }).round(2)
        
        # Calculate CTR from clicks/impressions
        monthly['CTR_Calculated'] = (monthly['Link clicks'] / monthly['Impressions'] * 100).replace([np.inf, -np.inf], 0).round(2)
        
        # Add month display names
        monthly_display = self.df.groupby('Month_Name')['Month_Display'].first()
        monthly = monthly.merge(monthly_display, left_index=True, right_index=True)
        
        return monthly.reset_index()
    
    def get_daily_data(self):
        """Get daily aggregated data for trend analysis"""
        daily = self.df.groupby('Reporting starts').agg({
            'Impressions': 'sum',
            'Link clicks': 'sum',
            'Amount spent (CAD)': 'sum'
        }).reset_index()
        
        daily['CTR'] = (daily['Link clicks'] / daily['Impressions'] * 100).replace([np.inf, -np.inf], 0)
        daily = daily.sort_values('Reporting starts')
        
        return daily
    
    def get_campaign_performance(self):
        """Get campaign-level performance metrics"""
        campaign_perf = self.df.groupby('Campaign').agg({
            'Amount spent (CAD)': 'sum',
            'Impressions': 'sum',
            'Reach': 'sum',
            'Link clicks': 'sum',
            'CTR (link click-through rate)': 'mean',
            'CPM (cost per 1,000 impressions) (CAD)': 'mean',
            'CPC (cost per link click) (CAD)': 'mean'
        }).round(2)
        
        campaign_perf['CTR_Calculated'] = (campaign_perf['Link clicks'] / campaign_perf['Impressions'] * 100).replace([np.inf, -np.inf], 0).round(2)
        campaign_perf = campaign_perf.rename(columns={'CTR_Calculated': 'CTR (%)'})
        
        return campaign_perf.reset_index()
    
    def get_ad_performance(self):
        """Get ad-level performance metrics"""
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
        
        ad_perf['CTR_Calculated'] = (ad_perf['Link clicks'] / ad_perf['Impressions'] * 100).replace([np.inf, -np.inf], 0).round(2)
        ad_perf = ad_perf.rename(columns={'CTR_Calculated': 'CTR (%)'})
        
        return ad_perf.reset_index()
    
    def get_format_comparison(self):
        """Get format comparison metrics"""
        format_comp = self.df.groupby('Format').agg({
            'Amount spent (CAD)': ['sum', 'mean'],
            'Impressions': ['sum', 'mean'],
            'Reach': ['sum', 'mean'],
            'Link clicks': ['sum', 'mean'],
            'CTR (link click-through rate)': 'mean',
            'CPM (cost per 1,000 impressions) (CAD)': 'mean',
            'CPC (cost per link click) (CAD)': 'mean'
        }).round(2)
        
        # Flatten column names
        format_comp.columns = ['_'.join(col).strip() for col in format_comp.columns.values]
        
        # Calculate CTR from clicks/impressions
        format_comp['CTR_Calculated'] = (format_comp['Link clicks_sum'] / format_comp['Impressions_sum'] * 100).replace([np.inf, -np.inf], 0).round(2)
        
        return format_comp.reset_index()

