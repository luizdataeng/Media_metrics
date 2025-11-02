# Marketing Analytics Dashboard

A comprehensive Streamlit-based dashboard for analyzing social media marketing metrics and campaign performance. This interactive web application provides real-time insights into ad performance, campaign metrics, and format comparisons through dynamic visualizations.

## Purpose

This project provides a user-friendly interface for marketing teams to:

- **Analyze Campaign Performance**: Track and compare performance metrics across different campaigns
- **Monitor Ad Performance**: Evaluate individual ad performance with detailed metrics
- **Visualize Trends**: Understand trends over time with interactive charts
- **Compare Formats**: Compare video vs image ad formats to optimize marketing strategy
- **Export Data**: Download analysis results as CSV files for reporting

## Features

- 📊 **Overview Dashboard**: High-level metrics and summary statistics
- 📈 **Trend Analysis**: Time-series visualizations of impressions, clicks, and spend
- 📅 **Monthly KPI Summary**: Aggregated monthly performance metrics
- 🎯 **Campaign Performance**: Detailed campaign-level analysis and comparisons
- 📱 **Ad Performance**: Individual ad-level metrics and filtering
- 🎬 **Format Comparison**: Video vs Image format performance analysis

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/luizdataeng/Media_metrics.git
   cd Media_metrics
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the data file exists**
   Ensure that `Final Dataset - Social Media Metrics Practice.csv` is present in the project root directory.

### Running the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`.

## Project Structure

```
├── app.py                    # Main Streamlit dashboard application
├── data_loader.py            # Data loading and preprocessing module
├── requirements.txt          # Python dependencies
├── Final Dataset - Social Media Metrics Practice.csv  # Source data file
└── README.md                 # This file
```

## Usage

1. **Launch the dashboard** using `streamlit run app.py`

2. **Navigate between pages** using the sidebar:
   - Overview: General statistics and format distribution
   - Trend Analysis: Time-series trends for impressions, clicks, and spend
   - Monthly KPI Summary: Monthly aggregated metrics
   - Campaign Performance: Campaign-level analysis
   - Ad Performance: Individual ad metrics
   - Format Comparison: Video vs Image comparison

3. **Use filters** in the sidebar to:
   - Filter by date range
   - Filter by ad format
   - Filter by campaign

4. **Export data** from Campaign Performance, Ad Performance, and Format Comparison pages using the export buttons to download CSV files.

## Dependencies

- **pandas** >= 2.0.0: Data manipulation and analysis
- **streamlit** >= 1.51.0: Web application framework
- **plotly** >= 5.18.0: Interactive visualizations
- **numpy** >= 1.24.0: Numerical computations

## Data Format

The dashboard expects a CSV file with the following key columns:
- `Reporting starts` / `Reporting ends`: Date columns
- `New Ad set name`: Campaign name
- `New Ad name`: Ad name
- `Ad Video`: Format type (Video/Image)
- `Amount spent (CAD)`: Spending amount
- `Impressions`: Number of impressions
- `Reach`: Reach metrics
- `Link clicks`: Number of clicks
- `CTR (link click-through rate)`: Click-through rate
- `CPM (cost per 1,000 impressions) (CAD)`: Cost per mille
- `CPC (cost per link click) (CAD)`: Cost per click

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

