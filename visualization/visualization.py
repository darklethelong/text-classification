"""
Visualization module for complaint detection results.
Provides functions for visualizing complaints in conversations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import os

import sys
sys.path.append('.')
import config


def plot_complaint_timeline(df, timestamp_column=None, save_path=None):
    """
    Plot a timeline of complaints with their probabilities.
    
    Args:
        df (pd.DataFrame): DataFrame with complaint predictions
        timestamp_column (str): Name of the timestamp column
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # If no timestamp column is provided, use the index as x-axis
    if timestamp_column is None or timestamp_column not in df.columns:
        x = np.arange(len(df))
        x_label = "Sample Index"
    else:
        x = pd.to_datetime(df[timestamp_column])
        x_label = "Time"
    
    # Create masks for complaints and non-complaints
    complaint_mask = df["is_complaint"] == 1
    non_complaint_mask = ~complaint_mask
    
    # Plot scatter points
    ax.scatter(x[complaint_mask.values], df["complaint_probability"][complaint_mask.values], 
              color='red', s=100, alpha=0.7, label='Complaint')
    ax.scatter(x[non_complaint_mask.values], df["complaint_probability"][non_complaint_mask.values], 
              color='blue', s=100, alpha=0.7, label='Non-Complaint')
    
    # Add threshold line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold (0.5)')
    
    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel('Complaint Probability')
    ax.set_title('Complaint Timeline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    
    # Rotate x-axis labels if using timestamps
    if timestamp_column is not None and timestamp_column in df.columns:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path)
    
    return plt.gcf()


def plot_complaint_percentage_gauge(percentage, save_path=None):
    """
    Plot gauge chart for complaint percentage.
    
    Args:
        percentage (float): Complaint percentage (0-1)
        save_path (str, optional): Path to save figure
        
    Returns:
        go.Figure: Plotly figure
    """
    # Create figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Complaint Percentage"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': percentage * 100
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        paper_bgcolor="white",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    # Save figure if path provided
    if save_path:
        fig.write_image(save_path)
    
    return fig


def plot_complaint_heatmap(df, save_path=None):
    """
    Plot heatmap of complaint intensities.
    
    Args:
        df (pd.DataFrame): Dataframe with complaint analysis
        save_path (str, optional): Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Create intensity data
    intensity_values = df["complaint_probability"].values.reshape(-1, 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot heatmap
    sns.heatmap(intensity_values.T, cmap="Reds", cbar_kws={"label": "Complaint Intensity"},
               xticklabels=False, yticklabels=False, ax=ax)
    
    # Add labels and title
    ax.set_xlabel("Conversation Flow")
    ax.set_title("Complaint Intensity Heatmap")
    
    # Save figure if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    return fig


def plot_complaint_wordcloud(df, text_column="text", save_path=None):
    """
    Plot wordcloud of complaint texts.
    
    Args:
        df (pd.DataFrame): Dataframe with complaint analysis
        text_column (str): Column containing text
        save_path (str, optional): Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Filter complaint texts
    # Convert is_complaint to boolean if it's not already
    if "is_complaint" in df.columns:
        df["is_complaint"] = df["is_complaint"].astype(bool)
        complaint_texts = df[df["is_complaint"] == True][text_column].tolist()
    else:
        # Fallback if is_complaint column doesn't exist
        complaint_texts = []
    
    # If no complaints, return empty figure
    if not complaint_texts:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.text(0.5, 0.5, "No complaints found", ha='center', va='center', fontsize=20)
        ax.axis('off')
        
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
        
        return fig
    
    # Combine texts
    complaint_text = " ".join(complaint_texts)
    
    # Create wordcloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color="white",
        max_words=100,
        contour_width=3,
        contour_color="steelblue"
    ).generate(complaint_text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot wordcloud
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Common Words in Complaints", fontsize=20)
    
    # Save figure if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    return fig


def plot_complaint_distribution(df, save_path=None):
    """
    Plot distribution of complaint probabilities.
    
    Args:
        df (pd.DataFrame): Dataframe with complaint analysis
        save_path (str, optional): Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(df["complaint_probability"], bins=20, kde=True, ax=ax)
    
    # Add threshold line
    ax.axvline(x=config.COMPLAINT_THRESHOLD, color='r', linestyle='--', alpha=0.7,
              label=f"Threshold ({config.COMPLAINT_THRESHOLD})")
    
    # Add labels and title
    ax.set_xlabel("Complaint Probability")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Complaint Probabilities")
    ax.legend()
    
    # Save figure if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    return fig


def plot_complaint_intensity_pie(df, save_path=None):
    """
    Plot pie chart of complaint intensities.
    
    Args:
        df (pd.DataFrame): Dataframe with complaint analysis
        save_path (str, optional): Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Count intensities
    intensity_counts = df["complaint_intensity"].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set colors
    colors = ["lightgreen", "yellow", "orange", "red"]
    
    # Plot pie chart
    wedges, texts, autotexts = ax.pie(
        intensity_counts,
        labels=intensity_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors
    )
    
    # Customize pie chart
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title("Complaint Intensity Distribution", fontsize=16)
    
    # Add legend
    ax.legend(
        wedges,
        intensity_counts.index,
        title="Intensity Levels",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    # Save figure if path provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    return fig


def generate_dashboard(df, output_dir, timestamp_column=None, text_column="text"):
    """
    Generate complete visualization dashboard.
    
    Args:
        df (pd.DataFrame): Dataframe with complaint analysis
        output_dir (str): Directory to save visualizations
        timestamp_column (str, optional): Column containing timestamps
        text_column (str): Column containing text
        
    Returns:
        dict: Dictionary of figures
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate complaint percentage
    complaint_percentage = df["is_complaint"].mean()
    
    # Generate visualizations
    figures = {}
    
    # Timeline chart
    timeline_path = os.path.join(output_dir, "complaint_timeline.png")
    figures["timeline"] = plot_complaint_timeline(df, timestamp_column, timeline_path)
    
    # Gauge chart
    gauge_path = os.path.join(output_dir, "complaint_gauge.png")
    figures["gauge"] = plot_complaint_percentage_gauge(complaint_percentage, gauge_path)
    
    # Heatmap
    heatmap_path = os.path.join(output_dir, "complaint_heatmap.png")
    figures["heatmap"] = plot_complaint_heatmap(df, heatmap_path)
    
    # Wordcloud
    wordcloud_path = os.path.join(output_dir, "complaint_wordcloud.png")
    figures["wordcloud"] = plot_complaint_wordcloud(df, text_column, wordcloud_path)
    
    # Distribution
    distribution_path = os.path.join(output_dir, "complaint_distribution.png")
    figures["distribution"] = plot_complaint_distribution(df, distribution_path)
    
    # Pie chart
    pie_path = os.path.join(output_dir, "complaint_intensity_pie.png")
    figures["pie"] = plot_complaint_intensity_pie(df, pie_path)
    
    print(f"Visualizations generated and saved to {output_dir}")
    
    return figures


def generate_report(df, output_file, timestamp_column=None, text_column="text"):
    """
    Generate HTML report with complaint analysis.
    
    Args:
        df (pd.DataFrame): Dataframe with complaint analysis
        output_file (str): Path to output HTML file
        timestamp_column (str, optional): Column containing timestamps
        text_column (str): Column containing text
    """
    # Calculate complaint percentage
    complaint_percentage = df["is_complaint"].mean()
    
    # Create base directory for images
    base_dir = os.path.dirname(output_file)
    img_dir = os.path.join(base_dir, "report_images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Generate visualizations
    generate_dashboard(df, img_dir, timestamp_column, text_column)
    
    # Get relative paths for images
    img_relative_dir = os.path.relpath(img_dir, base_dir)
    timeline_path = os.path.join(img_relative_dir, "complaint_timeline.png")
    gauge_path = os.path.join(img_relative_dir, "complaint_gauge.png")
    heatmap_path = os.path.join(img_relative_dir, "complaint_heatmap.png")
    wordcloud_path = os.path.join(img_relative_dir, "complaint_wordcloud.png")
    distribution_path = os.path.join(img_relative_dir, "complaint_distribution.png")
    pie_path = os.path.join(img_relative_dir, "complaint_intensity_pie.png")
    
    # Find highest intensity complaints
    high_complaints = df[df["complaint_intensity"].isin(["High", "Very High"])].sort_values(
        "complaint_probability", ascending=False
    )
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Complaint Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #1a468e; color: white; padding: 20px; text-align: center; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            .flex-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
            .chart {{ margin: 10px; max-width: 100%; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .summary {{ font-size: 18px; margin: 20px 0; }}
            .highlight {{ color: #d9534f; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Complaint Analysis Report</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <p class="summary">
                Analysis of <strong>{len(df)}</strong> conversation utterances found 
                <span class="highlight">{df["is_complaint"].sum()}</span> complaints, 
                representing <span class="highlight">{complaint_percentage:.1%}</span> of the conversation.
            </p>
        </div>
        
        <div class="section">
            <h2>Complaint Overview</h2>
            <div class="flex-container">
                <div class="chart">
                    <img src="{gauge_path}" alt="Complaint Percentage Gauge" style="max-width: 100%;" />
                </div>
                <div class="chart">
                    <img src="{pie_path}" alt="Complaint Intensity Distribution" style="max-width: 100%;" />
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Complaint Timeline</h2>
            <div class="chart">
                <img src="{timeline_path}" alt="Complaint Timeline" style="max-width: 100%;" />
            </div>
            <div class="chart">
                <img src="{heatmap_path}" alt="Complaint Heatmap" style="max-width: 100%;" />
            </div>
        </div>
        
        <div class="section">
            <h2>Complaint Distribution</h2>
            <div class="chart">
                <img src="{distribution_path}" alt="Complaint Distribution" style="max-width: 100%;" />
            </div>
            <div class="chart">
                <img src="{wordcloud_path}" alt="Complaint Wordcloud" style="max-width: 100%;" />
            </div>
        </div>
    """
    
    # Add high intensity complaints section if any exist
    if len(high_complaints) > 0:
        html_content += f"""
        <div class="section">
            <h2>High Intensity Complaints</h2>
            <p>The following utterances were identified as high intensity complaints:</p>
            <table>
                <tr>
                    <th>Text</th>
                    <th>Probability</th>
                    <th>Intensity</th>
                </tr>
        """
        
        # Add rows for high complaints (limit to top 10)
        for _, row in high_complaints.head(10).iterrows():
            html_content += f"""
                <tr>
                    <td>{row[text_column]}</td>
                    <td>{row['complaint_probability']:.3f}</td>
                    <td>{row['complaint_intensity']}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
    
    # Complete HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Report generated and saved to {output_file}")


def create_dashboard_app(df, timestamp_column=None, text_column="text"):
    """
    Create a Dash application for interactive visualizations.
    This function returns code that can be used to create a Dash app.
    
    Args:
        df (pd.DataFrame): Dataframe with complaint analysis
        timestamp_column (str, optional): Column containing timestamps
        text_column (str): Column containing text
        
    Returns:
        str: Dash application code
    """
    # Generate app code
    app_code = f"""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import os

# Load data
df = pd.read_csv('complaint_analysis.csv')

# Calculate complaint percentage
complaint_percentage = df["is_complaint"].mean()

# Create Dash app
app = dash.Dash(__name__, title="Complaint Analysis Dashboard")

# Define layout
app.layout = html.Div([
    html.Div([
        html.H1("Complaint Analysis Dashboard"),
        html.P(f"Analysis of {{len(df)}} conversation utterances"),
    ], style={{'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#1a468e', 'color': 'white'}}),
    
    html.Div([
        html.Div([
            html.H3("Complaint Percentage"),
            dcc.Graph(
                id='complaint-gauge',
                figure=go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=complaint_percentage * 100,
                    domain={{'x': [0, 1], 'y': [0, 1]}},
                    title={{'text': "Complaint %"}},
                    gauge={{
                        'axis': {{'range': [0, 100]}},
                        'bar': {{'color': "darkblue"}},
                        'steps': [
                            {{'range': [0, 25], 'color': "lightgreen"}},
                            {{'range': [25, 50], 'color': "yellow"}},
                            {{'range': [50, 75], 'color': "orange"}},
                            {{'range': [75, 100], 'color': "red"}}
                        ],
                        'threshold': {{
                            'line': {{'color': "red", 'width': 4}},
                            'thickness': 0.75,
                            'value': complaint_percentage * 100
                        }}
                    }}
                ))
            )
        ], style={{'width': '50%', 'display': 'inline-block'}}),
        
        html.Div([
            html.H3("Complaint Intensity Distribution"),
            dcc.Graph(
                id='intensity-pie',
                figure=px.pie(
                    df, 
                    names="complaint_intensity",
                    title="Complaint Intensity Distribution",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
            )
        ], style={{'width': '50%', 'display': 'inline-block'}}),
    ]),
    
    html.Div([
        html.H3("Complaint Timeline"),
        dcc.Graph(
            id='complaint-timeline',
            figure=px.line(
                df, 
                {'timestamp_str' if timestamp_column else 'index'}: {timestamp_column if timestamp_column else range(len(df))},
                y="complaint_probability",
                title="Complaint Timeline",
                markers=True
            ).add_hline(y=0.5, line_dash="dash", line_color="red")
        )
    ]),
    
    html.Div([
        html.H3("Complaint Distribution"),
        dcc.Graph(
            id='complaint-histogram',
            figure=px.histogram(
                df, 
                x="complaint_probability",
                nbins=20,
                title="Distribution of Complaint Probabilities"
            ).add_vline(x=0.5, line_dash="dash", line_color="red")
        )
    ]),
    
    html.Div([
        html.H3("High Intensity Complaints"),
        html.Div(id='complaint-table', children=[
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in ['Text', 'Probability', 'Intensity']])] +
                # Body
                [html.Tr([
                    html.Td(row['{text_column}']),
                    html.Td(f"{{row['complaint_probability']:.3f}}"),
                    html.Td(row['complaint_intensity'])
                ]) for _, row in df[df["complaint_intensity"].isin(["High", "Very High"])].head(10).iterrows()]
            )
        ])
    ], style={{'overflowX': 'auto', 'padding': '20px'}}),
    
    html.Div([
        html.Footer([
            html.P("Complaint Detection System - Generated " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
        ], style={{'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f0f0f0'}})
    ])
])

if __name__ == '__main__':
    # Save data for app
    df.to_csv('complaint_analysis.csv', index=False)
    
    # Run app
    app.run_server(debug=True)
"""
    
    return app_code 