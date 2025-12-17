import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Configure page
st.set_page_config(
    page_title="AI Market Trend Storyteller",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern 3-color palette (Notion-inspired)
PRIMARY_COLOR = "#2E3440"      # Dark slate
SECONDARY_COLOR = "#5E81AC"    # Muted blue
ACCENT_COLOR = "#88C0D0"       # Light blue
BACKGROUND_COLOR = "#FFFFFF"   # Clean white
TEXT_COLOR = "#00040C"         # Dark text

# Custom CSS for modern, clean UI
st.markdown(f"""
    <style>
    /* Ensure entire page has light background */
    html, body {{
        background-color: #FFFFFF;
    }}
    
    /* Main container */
    .main {{
        background-color: {BACKGROUND_COLOR};
    }}
    
    /* Global text color - dark text everywhere */
    .main, p, span, div, li, label, button, input {{
        color: {TEXT_COLOR} !important;
    }}
    
    /* Sidebar - light background with dark text */
    [data-testid="stSidebar"] {{
        background-color: #F9FAFB;
        border-right: 1px solid #E5E9F0;
    }}
    
    [data-testid="stSidebar"] * {{
        color: {TEXT_COLOR} !important;
    }}
    
    /* Navbar/Header - dark text */
    [data-testid="stHeader"] {{
        background-color: #FFFFFF;
    }}
    
    [data-testid="stHeader"] * {{
        color: {TEXT_COLOR} !important;
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {PRIMARY_COLOR};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }}
    
    /* Cards */
    .metric-card {{
        background-color: {BACKGROUND_COLOR};
        border: 1px solid #E5E9F0;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        height: 100%; /* Ensure cards in a row are same height */
    }}
    
    .insight-card {{
        background-color: {BACKGROUND_COLOR};
        border-left: 4px solid {SECONDARY_COLOR};
        border-radius: 4px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {SECONDARY_COLOR};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }}
    
    .stButton>button:hover {{
        background-color: {PRIMARY_COLOR};
        box-shadow: 0 4px 12px rgba(94, 129, 172, 0.3);
    }}
    
    /* Remove default streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* File uploader */
    [data-testid="stFileUploader"] {{
        background-color: {BACKGROUND_COLOR};
        border: 2px dashed {ACCENT_COLOR};
        border-radius: 8px;
        padding: 20px;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 6px;
        color: {TEXT_COLOR};
        padding: 8px 16px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {SECONDARY_COLOR};
        color: white;
    }}
    
    /* Text inputs */
    .stTextInput>div>div>input {{
        border-radius: 6px;
        border: 1px solid #E5E9F0;
    }}
    
    /* Success/Info messages */
    .stSuccess, .stInfo {{
        background-color: #ECEFF4;
        border-radius: 6px;
        border-left: 4px solid {ACCENT_COLOR};
    }}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'df' not in st.session_state:
    st.session_state.df = None


def configure_gemini(api_key):
    """Configure Gemini API"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {str(e)}")
        return False

def get_data_summary(df):
    """Generate a summary of the uploaded data"""
    summary = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "date_range": f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "N/A",
        "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
    }
    return summary

def analyze_data_with_gemini(df, tone, api_key):
    """Send data to Gemini for analysis and get structured JSON response"""
    
    # Prepare data summary for the LLM
    data_summary = get_data_summary(df)
    
    # Convert dataframe to a manageable format for the LLM
    data_sample = df.head(50).to_csv(index=False)
    data_stats = df.describe().to_csv()
    
    # Create prompt based on tone
    tone_instructions = {
        "Analyst": "Write in a professional, data-driven style with technical depth. Use metrics, percentages, and analytical language.",
        "Journalist": "Write in an engaging, narrative style that tells a story. Make it accessible to general readers while maintaining accuracy.",
        "Executive": "Write concisely with focus on bottom-line impact. Lead with key takeaways and strategic implications."
    }
    
    prompt = f"""You are an expert data analyst. Analyze the following dataset and provide comprehensive insights.

**Dataset Overview:**
- Total records: {data_summary['total_rows']}
- Columns: {', '.join(data_summary['columns'])}
- Date range: {data_summary['date_range']}

**Sample Data (first 50 rows):**
{data_sample}

**Statistical Summary:**
{data_stats}

**Task:**
1. Detect trends, patterns, and anomalies in the data
2. Identify correlations between variables
3. Explain the causes behind changes
4. Provide short-term forecasts based on observed patterns

**Tone:** {tone_instructions[tone]}

**CRITICAL: You must respond with ONLY a valid JSON object. No other text before or after. The JSON must have this exact structure:**

{{
    "executive_summary": "A concise 2-3 sentence summary of key findings",
    "key_metrics": [
        {{
            "metric": "Metric name",
            "value": "Value with unit",
            "change": "Percentage change or trend",
            "insight": "What this means"
        }}
    ],
    "trends": [
        {{
            "title": "Trend name",
            "description": "Detailed explanation of the trend",
            "impact": "Business impact",
            "time_period": "When this occurred"
        }}
    ],
    "correlations": [
        {{
            "variables": ["Variable 1", "Variable 2"],
            "relationship": "Description of correlation",
            "strength": "Strong/Moderate/Weak",
            "business_implication": "What this means for business"
        }}
    ],
    "anomalies": [
        {{
            "description": "What anomaly was detected",
            "location": "Where/when it occurred",
            "possible_cause": "Likely explanation"
        }}
    ],
    "forecast": {{
        "next_period": "Prediction for next period",
        "confidence": "High/Medium/Low",
        "key_drivers": ["Driver 1", "Driver 2"],
        "recommendation": "Strategic recommendation"
    }},
    "narrative": "A full narrative story (3-5 paragraphs) that weaves together all insights in the specified tone",
    "visualizations": [
        {{
            "chart_type": "line|bar|scatter|pie",
            "title": "Chart title",
            "x_axis": "Column name for x-axis",
            "y_axis": "Column name or list of columns for y-axis",
            "group_by": "Optional: column for grouping/color",
            "description": "What this chart shows"
        }}
    ]
}}

Remember: ONLY output valid JSON. No markdown, no code blocks, no explanations."""

    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Extract and parse JSON
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        analysis_result = json.loads(response_text)
        
        return analysis_result, None
        
    except json.JSONDecodeError as e:
        return None, f"Failed to parse AI response as JSON: {str(e)}\n\nResponse received: {response_text[:500]}"
    except Exception as e:
        return None, f"Error during analysis: {str(e)}"

def create_visualization(df, viz_config):
    """Create plotly visualization based on config from LLM with enhanced error handling"""
    try:
        # Make a copy to avoid modifying the original session state dataframe
        df_plot = df.copy()
        
        chart_type = viz_config.get('chart_type', 'line')
        title = viz_config.get('title', 'Chart')
        x_axis = viz_config.get('x_axis')
        y_axis = viz_config.get('y_axis')
        group_by = viz_config.get('group_by')

        # --- Validation ---
        if not x_axis or not y_axis:
            st.warning(f"Skipping chart '{title}': Missing x_axis or y_axis configuration.")
            return None
        
        if x_axis not in df_plot.columns:
            st.warning(f"Skipping chart '{title}': x-axis '{x_axis}' not found in data.")
            return None
        
        y_cols = []
        if isinstance(y_axis, list):
            y_cols = [y for y in y_axis if y in df_plot.columns]
            if not y_cols:
                st.warning(f"Skipping chart '{title}': None of the y-axis columns {y_axis} found in data.")
                return None
        else:
            if y_axis not in df_plot.columns:
                st.warning(f"Skipping chart '{title}': y-axis '{y_axis}' not found in data.")
                return None
            y_cols = [y_axis]

        if group_by and group_by not in df_plot.columns:
            st.warning(f"In chart '{title}': group_by column '{group_by}' not found. Plotting without grouping.")
            group_by = None
        # --- End Validation ---

        # --- Data Type Conversion ---
        # Try to convert x-axis to datetime if it looks like a date
        if df_plot[x_axis].dtype == 'object':
            try:
                df_plot[x_axis] = pd.to_datetime(df_plot[x_axis])
            except Exception:
                pass # If it fails, just plot it as-is

        # Try to convert y-axes to numeric, coercing errors
        for y_col in y_cols:
            df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')

        # Drop rows where numeric conversion failed for all y_cols
        df_plot = df_plot.dropna(subset=y_cols)
        if df_plot.empty:
            st.warning(f"Skipping chart '{title}': No valid numeric data found for y-axis/axes {y_cols}.")
            return None

        # --- Plotting ---
        fig = None
        if chart_type == 'line':
            if len(y_cols) > 1:
                fig = go.Figure()
                for y in y_cols:
                    fig.add_trace(go.Scatter(
                        x=df_plot[x_axis],
                        y=df_plot[y],
                        mode='lines+markers',
                        name=y
                    ))
                fig.update_layout(title=title)
            elif group_by:
                fig = px.line(df_plot, x=x_axis, y=y_cols[0], color=group_by, title=title)
            else:
                fig = px.line(df_plot, x=x_axis, y=y_cols[0], title=title)
                
        elif chart_type == 'bar':
            fig = px.bar(df_plot, x=x_axis, y=y_cols[0], color=group_by, title=title)
                
        elif chart_type == 'scatter':
            fig = px.scatter(df_plot, x=x_axis, y=y_cols[0], color=group_by, title=title)
                
        elif chart_type == 'pie':
            # Pie charts typically need names and values.
            fig = px.pie(df_plot, names=x_axis, values=y_cols[0], title=title)
            
        else:
            st.warning(f"Unsupported chart type '{chart_type}' for chart '{title}'. Defaulting to line chart.")
            fig = px.line(df_plot, x=x_axis, y=y_cols[0], color=group_by, title=title)
        
        # --- Layout Update ---
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif', color=PRIMARY_COLOR),
            title_font_size=16,
            hoverlabel=dict(bgcolor="white"),
            hovermode='x unified'
        )
        fig.update_xaxes(showgrid=True, gridcolor='#E5E9F0', showline=True, linecolor='#E5E9F0')
        fig.update_yaxes(showgrid=True, gridcolor='#E5E9F0', showline=True, linecolor='#E5E9F0')
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization '{title}': {str(e)}")
        return None

# Main App
def main():
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    # Header
    st.markdown(f"""
        <div style='text-align: center; padding: 2rem 0 1rem 0;'>
            <h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>📊 AI Market Trend Storyteller</h1>
            <p style='color: #6B7280; font-size: 1.1rem;'>Transform raw data into compelling business narratives</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        tone = st.selectbox(
            "📝 Narrative Tone",
            ["Analyst", "Journalist", "Executive"],
            help="Choose the style of insights"
        )
        
        st.markdown("---")
        
        st.markdown("### 📋 About")
        st.markdown("""
        This tool uses AI to:
        - 🔍 Detect trends & anomalies
        - 🧮 Analyze correlations
        - 📈 Generate forecasts
        - 📖 Create narratives
        """)
    
    # Main content
    # Check for API key AFTER sidebar is built
    if not api_key:
        st.error("🚨 Could not find GEMINI_API_KEY.")
        st.info("Please create a `.env` file in the app's root directory and add your API key:\n\n`GEMINI_API_KEY='your-api-key-here'`")
        st.stop()
    
    # File upload section
    st.markdown("### 📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with your sales, financial, or performance data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Show data preview
            with st.expander("👁️ Preview Data", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                with col3:
                    numeric_cols = len(df.select_dtypes(include=['int64', 'float64']).columns)
                    st.metric("Numeric Columns", numeric_cols)
                
                st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # Analyze button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Generate Insights", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing your data..."):
                        result, error = analyze_data_with_gemini(df, tone, api_key)
                        
                        if error:
                            st.error(error)
                        else:
                            st.session_state.analysis_result = result
                            st.session_state.analysis_complete = True
                            st.success("✅ Analysis complete!")
                            st.rerun()
            
            # Display results if analysis is complete
            if st.session_state.analysis_complete and st.session_state.analysis_result:
                result = st.session_state.analysis_result
                
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Executive Summary
                st.markdown(f"""
                    <div class='insight-card'>
                        <h3 style='margin-top: 0; color: {SECONDARY_COLOR};'>💡 Executive Summary</h3>
                        <p style='font-size: 1.1rem; line-height: 1.6;'>{result.get('executive_summary', 'N/A')}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Key Metrics
                if 'key_metrics' in result and result['key_metrics']:
                    st.markdown("### 📈 Key Metrics")
                    # Dynamically create columns
                    num_metrics = len(result['key_metrics'])
                    cols = st.columns(num_metrics)
                    for idx, metric in enumerate(result['key_metrics']):
                        with cols[idx]:
                            # Determine change color
                            change_val = str(metric.get('change', ''))
                            change_color = "#6B7280" # Default neutral color
                            if '+' in change_val or (change_val.endswith('%') and not change_val.startswith('-')):
                                change_color = "#10B981" # Green
                            elif '-' in change_val:
                                change_color = "#EF4444" # Red

                            st.markdown(f"""
                                <div class='metric-card'>
                                    <h4 style='color: {SECONDARY_COLOR}; margin: 0;'>{metric.get('metric', 'N/A')}</h4>
                                    <p style='font-size: 1.8rem; font-weight: 600; margin: 0.5rem 0; color: {PRIMARY_COLOR};'>{metric.get('value', 'N/A')}</p>
                                    <p style='color: {change_color}; margin: 0; font-weight: 500;'>{change_val}</p>
                                    <p style='font-size: 0.9rem; color: #6B7280; margin-top: 0.5rem;'>{metric.get('insight', 'N/A')}</p>

                                </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True) # Add some spacing
                
                # Tabs for different sections
                tabs = st.tabs(["📖 Full Narrative", "📊 Visualizations", "🔍 Trends", "🔗 Correlations", "⚠️ Anomalies", "🔮 Forecast"])
                
                # Full Narrative
                with tabs[0]:
                    st.markdown(result.get('narrative', 'No narrative available'))
                
                # Visualizations
                with tabs[1]:
                    if 'visualizations' in result and result['visualizations']:
                        for viz in result['visualizations']:
                            st.markdown(f"#### {viz.get('title', 'Chart')}")
                            if viz.get('description'):
                                st.caption(viz['description'])
                            
                            fig = create_visualization(st.session_state.df, viz)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No visualizations generated")
                
                # Trends
                with tabs[2]:
                    if 'trends' in result and result['trends']:
                        for trend in result['trends']:
                            st.markdown(f"""
                                <div class='insight-card'>
                                    <h4 style='color: {SECONDARY_COLOR}; margin-top: 0;'>{trend.get('title', 'Trend')}</h4>
                                    <p><strong>Description:</strong> {trend.get('description', 'N/A')}</p>
                                    <p><strong>Impact:</strong> {trend.get('impact', 'N/A')}</p>
                                    <p><strong>Time Period:</strong> {trend.get('time_period', 'N/A')}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No trends identified")
                
                # Correlations
                with tabs[3]:
                    if 'correlations' in result and result['correlations']:
                        for corr in result['correlations']:
                            st.markdown(f"""
                                <div class='insight-card'>
                                    <h4 style='color: {SECONDARY_COLOR}; margin-top: 0;'>
                                        {' ↔️ '.join(corr.get('variables', ['Variable 1', 'Variable 2']))}
                                    </h4>
                                    <p><strong>Relationship:</strong> {corr.get('relationship', 'N/A')}</p>
                                    <p><strong>Strength:</strong> <span style='color: {ACCENT_COLOR}; font-weight: 500;'>{corr.get('strength', 'N/A')}</span></p>
                                    <p><strong>Business Implication:</strong> {corr.get('business_implication', 'N/A')}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No correlations identified")
                
                # Anomalies
                with tabs[4]:
                    if 'anomalies' in result and result['anomalies']:
                        for anomaly in result['anomalies']:
                            st.markdown(f"""
                                <div class='insight-card' style='border-left-color: #EF4444;'>
                                    <h4 style='color: #EF4444; margin-top: 0;'>⚠️ {anomaly.get('description', 'Anomaly')}</h4>
                                    <p><strong>Location:</strong> {anomaly.get('location', 'N/A')}</p>
                                    <p><strong>Possible Cause:</strong> {anomaly.get('possible_cause', 'N/A')}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("✅ No significant anomalies detected")
                
                # Forecast
                with tabs[5]:
                    if 'forecast' in result:
                        forecast = result.get('forecast', {})
                        st.markdown(f"""
                            <div class='insight-card' style='border-left-color: {ACCENT_COLOR};'>
                                <h4 style='color: {SECONDARY_COLOR}; margin-top: 0;'>🔮 Next Period Forecast</h4>
                                <p style='font-size: 1.1rem;'><strong>{forecast.get('next_period', 'N/A')}</strong></p>
                                <p><strong>Confidence:</strong> <span style='color: {ACCENT_COLOR}; font-weight: 500;'>{forecast.get('confidence', 'N/A')}</span></p>
                                <p><strong>Key Drivers:</strong> {', '.join(forecast.get('key_drivers', []))}</p>
                                <p><strong>Recommendation:</strong> {forecast.get('recommendation', 'N/A')}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No forecast available")
                
                # Download results
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    json_str = json.dumps(result, indent=2)
                    st.download_button(
                        label="📥 Download Full Analysis (JSON)",
                        data=json_str,
                        file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
        except pd.errors.ParserError:
            st.error("Error: Could not parse the CSV file. Please check the file format.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()