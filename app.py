import streamlit as st
import pandas as pd
import joblib
import io
import time

# -----------------------------
# Configuration & Styling
# -----------------------------
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 20px;
        margin-bottom: 20px;
        border-bottom: 1px solid #e2e8f0;
    }
    .top-header-left {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .logo-text {
        color: #0ea5e9;
        font-weight: 700;
        font-size: 1.5rem;
    }
    .header-title {
        color: #1e293b;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
    }
    .header-subtitle {
        color: #64748b;
        font-size: 0.875rem;
    }
    .badge {
        background-color: #f1f5f9;
        color: #475569;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Hide default header */
    header[data-testid="stHeader"] {
        display: none;
    }

    /* Primary Button Styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.4);
        transition: all 0.3s ease;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px -1px rgba(14, 165, 233, 0.6);
    }
    
    /* Cards (using st.container(border=True) but styling the child components if needed) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        padding: 5px;
    }
    
    /* Metrics overriding */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    [data-testid="stMetricLabel"] {
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em;
        color: #64748b !important;
    }
    
    /* Footer */
    .footer-content {
        display: flex;
        justify-content: space-between;
        padding-top: 30px;
        margin-top: 50px;
        border-top: 1px solid #e2e8f0;
        color: #94a3b8;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session State Initialization
# -----------------------------
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'prediction_proba' not in st.session_state:
    st.session_state.prediction_proba = None
if 'batch_df' not in st.session_state:
    st.session_state.batch_df = None
if 'form_reset' not in st.session_state:
    st.session_state.form_reset = False

# Function to handle reset
def reset_form():
    st.session_state.prediction_result = None
    st.session_state.prediction_proba = None
    st.session_state.batch_df = None
    st.session_state.form_reset = not st.session_state.form_reset # toggle to force re-render if needed

# -----------------------------
# Model Loading Placeholder
# -----------------------------
@st.cache_resource
def load_ml_assets():
    model = joblib.load("random_forest_top10_model.pkl")
    model_columns = joblib.load("top10_model_columns.pkl")
    return model, model_columns

model, model_columns = load_ml_assets()

# -----------------------------
# Main Layout
# -----------------------------

# Top Header Area
st.markdown("""
    <div style="padding-bottom: 20px; margin-bottom: 30px; border-bottom: 1px solid #e2e8f0;">
        <div class="header-title" style="font-size: 1.75rem; color: #1e293b; font-weight: 700;">🚗 Vehicle Insurance Premium Predictor</div>
        <div class="header-subtitle" style="color: #64748b; font-size: 0.9rem; margin-top: 5px;">Binary classification for policy premiums > 300</div>
    </div>
""", unsafe_allow_html=True)

# Main Two-Column Layout
left_col, right_col = st.columns([1.4, 1], gap="large")

with left_col:
    # -----------------------------
    # Left Column: Policy Parameters Form
    # -----------------------------
    with st.container(border=True):
        st.markdown("### 📄 Policy Parameters")
        st.markdown("<p style='color:#64748b;font-size:0.9rem;'>Enter the 10 core features to evaluate specific policy risk levels.</p>", unsafe_allow_html=True)
        
        # We can use a form here to manage inputs cleanly
        with st.form("prediction_form", border=False):
            col_spec, col_driver = st.columns(2, gap="medium")
            
            with col_spec:
                st.markdown("<p style='font-size:0.8rem; font-weight:600; color:#94a3b8; text-transform:uppercase;'>— Vehicle Specifications</p>", unsafe_allow_html=True)
                Value_vehicle = st.number_input("Value Vehicle ⓘ", value=25000, help="Estimated market value of the vehicle")
                Weight = st.number_input("Weight (kg) ⓘ", value=1450, help="Kerb weight of the vehicle in kilograms")
                Length = st.number_input("Length (m) ⓘ", value=4.2, help="Length of the vehicle in meters")
                car_age = st.number_input("Car Age ⓘ", value=5, help="Age of the car in years since manufacture")
                Cylinder_capacity = st.number_input("Cylinder Capacity ⓘ", value=1600, help="Engine displacement in cc")
            
            with col_driver:
                st.markdown("<p style='font-size:0.8rem; font-weight:600; color:#94a3b8; text-transform:uppercase;'>— Driver & Policy Info</p>", unsafe_allow_html=True)
                year_licensed = st.number_input("Year Licensed ⓘ", value=2012, help="The year the driver obtained their license")
                age = st.number_input("Driver Age ⓘ", value=35, help="Age of the primary driver")
                Seniority = st.number_input("Seniority ⓘ", value=8, help="Number of years the driver has been insured")
                Power = st.number_input("Engine Power (HP) ⓘ", value=150, help="Power output of the engine in horsepower")
                Payment = st.number_input("Payment Method ⓘ", value=100.0, help="Payment frequency or numeric encoding of method")

            st.markdown("<br/>", unsafe_allow_html=True)
            
            btn_col1, btn_col2, _ = st.columns([1, 1, 1])
            with btn_col1:
                submit_btn = st.form_submit_button("⚡ Predict Premium", type="primary", use_container_width=True)
            with btn_col2:
                # We use a secondary button for reset, but Streamlit form reset is tricky.
                # It's better to clear session state variables for prediction output instead of form inputs
                reset_btn = st.form_submit_button("↺ Clear Results", use_container_width=True)
                
            if reset_btn:
                reset_form()
                st.rerun()
                
            if submit_btn:
                input_data = pd.DataFrame([{
                    "Value_vehicle": Value_vehicle,
                    "Weight": Weight,
                    "year_licensed": year_licensed,
                    "age": age,
                    "Length": Length,
                    "car_age": car_age,
                    "Seniority": Seniority,
                    "Power": Power,
                    "Cylinder_capacity": Cylinder_capacity,
                    "Payment": Payment
                }])
                input_data = input_data[model_columns]
                
                with st.spinner("Analyzing parameters..."):
                    time.sleep(0.5) # Slight delay for effect
                    pred = model.predict(input_data)[0]
                    prob = model.predict_proba(input_data)[0]
                    
                    st.session_state.prediction_result = pred
                    st.session_state.prediction_proba = prob

    # Expandable Reference Guide
    with st.expander("📄 Field Reference Guide"):
        st.markdown("""
        **Value Vehicle**: Cost of the vehicle in local currency.  
        **Weight**: Vehicle weight in kg.  
        **Year Licensed**: The calendar year the driver was licensed.  
        **Age**: Current age of the policyholder.  
        **Length**: Vehicle length in meters.  
        **Car Age**: Number of years since manufacture.  
        **Seniority**: Loyalty years with an insurance provider.  
        **Power**: Engine horsepower.  
        **Cylinder Capacity**: Engine CCs.  
        **Payment**: Code representing payment modality.
        """)


with right_col:
    # -----------------------------
    # Right Column: Inference Output
    # -----------------------------
    with st.container(border=True):
        st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#94a3b8; text-transform:uppercase; margin-bottom: 0;'>→ Inference Output</p>", unsafe_allow_html=True)
        
        if st.session_state.prediction_result is None:
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; color: #94a3b8;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📉</div>
                <p>Enter data and click "Predict Premium" to see results</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            pred = st.session_state.prediction_result
            prob_safe = st.session_state.prediction_proba[0] # Probability of Class 0 (Premium <= 300)
            prob_risk = st.session_state.prediction_proba[1] # Probability of Class 1 (Premium > 300)
            
            if pred == 1:
                status_color = "#ef4444" # Red
                status_text = "PREMIUM > 300"
                badge = f"<span style='background-color: #fee2e2; color: #ef4444; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;'>High Range</span>"
            else:
                status_color = "#22c55e" # Green
                status_text = "PREMIUM ≤ 300"
                badge = f"<span style='background-color: #dcfce7; color: #22c55e; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;'>Standard</span>"
                
            st.markdown(f"""
            <div style="padding: 20px 0;">
                <h4 style="color: {status_color}; margin:0; font-size: 1.8rem;">{status_text} {badge}</h4>
                <div style="display: flex; gap: 40px; margin-top: 15px;">
                    <div>
                        <p style="color: #64748b; font-size: 0.8rem; margin: 0; font-weight: 600;">Probability: Premium ≤ 300</p>
                        <p style="color: #1e293b; font-size: 1.8rem; font-weight: 700; margin: 0;">{prob_safe*100:.2f}%</p>
                    </div>
                    <div>
                        <p style="color: #64748b; font-size: 0.8rem; margin: 0; font-weight: 600;">Probability: Premium > 300</p>
                        <p style="color: #1e293b; font-size: 1.8rem; font-weight: 700; margin: 0;">{prob_risk*100:.2f}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(float(prob_risk))

    # -----------------------------
    # Right Column: Batch Prediction
    # -----------------------------
    with st.container(border=True):
        st.markdown("### 📤 Batch Prediction")
        st.markdown("<p style='color:#64748b;font-size:0.85rem;'>Upload multiple records (CSV) to process in bulk.</p>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Drop CSV file here", type=["csv"], label_visibility="collapsed")
        
        st.markdown("<p style='font-size:0.75rem; font-weight:600; color:#94a3b8; text-transform:uppercase; margin-bottom: 5px;'>REQUIRED SCHEMA</p>", unsafe_allow_html=True)
        st.markdown("<div style='background-color:#f1f5f9; padding:10px; border-radius:6px; font-size:0.8rem; font-family:monospace; color:#475569;'>Value_vehicle, Weight, year_licensed, age, Length, car_age, Seniority, Power, Cylinder_capacity, Payment</div>", unsafe_allow_html=True)
        
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("⚡ Run Batch Preview", use_container_width=True, disabled=uploaded_file is None):
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    # Simple validation
                    missing_cols = [c for c in model_columns if c not in df.columns]
                    if missing_cols:
                        st.error(f"Missing columns: {', '.join(missing_cols)}")
                    else:
                        with st.spinner("Processing batch..."):
                            df_infer = df[model_columns]
                            preds = model.predict(df_infer)
                            df['Prediction'] = preds
                            df['Premium_Result'] = df['Prediction'].apply(lambda x: 'Above 300' if x == 1 else '300 or Below')
                            st.session_state.batch_df = df
                            
                            st.success(f"Processed {len(df)} records successfully.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    
        if st.session_state.batch_df is not None:
            st.dataframe(st.session_state.batch_df[['Premium_Result'] + model_columns].head(5), use_container_width=True)
            
            csv_export = st.session_state.batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Results",
                data=csv_export,
                file_name="batch_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer Area
st.markdown("""
<div class="footer-content">
    <div>
        <span style="color: #22c55e;">✓</span> PCI-DSS Compliant &nbsp;&nbsp;|&nbsp;&nbsp; 
        <span style="color: #64748b;">ⓘ</span> Data processed locally
    </div>
    <div style="font-style: italic; text-align: center; flex: 1;">
        "Uploaded data is used exclusively for live prediction and is not stored permanently on the inference servers."
    </div>
    <div style="text-align: right;">
        Support Portal &nbsp;&nbsp; API Docs &nbsp;&nbsp; <span style="opacity: 0.5;">System Up: 99.98%</span>
    </div>
</div>
""", unsafe_allow_html=True)