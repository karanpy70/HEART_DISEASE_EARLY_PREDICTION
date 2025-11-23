import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
import re
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- CONFIGURATION ---
st.set_page_config(page_title="Heart Disease Risk System BY KARAN", page_icon="❤️", layout="wide")

# --- TESSERACT SETUP ---
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # Fallback logic for common user installs
    user_path = os.path.expanduser('~')
    possible_paths = [
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.join(user_path, r'AppData\Local\Programs\Tesseract-OCR\tesseract.exe')
    ]
    for p in possible_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break

# -----------------------------------------------------------
# !!! CSV PATH !!!
# Updated to your specific path
CSV_PATH = r"A:\heart-disease-project\data\raw\heart_statlog_cleveland_hungary_final.csv" 
# -----------------------------------------------------------


# --- 1. Load and Cache Data ---
@st.cache_data
def load_data():
    """Loads, cleans, and imputes median values for missing/zero BP and Cholesterol."""
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        st.error(f"Data file not found at: {CSV_PATH}. Please ensure the file is present.")
        return None
        
    # Clean 0 values by replacing them with NaN for median imputation
    df['resting bp s'] = df['resting bp s'].replace(0, np.nan)
    df['cholesterol'] = df['cholesterol'].replace(0, np.nan)
    
    # Impute missing values using the median strategy
    imputer = SimpleImputer(strategy='median')
    cols_to_impute = ['resting bp s', 'cholesterol']
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
    return df

# --- 2. Build and Train Model ---
@st.cache_resource
def build_model(df):
    """Trains the RandomForest model using a standard pipeline (Scaling + One-hot Encoding)."""
    X = df.drop(columns=['target', 'Patient_Name'], errors='ignore')
    y = df['target']
    
    # Define features for pre-processing
    categorical_features = ['chest pain type', 'resting ecg', 'ST slope', 'sex', 'fasting blood sugar', 'exercise angina']
    numerical_features = [c for c in X.columns if c not in categorical_features]

    # Create Preprocessing Pipeline (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('scaler', StandardScaler())]), numerical_features),
            ('cat', Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))]), categorical_features)
        ])

    # Combine with Classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
    model.fit(X, y)
    return model

# --- 3. OCR ENGINE ---
def extract_data_from_image(image):
    """Scans image and extracts key health parameters using regex."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    try:
        text = pytesseract.image_to_string(gray)
    except Exception as e:
        return None, f"Tesseract Error: {str(e)}"
    
    data = {}
    
    # Regex Patterns for all numerical inputs
    patterns = {
        'age': r'(?:age|years)\D*(\d{2,3})',
        'cholesterol': r'(?:cholesterol|chol|total)\D*(\d{3})',
        'resting bp s': r'(\d{2,3})\s*/\s*(\d{2,3})', # Captures systolic BP
        'max heart rate': r'(?:max|maximum)\D*heart\D*rate\D*(\d{2,3})',
        'oldpeak': r'(?:oldpeak|depression)\D*(\d+\.?\d*)',
        'ST slope': r'slope\D*(\d)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # For BP, we only take the first group (Systolic)
            val = match.group(1) if key != 'resting bp s' else match.group(1)
            data[key] = float(val) if '.' in val else int(val)

    # Fasting Blood Sugar (requires conditional check)
    glu_match = re.search(r'(?:glucose|sugar|bs)\D*(\d{2,3})', text, re.IGNORECASE)
    if glu_match:
        glucose = int(glu_match.group(1))
        # fasting blood sugar is a binary column (1 if > 120, 0 otherwise)
        data['fasting blood sugar'] = 1 if glucose > 120 else 0
        
    return data, text

# --- 4. MAIN INTERFACE ---
def main():
    # Load Data & Model (Cached)
    df = load_data()
    if df is None:
        st.stop()
    model = build_model(df)

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose Mode:", 
        ["📝 Manual Risk Calculator", "🔍 Search Database", "📸 Scan New Report"]
    )

    # Initialize session state for OCR data persistence
    if 'ocr_defaults' not in st.session_state:
        st.session_state['ocr_defaults'] = {
            'age': 50, 'cholesterol': 200, 'resting bp s': 120, 'fasting blood sugar': 0,
            'max heart rate': 150, 'oldpeak': 1.0, 'ST slope': 2
        }

    # Helper function to display results in all modes
    def display_risk_result(prob):
        risk = prob * 100
        st.divider()
        st.subheader("❤️ Heart Risk Analysis BY KARAN")
        
        col_res1, col_res2 = st.columns([1, 3])
        
        with col_res1:
            if risk < 30:
                st.metric("Risk Probability", f"{risk:.1f}%", "Low Risk")
                st.balloons()
            elif risk < 70:
                st.metric("Risk Probability", f"{risk:.1f}%", "Moderate Risk", delta_color="off")
            else:
                st.metric("Risk Probability", f"{risk:.1f}%", "High Risk", delta_color="inverse")
        
        with col_res2:
            if risk > 70:
                st.error("🚨 High probability of heart disease detected. Immediate medical consultation recommended.")
            elif risk > 30:
                st.warning("⚠️ Moderate risk detected. Focus on lifestyle changes and monitoring.")
            else:
                st.success("✅ Low risk detected. Keep maintaining a healthy lifestyle!")


    # ==========================================
    # MODE 1: MANUAL RISK CALCULATOR (NEW MODE)
    # ==========================================
    if app_mode == "📝 Manual Risk Calculator":
        st.title("📝 Manual Heart Risk Calculator")
        st.markdown("Enter patient vitals manually to check heart disease probability.")
        st.divider()

        with st.form("manual_form"):
            st.subheader("Patient Vitals")
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Age", min_value=1, max_value=120, value=50)
            sex = c2.selectbox("Sex", ["Male", "Female"])
            cp_input = c3.selectbox("Chest Pain Type", ["1: Typical Angina", "2: Atypical Angina", "3: Non-Anginal Pain", "4: Asymptomatic"])
            
            st.subheader("Blood & Pressure Readings")
            c4, c5, c6 = st.columns(3)
            bp = c4.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
            chol = c5.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            bs_input = c6.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])

            st.subheader("ECG & Stress Test Data")
            c7, c8, c9 = st.columns(3)
            ecg_input = c7.selectbox("Resting ECG Results", ["0: Normal", "1: ST-T Wave Abnormality", "2: LV Hypertrophy"])
            max_hr = c8.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
            ex_ang_input = c9.selectbox("Exercise Induced Angina?", ["No", "Yes"])
            
            c10, c11 = st.columns(2)
            oldpeak = c10.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
            slope_options = ["1: Upsloping", "2: Flat", "3: Downsloping"]
            slope_input = c11.selectbox("ST Slope", slope_options, index=1)

            submit_manual = st.form_submit_button("❤️ Calculate Risk Now")

        if submit_manual:
            # Prepare input data for the model
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [1 if sex == "Male" else 0],
                'chest pain type': [int(cp_input.split(":")[0])],
                'resting bp s': [bp],
                'cholesterol': [chol],
                'fasting blood sugar': [1 if bs_input == "Yes" else 0],
                'resting ecg': [int(ecg_input.split(":")[0])],
                'max heart rate': [max_hr],
                'exercise angina': [1 if ex_ang_input == "Yes" else 0],
                'oldpeak': [oldpeak],
                'ST slope': [int(slope_input.split(":")[0])]
            })
            
            prob = model.predict_proba(input_data)[0][1]
            display_risk_result(prob)


    # ==========================================
    # MODE 2: SEARCH DATABASE
    # ==========================================
    elif app_mode == "🔍 Search Database":
        st.title("🏥 Patient Database Search")
        st.markdown("Search for an existing patient record in the dataset.")
        st.divider()
        
        search_name = st.text_input("Enter Patient Name:", placeholder="e.g., Regler, Adlam...")

        if search_name:
            patient_record = df[df['Patient_Name'].str.contains(search_name, case=False, na=False)]
            if not patient_record.empty:
                patient_row = patient_record.iloc[[0]]
                st.success(f"Record Found: **{patient_row['Patient_Name'].values[0]}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📄 Medical Report")
                    display_data = patient_row.drop(columns=['Patient_Name', 'target']).T
                    display_data.columns = ['Value']
                    st.dataframe(display_data, height=400)

                with col2:
                    X_new = patient_row.drop(columns=['target', 'Patient_Name'], errors='ignore')
                    prob = model.predict_proba(X_new)[0][1]
                    display_risk_result(prob)
            else:
                st.warning("Patient not found.")

    # ==========================================
    # MODE 3: SCAN NEW REPORT (UPDATED FLOW)
    # ==========================================
    elif app_mode == "📸 Scan New Report":
        st.title("📸 AI Medical Report Scanner")
        st.markdown("Upload a lab report image, click 'Analyze Report Only' to extract data, and then validate the form before prediction.")
        st.divider()

        uploaded_file = st.file_uploader("Upload Report (Image: PNG, JPG)", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Read image bytes
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            col_img, col_btn = st.columns([1, 2])
            with col_img:
                st.image(image, caption='Uploaded Report', width=250)
            
            # --- STEP 1: ANALYSIS BUTTON ---
            with col_btn:
                st.markdown("### Step 1: Read Image Text")
                if st.button("🔍 Analyze Report Only", key="scan_button"):
                    with st.spinner('Reading text from image...'):
                        extracted_data, raw_text = extract_data_from_image(image)
                        
                        if extracted_data is not None and len(extracted_data) > 0:
                            st.success(f"✅ Analysis Complete! Found {len(extracted_data)} values.")
                            # Update Session State with OCR data
                            st.session_state['ocr_defaults'].update(extracted_data)
                            st.session_state['raw_text'] = raw_text
                            st.info("👇 **Step 2:** Review the data below and click '🚀 Analyze Risk Level'")
                        else:
                            st.error("Could not extract numerical data. Please fill the form manually.")
                            st.session_state['raw_text'] = raw_text

            # Show raw OCR text if available
            if 'raw_text' in st.session_state:
                with st.expander("View Raw OCR Text"):
                    st.text(st.session_state['raw_text'])

        # --- STEP 2: VALIDATE DATA & PREDICT FORM ---
        st.divider()
        st.subheader("📝 Step 2: Validate Data & Predict")
        
        # Get defaults from session state to populate the form
        defaults = st.session_state['ocr_defaults']
        
        with st.form("ocr_prediction_form"):
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Age", value=defaults.get('age', 50))
            bp = c2.number_input("Resting BP", value=defaults.get('resting bp s', 120))
            chol = c3.number_input("Cholesterol", value=defaults.get('cholesterol', 200))
            
            c4, c5, c6 = st.columns(3)
            sex_input = c4.selectbox("Sex", ["Male", "Female"])
            cp_input = c5.selectbox("Chest Pain Type", ["1: Typical", "2: Atypical", "3: Non-Anginal", "4: Asymptomatic"])
            bs_index = defaults.get('fasting blood sugar', 0)
            bs_input = c6.selectbox("Fasting BS > 120?", ["No", "Yes"], index=bs_index)
            
            c7, c8, c9 = st.columns(3)
            ecg_input = c7.selectbox("Resting ECG", ["0: Normal", "1: ST-T Abnormality", "2: LV Hypertrophy"])
            max_hr = c8.number_input("Max Heart Rate", value=defaults.get('max heart rate', 150))
            ex_ang_input = c9.selectbox("Exercise Induced Angina?", ["No", "Yes"])
            
            c10, c11 = st.columns(2)
            oldpeak = c10.number_input("Oldpeak (ST Depression)", value=float(defaults.get('oldpeak', 1.0)), step=0.1)
            # Match the value from the dictionary to the index of the selectbox options
            slope_options_map = {1: 0, 2: 1, 3: 2} 
            slope_value = defaults.get('ST slope', 2)
            slope_index = slope_options_map.get(slope_value, 1) # Default to 2: Flat
            
            slope_input = c11.selectbox("ST Slope", ["1: Upsloping", "2: Flat", "3: Downsloping"], index=slope_index)
            
            submit_scan = st.form_submit_button("🚀 Analyze Risk Level")

        if submit_scan:
            # Prepare Data
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [1 if sex_input == "Male" else 0],
                'chest pain type': [int(cp_input.split(":")[0])],
                'resting bp s': [bp],
                'cholesterol': [chol],
                'fasting blood sugar': [1 if bs_input == "Yes" else 0],
                'resting ecg': [int(ecg_input.split(":")[0])],
                'max heart rate': [max_hr],
                'exercise angina': [1 if ex_ang_input == "Yes" else 0],
                'oldpeak': [oldpeak],
                'ST slope': [int(slope_input.split(":")[0])]
            })
            
            prob = model.predict_proba(input_data)[0][1]
            display_risk_result(prob)


if __name__ == "__main__":
    main()