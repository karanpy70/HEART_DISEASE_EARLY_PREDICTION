import os

# --- 1. Suppress TensorFlow Warnings (Must be before importing tensorflow) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
from src.data_pipeline import HeartDiseaseDataPipeline

# Ensure 'src' folder is visible
sys.path.append(os.path.join(os.path.dirname(__file__), 'src')) 

def main():
    print("\n" + "="*50)
    print("   ❤️  HEART DISEASE ANALYSIS SYSTEM INITIALIZED")
    print("="*50 + "\n")

    # --- Configuration ---
    # Absolute path to ensure we find the file
    csv_path = r'A:\heart-disease-project\data\raw\heart_statlog_cleveland_hungary_final.csv'
    model_path = 'heart_disease_model.keras'
    preprocessor_path = 'preprocessor.pkl'
    
    # ----------------------------------------------------
    # Check 1: CSV File
    # ----------------------------------------------------
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: CSV file NOT found at: {csv_path}")
        print("   Please check the file path and try again.")
        return
    else:
        print(f"✅ CSV File found.")

    # ----------------------------------------------------
    # Check 2: Saved Model Files (.keras and .pkl)
    # ----------------------------------------------------
    model_ready = os.path.exists(model_path) and os.path.exists(preprocessor_path)
    
    if not model_ready:
        print("\n⚠️  WARNING: Model files not found.")
        print("   (heart_disease_model.keras or preprocessor.pkl is missing)")
        print("   -> Running analysis and will display a placeholder instead of the Confusion Matrix.")
        print("   -> Run 'python train_model.py' to generate these files.")
    else:
        print("✅ Trained Model & Preprocessor found.")

    # Initialize Pipeline
    print("\n--- Step 1: Loading Data ---")
    pipeline = HeartDiseaseDataPipeline(csv_path)
    
    # Load Data
    if not pipeline.load_and_clean_data():
        return
    
    # ----------------------------------------------------
    # Step 2: Evaluate Accuracy and Show All 4 Graphs
    # ----------------------------------------------------
    print("\n--- Step 2: Running Evaluation and Generating All 4 Graphs ---")
    print("   (A single window should open with 4 plots: 3 data graphs + Confusion Matrix. Close it to continue.)")
    
    # This single function now performs evaluation AND plotting.
    # The erroneous call to show_graphs() is removed.
    pipeline.train_and_evaluate()

    print("\n" + "="*50)
    print("   ✅ ANALYSIS COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()