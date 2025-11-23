import pandas as pd
import numpy as np
import joblib
import os

# Scikit-learn for Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# TensorFlow / Keras for Neural Network (CNN)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
CSV_PATH = r"A:\heart-disease-project\data\raw\heart_statlog_cleveland_hungary_final.csv"
MODEL_SAVE_PATH = "heart_disease_model.keras"  # Saving as Keras file
PREPROCESSOR_SAVE_PATH = "preprocessor.pkl"     # Saving the scaler/encoder separately

def train_cnn_model():
    print(f"Loading data from: {CSV_PATH}")
    
    # 1. Load Data
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("❌ Error: CSV file not found.")
        return

    # 2. Data Cleaning
    if 'Patient_Name' in df.columns:
        df = df.drop(columns=['Patient_Name'])
        
    df['resting bp s'] = df['resting bp s'].replace(0, np.nan)
    df['cholesterol'] = df['cholesterol'].replace(0, np.nan)
    
    imputer = SimpleImputer(strategy='median')
    cols_to_impute = ['resting bp s', 'cholesterol']
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    # 3. Define Features (X) and Target (y)
    X = df.drop(columns=['target'])
    y = df['target']

    # 4. Preprocessing Pipeline
    # We must process data BEFORE feeding it to the Neural Network
    categorical_features = ['chest pain type', 'resting ecg', 'ST slope', 'sex', 'fasting blood sugar', 'exercise angina']
    numerical_features = [c for c in X.columns if c not in categorical_features]

    # Neural Networks require scaling (StandardScaler) for all inputs
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
        ])

    # Fit and Transform the data
    print("Preprocessing data...")
    X_processed = preprocessor.fit_transform(X)
    
    # Save the preprocessor immediately so we can use it in the App
    joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)
    print(f"Preprocessor saved to {PREPROCESSOR_SAVE_PATH}")

    # 5. Reshape for CNN
    # CNNs expect 3D input: (Samples, Features, 1)
    X_cnn = X_processed.reshape(X_processed.shape[0], X_processed.shape[1], 1)
    input_shape = (X_cnn.shape[1], 1)

    # 6. Build 1D CNN Architecture
    print("Building CNN Architecture...")
    model = Sequential()

    # First Convolutional Block
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Second Convolutional Block
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Third Convolutional Block
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # Binary classification (0 or 1)

    # Compile
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # 7. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.15, random_state=42, stratify=y)

    # 8. Train
    print("Starting Training (This may take a moment)...")
    
    # Callbacks to ensure high accuracy without wasting time
    early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(
        X_train, y_train, 
        epochs=150,           # High epochs to aim for convergence
        batch_size=16,        # Smaller batch size often generalizes better
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # 9. Evaluation
    print("\n--- Final Evaluation ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Predictions for Confusion Matrix
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'CNN Confusion Matrix (Acc: {accuracy*100:.2f}%)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # 10. Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ CNN Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_cnn_model()