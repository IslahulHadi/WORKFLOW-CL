# -*- coding: utf-8 -*-
"""modelling.py

Proyek Customer Segmentation oleh Ahul (IslahulHadi)

Script ini melakukan TRAINING MODEL K-Means Clustering
menggunakan data RFM yang sudah di-preprocessing.

Input : OnlineRetail_RFM.csv (hasil dari preprocessing)
Output: Model K-Means (.pkl), Scaler (.pkl), Log ke MLflow/DagsHub
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import dagshub
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. LOAD DATA RFM YANG SUDAH DI-PREPROCESSING
# ============================================================
print("=" * 60)
print("STEP 1: Load Data RFM")
print("=" * 60)

rfm = pd.read_csv('OnlineRetail_RFM.csv', index_col='CustomerID')
print(f"Data RFM loaded: {rfm.shape[0]} customers, {rfm.shape[1]} fitur")
print(f"Kolom: {list(rfm.columns)}")
print(f"\nStatistik RFM:")
print(rfm.describe())

# ============================================================
# 2. SCALING / NORMALISASI
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Scaling Data dengan StandardScaler")
print("=" * 60)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

rfm_scaled_df = pd.DataFrame(
    rfm_scaled,
    index=rfm.index,
    columns=['Recency', 'Frequency', 'Monetary']
)

print("Scaling selesai.")
print(f"Mean setelah scaling: {rfm_scaled_df.mean().values}")
print(f"Std setelah scaling:  {rfm_scaled_df.std().values}")

# ============================================================
# 3. TRAINING MODEL K-MEANS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Training Model K-Means (K=4)")
print("=" * 60)

# Hyperparameter
N_CLUSTERS = 4
RANDOM_STATE = 42

# Training
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10,
    max_iter=300
)
labels = kmeans.fit_predict(rfm_scaled_df)

# Assign cluster ke RFM
rfm['Cluster'] = labels

print(f"Model KMeans dengan K={N_CLUSTERS} berhasil di-train.")
print(f"Inertia (SSE): {kmeans.inertia_:.2f}")

# ============================================================
# 4. EVALUASI MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Evaluasi Model")
print("=" * 60)

# Silhouette Score
sil_score = silhouette_score(rfm_scaled_df, labels)
print(f"Silhouette Score: {sil_score:.4f}")

# Rata-rata RFM per cluster
print("\nRata-rata RFM per Cluster:")
print(rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean())

# Distribusi cluster
print("\nDistribusi Customer per Cluster:")
print(rfm['Cluster'].value_counts().sort_index())

# ============================================================
# 5. LOG KE MLFLOW / DAGSHUB
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Log ke MLflow / DagsHub")
print("=" * 60)

dagshub.init(repo_owner='IslahulHadi', repo_name='Retail-OpsML-Ahul', mlflow=True)

with mlflow.start_run(run_name="KMeans_RFM_Segmentation"):
    # Log parameters
    mlflow.log_param("n_clusters", N_CLUSTERS)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("n_init", 10)
    mlflow.log_param("max_iter", 300)
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("features", "Recency, Frequency, Monetary")

    # Log metrics
    mlflow.log_metric("silhouette_score", sil_score)
    mlflow.log_metric("inertia", kmeans.inertia_)
    mlflow.log_metric("n_customers", rfm.shape[0])

    # Log model
    mlflow.sklearn.log_model(kmeans, "kmeans_rfm_model")

    print("Berhasil log ke MLflow/DagsHub!")

# ============================================================
# 6. SIMPAN MODEL DAN SCALER
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Simpan Model dan Scaler")
print("=" * 60)

joblib.dump(kmeans, 'kmeans_rfm_model.pkl')
joblib.dump(scaler, 'rfm_scaler.pkl')

print("Model tersimpan: kmeans_rfm_model.pkl")
print("Scaler tersimpan: rfm_scaler.pkl")
print("\n✅ Training selesai!")
