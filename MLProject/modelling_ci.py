# -*- coding: utf-8 -*-
"""modelling_ci.py

Versi CI/CD dari modelling.py — tanpa koneksi DagsHub
agar bisa jalan di GitHub Actions tanpa credentials.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. LOAD DATA RFM
# ============================================================
print("=" * 60)
print("STEP 1: Load Data RFM")
print("=" * 60)

rfm = pd.read_csv('OnlineRetail_RFM.csv', index_col='CustomerID')
print(f"Data RFM loaded: {rfm.shape[0]} customers, {rfm.shape[1]} fitur")

# ============================================================
# 2. SCALING
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

# ============================================================
# 3. TRAINING MODEL K-MEANS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Training Model K-Means (K=4)")
print("=" * 60)

N_CLUSTERS = 4
RANDOM_STATE = 42

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10,
    max_iter=300
)
labels = kmeans.fit_predict(rfm_scaled_df)
rfm['Cluster'] = labels

print(f"Model KMeans dengan K={N_CLUSTERS} berhasil di-train.")
print(f"Inertia (SSE): {kmeans.inertia_:.2f}")

# ============================================================
# 4. EVALUASI MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Evaluasi Model")
print("=" * 60)

sil_score = silhouette_score(rfm_scaled_df, labels)
print(f"Silhouette Score: {sil_score:.4f}")

print("\nRata-rata RFM per Cluster:")
print(rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean())

print("\nDistribusi Customer per Cluster:")
print(rfm['Cluster'].value_counts().sort_index())

# ============================================================
# 5. SIMPAN MODEL DAN SCALER
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Simpan Model dan Scaler")
print("=" * 60)

joblib.dump(kmeans, 'kmeans_rfm_model.pkl')
joblib.dump(scaler, 'rfm_scaler.pkl')

print("Model tersimpan: kmeans_rfm_model.pkl")
print("Scaler tersimpan: rfm_scaler.pkl")
print("\n✅ CI Training selesai!")
