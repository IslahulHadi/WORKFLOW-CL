FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir pandas numpy scikit-learn mlflow==2.19.0 dagshub joblib matplotlib

# Copy files
# modelling_ci.py dipakai (bukan modelling.py) karena image publik ini
# tidak punya akses ke DagsHub credentials saat dijalankan oleh siapa pun
COPY modelling_ci.py .
COPY OnlineRetail_RFM.csv .

# Run modelling -> menghasilkan kmeans_rfm_model.pkl & rfm_scaler.pkl
CMD ["python", "modelling_ci.py"]
