FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir pandas numpy scikit-learn mlflow==2.19.0 dagshub joblib matplotlib

# Copy files
COPY modelling.py .
COPY OnlineRetail_RFM.csv .

# Run modelling
CMD ["python", "modelling.py"]
