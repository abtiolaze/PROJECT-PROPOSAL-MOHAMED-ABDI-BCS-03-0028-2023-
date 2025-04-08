import json
from flask import Flask, request, render_template, jsonify, send_file, url_for
import pandas as pd
import os
import statsmodels.api as sm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import logging
import traceback

app = Flask(__name__, static_folder="static")

STATIC_FOLDER = "static"
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_expiry')
def check_expiry():
    try:
        predictions_file = os.path.join(STATIC_FOLDER, "predictions.json")
        if not os.path.exists(predictions_file):
            return "No prediction data available. Please upload a CSV first.", 400

        with open(predictions_file, 'r') as f:
            predictions = json.load(f)

        expiry_csv = os.path.join(STATIC_FOLDER, "original_data_with_expiry.csv")
        if not os.path.exists(expiry_csv):
            return "No expiry data available. Please upload a CSV with expiry dates.", 400

        expiry_df = pd.read_csv(expiry_csv)

        # Get all columns that are clearly expiry date columns
        expiry_columns = [col for col in expiry_df.columns if "expiry" in col.lower()]
        
        # Get drug columns (those that don't contain "date" or "expiry" and aren't the main date column)
        drug_columns = [col for col in expiry_df.columns 
                       if not any(x in col.lower() for x in ["date", "expiry"]) 
                       and col.lower() != "date"]

        expiry_info = []
        today = pd.to_datetime("today")

        for drug in drug_columns:
            # Find the exact matching expiry column (e.g., "M01AB" matches "M01AB Expiry Date")
            expiry_col = next((col for col in expiry_columns 
                             if col.lower() == f"{drug.lower()} expiry date"), None)
            
            if not expiry_col:
                continue

            expiry_dates = pd.to_datetime(expiry_df[expiry_col], errors='coerce')

            expired = (expiry_dates < today).sum()
            expiring_soon = ((expiry_dates >= today) & (expiry_dates <= today + pd.Timedelta(days=30))).sum()

            expiry_info.append({
                'drug': drug,
                'expired': int(expired),
                'expiring_soon': int(expiring_soon)
            })

        return render_template('expiry_result.html', expiry_info=expiry_info)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        months = int(request.form.get('months', 12))

        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({"error": "Invalid file. Only CSV files are allowed."}), 400

        df = pd.read_csv(file)
        original_df = df.copy()

        date_column = next((col for col in df.columns if "date" in col.lower()), None)
        if date_column is None:
            return jsonify({"error": "No 'date' column found in the file"}), 400

        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.sort_values(by=date_column).set_index(date_column)

        available_drugs = [col for col in df.columns if not any(x in col.lower() for x in ["expiry", date_column.lower()])]
        if not available_drugs:
            return jsonify({"error": "No valid drug data found in the uploaded file"}), 400

        expiry_csv_path = os.path.join(STATIC_FOLDER, "original_data_with_expiry.csv")
        original_df.to_csv(expiry_csv_path, index=False)

        df = df[available_drugs]
        last_date = df.index.max()
        future_dates = pd.date_range(start=last_date, periods=months + 1, freq='M')[1:]

        predictions = []
        low_stock_warnings = {}

        weight_arima = 0.5
        weight_lstm = 0.5
        lstm_models = {}

        for i, future_date in enumerate(future_dates):
            row = {"date": future_date.strftime('%b-%Y')}

            for drug in available_drugs:
                if df[drug].isnull().all():
                    row[f"{drug}_Hybrid"] = "No data"
                    continue

                avg_demand = df[drug].mean()
                low_stock_threshold = avg_demand * 0.2
                current_stock = df[drug].iloc[-1] if not df[drug].isnull().all() else 0

                if current_stock <= low_stock_threshold:
                    row[f"{drug}_LowStockWarning"] = "⚠️ Low Stock!"
                    low_stock_warnings[drug] = f"Stock below 20% threshold! Current: {round(current_stock)}, Threshold: {round(low_stock_threshold)}"

                arima_forecast, lstm_forecast = ["ARIMA failed"] * months, ["LSTM failed"] * months

                try:
                    arima_model = sm.tsa.ARIMA(df[drug].dropna(), order=(5, 1, 0))
                    arima_fit = arima_model.fit()
                    arima_forecast = arima_fit.forecast(steps=months).tolist()
                except Exception as e:
                    print(f"ARIMA failed for {drug}: {e}")

                try:
                    if drug not in lstm_models:
                        past_steps = 12
                        df_lstm = df[drug].dropna().values.reshape(-1, 1)
                        scaler = MinMaxScaler()
                        df_lstm_scaled = scaler.fit_transform(df_lstm)

                        X_train, y_train = [], []
                        for j in range(past_steps, len(df_lstm_scaled)):
                            X_train.append(df_lstm_scaled[j - past_steps:j])
                            y_train.append(df_lstm_scaled[j])

                        X_train, y_train = np.array(X_train), np.array(y_train)
                        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

                        model = Sequential([
                            LSTM(50, return_sequences=True, input_shape=(past_steps, 1)),
                            LSTM(50),
                            Dense(1)
                        ])
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

                        lstm_models[drug] = (model, scaler)

                    model, scaler = lstm_models[drug]
                    last_data = df_lstm_scaled[-past_steps:].reshape(1, past_steps, 1)
                    lstm_predictions = []

                    for _ in range(months):
                        pred = model.predict(last_data)
                        lstm_predictions.append(pred[0, 0])
                        last_data = np.append(last_data[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

                    lstm_forecast = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten().tolist()
                except Exception as e:
                    print(f"LSTM failed for {drug}: {e}")

                hybrid_forecast = [
                    round((weight_arima * a + weight_lstm * l)) if isinstance(a, (int, float)) and isinstance(l, (int, float)) else "Hybrid failed"
                    for a, l in zip(arima_forecast, lstm_forecast)
                ]

                row[f"{drug}_Hybrid"] = hybrid_forecast[i] if i < len(hybrid_forecast) else "Error"

            predictions.append(row)

        for row in predictions:
            for key, value in row.items():
                if key != "date" and isinstance(value, (int, float)):
                    row[key] = round(value)

        prediction_df = pd.DataFrame(predictions)
        columns_to_keep = ['date'] + [col for col in prediction_df.columns if col.endswith('_Hybrid')]
        csv_export_df = prediction_df[columns_to_keep].copy()
        column_rename = {col: col.replace('_Hybrid', '') for col in csv_export_df.columns if col != 'date'}
        csv_export_df.rename(columns=column_rename, inplace=True)

        csv_path = os.path.join(STATIC_FOLDER, "predictions.csv")
        csv_export_df.to_csv(csv_path, index=False)

        with open("static/predictions.json", "w") as f:
            json.dump(predictions, f)

        return render_template("result.html", predictions=predictions, low_stock_warnings=low_stock_warnings, csv_path=url_for('download_csv'))

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/download_csv')
def download_csv():
    csv_path = os.path.join(STATIC_FOLDER, "predictions.csv")
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

@app.route('/dashboard')
def dashboard():
    try:
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)

        predictions_file = os.path.join(STATIC_FOLDER, "predictions.json")
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)

        df = pd.DataFrame(predictions)
        df['Date'] = pd.to_datetime(df['date'], format='%b-%Y')
        hybrid_columns = [col for col in df.columns if '_Hybrid' in col]

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 8))
        plt.clf()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for i, col in enumerate(hybrid_columns):
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            valid_data = numeric_data[numeric_data.notna()]
            valid_dates = df['Date'][numeric_data.notna()]
            plt.plot(valid_dates, valid_data, 
                     label=col.replace('_Hybrid', ''), 
                     color=colors[i % len(colors)], 
                     marker='o', 
                     linewidth=2, 
                     markersize=8)

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Predicted Sales', fontsize=12)
        plt.title('Drug Sales Forecast Trends', fontsize=15)
        plt.legend(title='Drugs', loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()

        dashboard_path = os.path.join(STATIC_FOLDER, "dashboard.png")
        plt.savefig(dashboard_path, dpi=300)
        plt.close()

        low_stock_warnings = {}
        for row in predictions:
            for drug, value in row.items():
                if '_LowStockWarning' in drug and value:
                    drug_name = drug.replace('_LowStockWarning', '')
                    low_stock_warnings[drug_name] = value

        return render_template("dashboard.html", 
                               predictions=predictions, 
                               low_stock_warnings=low_stock_warnings,
                               dashboard_image=url_for('static', filename='dashboard.png'))

    except Exception as e:
        traceback.print_exc()
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
