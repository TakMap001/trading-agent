# model_training_gold.py — for Gold

import pandas as pd
import xgboost as xgb
import shap
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

df = pd.read_csv("data/gold_data.csv")
df = df[df['signal'].isin([0, 1])]

if len(df) < 100:
    print("⚠️ Not enough data to train for XAUUSDm.")
else:
    X = df.drop(columns=['time', 'signal'], errors='ignore')
    y = df['signal']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    win_rate = round((y_pred == y_test).mean() * 100, 2)
    profit_factor = round((y_pred == y_test).sum() / max(1, (y_pred != y_test).sum()), 2)

    print("\n🧠 Training model for XAUUSDm...")
    print(report)
    print(f"✅ Win Rate: {win_rate}%")
    print(f"💰 Profit Factor: {profit_factor}")

    joblib.dump(model, "models/XAUUSDm_model.pkl")
    with open("models/XAUUSDm_features.txt", 'w') as f:
        for col in X.columns:
            f.write(col + '\n')

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig("plots/shap_XAUUSDm.png")
    plt.clf()
