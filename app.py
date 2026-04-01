import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, mean_squared_error,
                           confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up page
st.set_page_config(page_title="Market Analysis (Stocks & Forex)", layout="wide")
st.title("🎯 Realistic Hybrid Model: No Look-ahead Bias")

# Shared functions
def load_data(ticker, start_date, end_date):
    """Load market data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), progress=False)
        if data.empty:
            return None
        
        # Flatten multi-index columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        return data.dropna()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data_no_leakage(data):
    """
    PROPER feature engineering WITHOUT look-ahead bias
    All features use ONLY data available BEFORE the prediction day
    """
    data = data.copy()
    
    # SHIFT ALL FEATURES BY 1 DAY TO PREVENT LOOK-AHEAD
    # This ensures we only use information known at the start of each day
    
    # Previous day's OHLC data (KNOWN at prediction time)
    data['Prev_Close'] = data['Close'].shift(1)
    data['Prev_Open'] = data['Open'].shift(1)
    data['Prev_High'] = data['High'].shift(1)
    data['Prev_Low'] = data['Low'].shift(1)
    
    # Basic price features using PREVIOUS day's data only
    data['Prev_Price_Range'] = (data['Prev_High'] - data['Prev_Low']) / data['Prev_Close']
    data['Prev_Body_Size'] = abs(data['Prev_Close'] - data['Prev_Open']) / data['Prev_Open']
    
    # Previous day's candle patterns
    data['Prev_Is_Bullish'] = (data['Prev_Close'] > data['Prev_Open']).astype(int)
    
    # Momentum using historical data only (shifted to use only past data)
    for period in [1, 3, 5, 10, 20]:
        data[f'Momentum_{period}d'] = data['Close'].pct_change(period).shift(1)
    
    # Volatility using historical data only
    for period in [5, 10, 20]:
        data[f'Return_Volatility_{period}d'] = data['Close'].pct_change().rolling(period).std().shift(1)
        # High-Low volatility using previous days only
        data[f'HL_Volatility_{period}d'] = (data['High'] - data['Low']).rolling(period).mean().shift(1) / data['Close'].shift(1)
    
    # Gap from previous day's close to today's open (known at market open)
    data['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    data['Gap_Size'] = abs(data['Gap'])
    data['Gap_Direction'] = np.sign(data['Gap'])
    
    # Moving averages (as of previous day close)
    for period in [5, 10, 20, 50]:
        ma = data['Close'].rolling(period).mean().shift(1)
        data[f'Close_vs_MA{period}'] = data['Prev_Close'] / ma
        data[f'MA{period}_Slope'] = ma.pct_change()
    
    # Distance from recent highs/lows (as of previous day)
    for period in [5, 10, 20]:
        data[f'Dist_From_{period}d_High'] = (data['High'].rolling(period).max().shift(1) - data['Prev_Close']) / data['Prev_Close']
        data[f'Dist_From_{period}d_Low'] = (data['Prev_Close'] - data['Low'].rolling(period).min().shift(1)) / data['Prev_Close']
    
    # Volatility ratio (current vs historical)
    avg_range_10d = (data['High'] - data['Low']).rolling(10).mean().shift(1)
    current_range = data['Prev_High'] - data['Prev_Low']
    data['Volatility_Ratio'] = current_range / avg_range_10d
    
    # Previous day's closing price position relative to its range
    prev_range = data['Prev_High'] - data['Prev_Low']
    data['Prev_Position_In_Range'] = np.where(
        prev_range > 0,
        (data['Prev_Close'] - data['Prev_Low']) / prev_range,
        0.5
    )
    
    # Previous day's shadows
    prev_max = pd.concat([data['Prev_Close'], data['Prev_Open']], axis=1).max(axis=1)
    prev_min = pd.concat([data['Prev_Close'], data['Prev_Open']], axis=1).min(axis=1)
    data['Prev_Upper_Shadow'] = (data['Prev_High'] - prev_max) / data['Prev_Close']
    data['Prev_Lower_Shadow'] = (prev_min - data['Prev_Low']) / data['Prev_Close']
    
    # Target: tomorrow's direction (1 if up, 0 if down)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Drop rows with NaN (from shifts and rolling calculations)
    data.dropna(inplace=True)
    
    return data

def walk_forward_validation(X, y, n_splits=5, n_estimators=100, max_depth=6):
    """
    Time-series cross-validation for realistic performance estimation
    """
    if len(X) < n_splits * 20:  # Need enough data
        n_splits = max(2, len(X) // 20)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    models = {}
    metrics_history = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Skip if test set is too small
        if len(y_test) < 5:
            continue
        
        # Train models
        xgb_model = XGBClassifier(
            n_estimators=min(n_estimators, 50),  # Reduce for speed
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=min(n_estimators, 50),
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        # Custom ensemble
        class SimpleEnsemble:
            def __init__(self, xgb_model, rf_model):
                self.xgb = xgb_model
                self.rf = rf_model
                
            def fit(self, X, y):
                self.xgb.fit(X, y)
                self.rf.fit(X, y)
                return self
                
            def predict(self, X):
                xgb_pred = self.xgb.predict(X)
                rf_pred = self.rf.predict(X)
                # Majority voting
                ensemble_pred = (xgb_pred + rf_pred) >= 1
                return ensemble_pred.astype(int)
                
            def predict_proba(self, X):
                xgb_proba = self.xgb.predict_proba(X)
                rf_proba = self.rf.predict_proba(X)
                # Average probabilities
                return (xgb_proba + rf_proba) / 2
        
        ensemble = SimpleEnsemble(xgb_model, rf_model)
        ensemble.fit(X_train, y_train)
        
        # Store models from last fold
        if fold == n_splits - 1:
            models['XGBoost'] = xgb_model
            models['Random Forest'] = rf_model
            models['Ensemble'] = ensemble
        
        # Evaluate all models
        for name, model in [('XGBoost', xgb_model), ('Random Forest', rf_model), ('Ensemble', ensemble)]:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics_history.append({
                'Fold': fold + 1,
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1': f1_score(y_test, y_pred, zero_division=0),
                'AUC-ROC': roc_auc_score(y_test, y_proba)
            })
    
    # If no metrics were collected
    if not metrics_history:
        return {}, {}, pd.DataFrame()
    
    # Calculate summary statistics
    metrics_df = pd.DataFrame(metrics_history)
    
    results = {}
    for model in metrics_df['Model'].unique():
        model_df = metrics_df[metrics_df['Model'] == model]
        if len(model_df) > 0:
            results[model] = {
                'Accuracy': {
                    'mean': model_df['Accuracy'].mean(),
                    'std': model_df['Accuracy'].std(),
                    'min': model_df['Accuracy'].min(),
                    'max': model_df['Accuracy'].max()
                },
                'F1': {
                    'mean': model_df['F1'].mean(),
                    'std': model_df['F1'].std()
                },
                'AUC-ROC': {
                    'mean': model_df['AUC-ROC'].mean(),
                    'std': model_df['AUC-ROC'].std()
                }
            }
    
    return models, results, metrics_df

def plot_feature_importance(model, features, model_name, top_n=15):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        fig, ax = plt.subplots(figsize=(12, 8))
        importance_df = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        
        if len(importance_df) > top_n:
            importance_df = importance_df.tail(top_n)
        
        importance_df.plot.barh(ax=ax)
        ax.set_title(f"{model_name} - Top {len(importance_df)} Feature Importance\n(All features use PREVIOUS day's data)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        return fig
    return None

# Sidebar controls
st.sidebar.header("Controls")
market = st.sidebar.radio("Market Type", ["Stocks", "Forex"])

if market == "Stocks":
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
else:
    ticker = st.sidebar.selectbox("Forex Pair", ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "USDCAD=X"])

start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Model parameters
st.sidebar.subheader("Model Parameters")
n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 3, 10, 6)
n_splits = st.sidebar.slider("Time Series CV Folds", 3, 10, 5)

# Main analysis
if st.sidebar.button("Run Realistic Analysis"):
    with st.spinner(f"Training models WITHOUT look-ahead bias for {ticker}..."):
        # Load and preprocess data
        data = load_data(ticker, start_date, end_date)
        if data is None or len(data) < 100:
            st.error(f"Failed to load sufficient data for {ticker} (need at least 100 days)")
            st.stop()
        
        st.success(f"Loaded {len(data)} days of data")
        
        # Preprocess WITHOUT look-ahead bias
        processed = preprocess_data_no_leakage(data)
        
        # Define features
        feature_cols = [col for col in processed.columns 
                       if col not in ['Target', 'Close', 'Open', 'High', 'Low', 'Volume'] 
                       and not col.startswith('Prev_')]  # Already have Prev_ prefix
        
        # Add all Prev_ columns
        prev_cols = [col for col in processed.columns if col.startswith('Prev_')]
        feature_cols = prev_cols + [col for col in feature_cols if col not in prev_cols]
        
        X = processed[feature_cols]
        y = processed['Target']
        
        st.info(f"Using {len(feature_cols)} LAGGED features - NO look-ahead bias")
        
        # Walk-forward validation
        models, results, cv_history = walk_forward_validation(
            X, y, n_splits=n_splits, n_estimators=n_estimators, max_depth=max_depth
        )
        
        # Check if we got results
        if not results:
            st.error("Not enough data for cross-validation. Try a longer date range.")
            st.stop()
        
        # Display results
        st.header(f"🎯 Realistic Model Analysis: {ticker}")
        
        # Model comparison with realistic ranges
        st.subheader("📊 Model Performance (Time Series Cross-Validation)")
        
        # Create summary dataframe
        summary_data = []
        for model_name, metrics in results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy (mean)': f"{metrics['Accuracy']['mean']:.1%}",
                'Accuracy Range': f"{metrics['Accuracy']['min']:.1%}-{metrics['Accuracy']['max']:.1%}",
                'F1 (mean)': f"{metrics['F1']['mean']:.1%}",
                'AUC-ROC': f"{metrics['AUC-ROC']['mean']:.3f} ± {metrics['AUC-ROC']['std']:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Interpretation
        st.subheader("🔍 What This Means")
        
        ensemble_accuracy = results.get('Ensemble', {}).get('Accuracy', {}).get('mean', 0.5)
        
        if market == "Forex":
            if ensemble_accuracy < 0.52:
                st.warning("⚠️ 48-52% accuracy: Model is barely better than random (typical for forex)")
            elif ensemble_accuracy < 0.55:
                st.info("📈 52-55% accuracy: Slight edge - potentially profitable with good risk management")
            elif ensemble_accuracy < 0.58:
                st.success("🌟 55-58% accuracy: Excellent result for forex - verify no remaining bias")
            else:
                st.error(f"⚠️ {ensemble_accuracy:.1%} is suspiciously high for forex - check for remaining look-ahead bias")
        else:
            if ensemble_accuracy < 0.53:
                st.info("📊 50-53% accuracy: Typical for stocks")
            elif ensemble_accuracy < 0.58:
                st.success("📈 53-58% accuracy: Good predictive power")
            else:
                st.warning("⚠️ >58% accuracy is possible for stocks but verify features")
        
        # Feature importance
        if models:
            st.subheader("🔑 Feature Importance (All use LAGGED data)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'XGBoost' in models:
                    fig_xgb = plot_feature_importance(models['XGBoost'], feature_cols, 'XGBoost')
                    if fig_xgb:
                        st.pyplot(fig_xgb)
            
            with col2:
                if 'Random Forest' in models:
                    fig_rf = plot_feature_importance(models['Random Forest'], feature_cols, 'Random Forest')
                    if fig_rf:
                        st.pyplot(fig_rf)
        
        # Cross-validation performance chart
        if not cv_history.empty:
            st.subheader("📈 Model Performance Across Folds")
            
            fig_cv, ax_cv = plt.subplots(figsize=(12, 6))
            for model in cv_history['Model'].unique():
                model_data = cv_history[cv_history['Model'] == model]
                if not model_data.empty:
                    ax_cv.plot(model_data['Fold'], model_data['Accuracy'], marker='o', label=model)
            
            ax_cv.axhline(y=0.5, color='gray', linestyle='--', label='Random (50%)')
            ax_cv.set_xlabel('Fold')
            ax_cv.set_ylabel('Accuracy')
            ax_cv.set_title('Accuracy Across Time Series Folds')
            ax_cv.legend()
            ax_cv.grid(True, alpha=0.3)
            st.pyplot(fig_cv)
        
        # Confusion matrix for best model
        if models:
            st.subheader("📊 Confusion Matrix (Last Fold)")
            
            selected_model = st.selectbox("Select model for confusion matrix", list(models.keys()))
            
            # Get last fold test data
            tscv = TimeSeriesSplit(n_splits=n_splits)
            test_idx = None
            for train_idx, test_idx_temp in tscv.split(X):
                test_idx = test_idx_temp
            
            if test_idx is not None and len(test_idx) > 0:
                X_test_last = X.iloc[test_idx]
                y_test_last = y.iloc[test_idx]
                
                fig_cm = plt.figure(figsize=(8, 6))
                y_pred = models[selected_model].predict(X_test_last)
                cm = confusion_matrix(y_test_last, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{selected_model} - Confusion Matrix (Last Fold)')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig_cm)
        
        # Next day prediction
        st.subheader("🔮 Next Day Prediction")
        
        latest_features = X.iloc[-1:].values
        
        if models:
            pred_cols = st.columns(len(models))
            for idx, (name, model) in enumerate(models.items()):
                with pred_cols[idx]:
                    pred = model.predict(latest_features)[0]
                    proba = model.predict_proba(latest_features)[0]
                    
                    direction = "📈 UP" if pred == 1 else "📉 DOWN"
                    confidence = max(proba)
                    
                    st.metric(
                        label=name,
                        value=direction,
                        delta=f"{confidence:.1%} confidence"
                    )
        
        # Store results for comparison
        if market == "Stocks":
            st.session_state.stock_accuracy = ensemble_accuracy
            st.session_state.stock_ticker = ticker
            st.session_state.stock_models = models
            st.session_state.stock_results = results
        else:
            st.session_state.forex_accuracy = ensemble_accuracy
            st.session_state.forex_ticker = ticker
            st.session_state.forex_models = models
            st.session_state.forex_results = results

# Comparison section
if 'stock_accuracy' in st.session_state and 'forex_accuracy' in st.session_state:
    st.header("🔍 Cross-Market Comparison (No Look-ahead Bias)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📈 Stocks: {st.session_state.stock_ticker}")
        st.metric("Ensemble Accuracy", f"{st.session_state.stock_accuracy:.1%}")
        
        if st.session_state.stock_accuracy > 0.58:
            st.warning("High accuracy - verify no bias")
        elif st.session_state.stock_accuracy > 0.53:
            st.success("Good predictive power")
        else:
            st.info("Typical for stocks")
    
    with col2:
        st.subheader(f"💱 Forex: {st.session_state.forex_ticker}")
        st.metric("Ensemble Accuracy", f"{st.session_state.forex_accuracy:.1%}")
        
        if st.session_state.forex_accuracy > 0.55:
            st.error("⚠️ Suspiciously high for forex (>55%) - likely still has look-ahead bias")
        elif st.session_state.forex_accuracy > 0.52:
            st.success("Good forex model (52-55%)")
        elif st.session_state.forex_accuracy > 0.48:
            st.info("Typical forex accuracy (48-52%)")
        else:
            st.error("Below random - model needs improvement")
    
    # Comparison chart
    if st.session_state.stock_models and st.session_state.forex_models:
        fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
        
        models_to_plot = ['XGBoost', 'Random Forest', 'Ensemble']
        stock_scores = []
        forex_scores = []
        
        for model in models_to_plot:
            # Get stock accuracy for this model
            if model in st.session_state.stock_results:
                stock_scores.append(st.session_state.stock_results[model]['Accuracy']['mean'])
            else:
                stock_scores.append(0)
            
            # Get forex accuracy for this model
            if model in st.session_state.forex_results:
                forex_scores.append(st.session_state.forex_results[model]['Accuracy']['mean'])
            else:
                forex_scores.append(0)
        
        x = np.arange(len(models_to_plot))
        width = 0.35
        
        ax_comp.bar(x - width/2, stock_scores, width, label='Stocks', color='blue', alpha=0.7)
        ax_comp.bar(x + width/2, forex_scores, width, label='Forex', color='green', alpha=0.7)
        
        ax_comp.axhline(y=0.5, color='gray', linestyle='--', label='Random (50%)')
        ax_comp.set_xlabel('Model')
        ax_comp.set_ylabel('Accuracy')
        ax_comp.set_title('Stock vs Forex - Realistic Comparison')
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels(models_to_plot)
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)
        ax_comp.set_ylim([0, 1])
        
        st.pyplot(fig_comp)
        
        # Summary table
        comparison_data = {
            'Metric': ['Stock Accuracy', 'Forex Accuracy', 'Difference'],
            'Value': [
                f"{st.session_state.stock_accuracy:.1%}",
                f"{st.session_state.forex_accuracy:.1%}",
                f"{(st.session_state.stock_accuracy - st.session_state.forex_accuracy):+.1%}"
            ]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)