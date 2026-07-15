"""
Quantifying the Impact of Formula-Driven Dependencies in Circuit Performance Prediction.
Complete analysis and figure generation pipeline.

Usage:
    1. Ensure 'perform101.csv' is in the same directory (optional, script will fallback to synthetic data).
    2. Install dependencies: pip install xgboost catboost lightgbm scipy torch scikit-learn matplotlib seaborn pandas numpy
    3. Run: python circuit_leakage_study.py
"""

import os
import warnings
import time
import json
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except Exception:
    HAS_TORCH = False
    DEVICE = 'cpu'

# --- Configuration ---
N_REPEATS = 10  
SEEDS = list(range(N_REPEATS))  
DATA_SEED = 42  
REPRESENTATIVE_SPLIT_SEED = SEEDS[0]  
REAL_DATA_PATH = 'perform101.csv'  


# =============================================================================
# SECTION 1: Real Dataset Analysis
# =============================================================================

def analyze_real_cktgnn_dataset(path=REAL_DATA_PATH):
    if not os.path.exists(path):
        print(f"[Warning] {path} not found. Skipping real-data validation.")
        return None

    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'valid' in df.columns:
        df = df.drop(columns=['valid'])

    corr_with_fom = df.corr()['fom'].drop('fom').sort_values(ascending=False)
    
    # Formula fit analysis
    X = df[['bw', 'pm', 'gain']]
    y = df['fom']
    lin = LinearRegression().fit(X, y)
    r2_formula = lin.score(X, y)

    results_real = {}
    for scenario, feat_cols in [('with_bw', ['gain', 'pm', 'bw']),
                                 ('without_bw', ['gain', 'pm'])]:
        Xs = df[feat_cols].values
        ys = df['fom'].values
        results_real[scenario] = run_repeated_split_experiments(
            Xs, ys, feature_names=feat_cols, scenario_label=f"REAL-{scenario}")
            
    return {
        'correlations': corr_with_fom,
        'formula_r2': r2_formula,
        'formula_coefs': dict(zip(['bw', 'pm', 'gain'], lin.coef_)),
        'results_real': results_real,
    }


# =============================================================================
# SECTION 2: Synthetic Dataset Generation
# =============================================================================

def create_synthetic_dataset(n_samples=10000, fom_weights=(0.7, 0.2, 0.1),
                             missing_rate=0.02, seed=DATA_SEED):
    assert abs(sum(fom_weights) - 1.0) < 1e-9, "fom_weights must sum to 1.0"
    rng = np.random.RandomState(seed)

    n = n_samples
    data = {
        'num_transistors': rng.randint(10, 500, n),
        'supply_voltage': rng.normal(1.8, 0.3, n),
        'temperature': rng.normal(27, 15, n),
        'process_variation': rng.normal(0, 0.1, n),
        'area_um2': rng.exponential(1000, n),
        'power_static_mw': rng.exponential(0.5, n),
        'power_dynamic_mw': rng.exponential(2.0, n),
        'capacitance_pf': rng.exponential(10, n),
        'resistance_ohm': rng.exponential(100, n),
        'frequency_mhz': rng.exponential(1000, n),
    }
    df = pd.DataFrame(data)

    df['total_power_mw'] = df['power_static_mw'] + df['power_dynamic_mw']
    df['power_density'] = df['total_power_mw'] / (df['area_um2'] + 1e-6)
    df['transistor_density'] = df['num_transistors'] / (df['area_um2'] + 1e-6)
    df['rc_constant'] = df['resistance_ohm'] * df['capacitance_pf'] * 1e-12
    df['voltage_normalized'] = (df['supply_voltage'] - 1.8) / 0.3
    df['temp_normalized'] = (df['temperature'] - 27) / 15

    base_performance = (df['frequency_mhz'] * df['supply_voltage']
                        * np.sqrt(df['num_transistors'])) / (df['total_power_mw'] + 1e-6)
    noise_factor = 0.2
    df['bandwidth_mhz'] = base_performance * (1 + rng.normal(0, noise_factor, n))
    df['bandwidth_mhz'] = np.maximum(df['bandwidth_mhz'], 1.0)

    w_bw, w_fvp, w_td = fom_weights
    fvp_component = (df['frequency_mhz'] * df['supply_voltage']) / (df['total_power_mw'] + 1e-6)
    td_component = df['num_transistors'] / (df['area_um2'] + 1e-6) * 1000

    fom_components = [df['bandwidth_mhz'] * w_bw, fvp_component * w_fvp, td_component * w_td]
    fom_sum = sum(fom_components)
    df['fom'] = fom_sum + rng.normal(0, fom_sum * 0.1)
    df['fom'] = (df['fom'] - df['fom'].min()) / (df['fom'].max() - df['fom'].min()) * 100

    df['efficiency'] = df['fom'] / (df['total_power_mw'] + 1e-6)
    df['performance_per_area'] = df['fom'] / (df['area_um2'] + 1e-6)

    # MCAR Missingness Injection
    mcar_cols = ['supply_voltage', 'temperature', 'capacitance_pf', 'resistance_ohm']
    for col in mcar_cols:
        mask = rng.rand(n) < missing_rate
        df.loc[mask, col] = np.nan

    # Verification of MCAR assumption
    for col in mcar_cols:
        is_missing = df[col].isna().astype(int)
        if is_missing.sum() > 0:
            point_biserial = np.corrcoef(is_missing, df['fom'])[0, 1]
            assert abs(point_biserial) < 0.05, f"MCAR violation detected in {col}"

    return df


# =============================================================================
# SECTION 3: Preprocessing
# =============================================================================

def prepare_features(df, target_col='fom', exclude_features=None, skew_threshold=5.0,
                     protect_from_transform=('bandwidth_mhz',)):
    from scipy.stats import skew as _skew
    exclude_features = exclude_features or []
    exclude_cols = [target_col] + exclude_features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    X = df[feature_cols].copy()
    n_missing = X.isna().sum().sum()
    if n_missing > 0:
        X = X.fillna(X.median())

    for col in X.columns:
        if col in protect_from_transform:
            continue
        if (X[col] >= 0).all():
            s = _skew(X[col].values)
            if np.isfinite(s) and s > skew_threshold:
                lo, hi = X[col].quantile(0.005), X[col].quantile(0.995)
                clipped = X[col].clip(lo, hi)
                X[col] = np.log1p(clipped)

    constant_cols = X.columns[X.var() < 1e-10]
    if len(constant_cols) > 0:
        X = X.drop(columns=constant_cols)

    y = df[target_col].copy()
    return X, y


# =============================================================================
# SECTION 4: Models and Repeated Split Evaluation
# =============================================================================

MODEL_FACTORY = {
    'Linear Regression': lambda seed: LinearRegression(),
    'Random Forest': lambda seed: RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
        random_state=seed, n_jobs=-1),
    'XGBoost': lambda seed: xgb.XGBRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=seed, verbosity=0),
}
if HAS_CATBOOST:
    MODEL_FACTORY['CatBoost'] = lambda seed: CatBoostRegressor(
        iterations=300, depth=8, learning_rate=0.1, random_seed=seed, verbose=False)
if HAS_LIGHTGBM:
    MODEL_FACTORY['LightGBM'] = lambda seed: LGBMRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.1, random_state=seed, verbosity=-1)


class MLPRegressorTorch(nn.Module if HAS_TORCH else object):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_eval_mlp(X_train, y_train, X_val, y_val, X_test, y_test, seed, epochs=100, patience=15):
    torch.manual_seed(seed)
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(np.asarray(y_train, dtype=float).reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(np.asarray(y_val, dtype=float).reshape(-1, 1)).ravel()

    model = MLPRegressorTorch(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5)
    crit = nn.MSELoss()

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.FloatTensor(np.asarray(X)), torch.FloatTensor(np.asarray(y)))
        bs = max(16, min(128, len(ds) // 10))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle and len(ds) > bs)

    train_loader = to_loader(X_train, y_train_s, True)
    val_loader = to_loader(X_val, y_val_s, False)
    test_loader = to_loader(X_test, y_test, False)  

    best_val, best_state, bad_epochs = float('inf'), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vloss, nb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vloss += crit(model(xb), yb).item()
                nb += 1
        vloss /= max(nb, 1)
        sched.step(vloss)
        if vloss < best_val:
            best_val, best_state, bad_epochs = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    preds_s, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds_s.extend(model(xb).cpu().numpy())
            actuals.extend(yb.numpy())
            
    preds = y_scaler.inverse_transform(np.array(preds_s).reshape(-1, 1)).ravel()
    return preds, np.array(actuals)


def run_repeated_split_experiments(X, y, feature_names, scenario_label):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    per_model_metrics = {name: {'r2': [], 'mae': [], 'rmse': [], 'time': []}
                          for name in list(MODEL_FACTORY.keys()) + (['MLP'] if HAS_TORCH else [])}

    for seed in SEEDS:
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=seed)

        scaler = RobustScaler().fit(X_train)
        X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

        for name, factory in MODEL_FACTORY.items():
            model = factory(seed)
            t0 = time.time()
            model.fit(X_train_s, y_train)
            dt = time.time() - t0
            pred = model.predict(X_test_s)
            per_model_metrics[name]['r2'].append(r2_score(y_test, pred))
            per_model_metrics[name]['mae'].append(mean_absolute_error(y_test, pred))
            per_model_metrics[name]['rmse'].append(np.sqrt(mean_squared_error(y_test, pred)))
            per_model_metrics[name]['time'].append(dt)

        if HAS_TORCH:
            t0 = time.time()
            pred, actual = train_eval_mlp(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, seed)
            dt = time.time() - t0
            per_model_metrics['MLP']['r2'].append(r2_score(actual, pred))
            per_model_metrics['MLP']['mae'].append(mean_absolute_error(actual, pred))
            per_model_metrics['MLP']['rmse'].append(np.sqrt(mean_squared_error(actual, pred)))
            per_model_metrics['MLP']['time'].append(dt)

    return per_model_metrics


# =============================================================================
# SECTION 5: Statistics
# =============================================================================

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))

def bootstrap_ci_diff(a, b, n_boot=5000, seed=0):
    rng = np.random.RandomState(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = np.array([rng.choice(a, len(a), replace=True).mean()
                       - rng.choice(b, len(b), replace=True).mean() for _ in range(n_boot)])
    return np.percentile(diffs, [2.5, 97.5])

def compare_scenarios(results_with, results_without, label=""):
    rows = []
    for name in results_with:
        if name not in results_without:
            continue
        a = np.array(results_with[name]['r2']) 
        b = np.array(results_without[name]['r2']) 
        try:
            stat, p = stats.wilcoxon(a - b)
        except ValueError:
            stat, p = np.nan, np.nan
            
        d = cohens_d(a, b)
        delta = cliffs_delta(a, b)
        ci = bootstrap_ci_diff(a, b)
        rel_drop_pct = 100 * (a.mean() - b.mean()) / a.mean() if a.mean() != 0 else np.nan
        
        rows.append({
            'model': name, 
            'r2_with_mean': a.mean(), 
            'r2_with_std': a.std(),
            'r2_without_mean': b.mean(), 
            'r2_without_std': b.std(),
            'abs_drop': a.mean() - b.mean(), 
            'rel_drop_pct': rel_drop_pct,
            'wilcoxon_p': p, 
            'cohens_d': d, 
            'cliffs_delta': delta,
            'ci95_low': ci[0], 
            'ci95_high': ci[1],
        })
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 6: Figure Generation
# =============================================================================

def _lazy_plotting_imports():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 150
    return plt, sns

def figure2_real_data(real_csv=REAL_DATA_PATH, outdir='figures'):
    plt, sns = _lazy_plotting_imports()
    if not os.path.exists(real_csv):
        return
    df = pd.read_csv(real_csv)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'valid' in df.columns:
        df = df.drop(columns=['valid'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    sample = df.sample(min(3000, len(df)), random_state=REPRESENTATIVE_SPLIT_SEED)
    ax.scatter(sample['bw'], sample['fom'], alpha=0.3, s=10, color='#2b6cb0')
    z = np.polyfit(df['bw'], df['fom'], 1)
    xs = np.linspace(df['bw'].min(), df['bw'].max(), 100)
    ax.plot(xs, np.poly1d(z)(xs), 'r--', lw=2, label=f'fit: fom={z[0]:.3f}*bw+{z[1]:.2f}')
    ax.set_xlabel('bandwidth (bw)')
    ax.set_ylabel('Figure of Merit (fom)')
    ax.set_title(f'bw vs fom (r = {df["bw"].corr(df["fom"]):.6f})')
    ax.legend(fontsize=8)

    ax = axes[1]
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f', square=True, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title('Correlation matrix (original CktGNN features)')

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/figure2_original_dataset_leakage.png')
    plt.close()

def figure3_4_synthetic(fom_weights=(0.7, 0.2, 0.1), outdir='figures'):
    plt, sns = _lazy_plotting_imports()
    df = create_synthetic_dataset(fom_weights=fom_weights, seed=DATA_SEED)

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f',
                square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 6})
    plt.title('Synthetic Dataset Correlation Matrix (upper triangle hidden)')
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/figure3_synthetic_correlation_matrix.png')
    plt.close()

    plt.figure(figsize=(8, 5.5))
    sample = df.sample(min(3000, len(df)), random_state=REPRESENTATIVE_SPLIT_SEED)
    plt.scatter(sample['bandwidth_mhz'], sample['fom'], alpha=0.3, s=10, color='#c0392b')
    z = np.polyfit(df['bandwidth_mhz'], df['fom'], 1)
    xs = np.linspace(df['bandwidth_mhz'].min(), df['bandwidth_mhz'].max(), 100)
    plt.plot(xs, np.poly1d(z)(xs), 'k--', lw=2)
    r = df['bandwidth_mhz'].corr(df['fom'])
    plt.xlabel('bandwidth_mhz (shortcut feature)')
    plt.ylabel('FoM (target)')
    plt.title(f'Synthetic Shortcut-Feature Demonstration (r = {r:.4f})')
    plt.tight_layout()
    plt.savefig(f'{outdir}/figure4_synthetic_leakage_scatter.png')
    plt.close()

def _run_one_split(df, exclude_features, split_seed=REPRESENTATIVE_SPLIT_SEED):
    X, y = prepare_features(df, exclude_features=exclude_features)
    X, y = X.values.astype(float), y.values.astype(float)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=split_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=split_seed)
    scaler = RobustScaler().fit(X_train)
    X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    results = {}
    for name, factory in MODEL_FACTORY.items():
        model = factory(split_seed)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        results[name] = {'y_true': y_test, 'y_pred': pred, 'r2': r2_score(y_test, pred)}

    if HAS_TORCH:
        pred, actual = train_eval_mlp(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, split_seed)
        results['MLP'] = {'y_true': actual, 'y_pred': pred, 'r2': r2_score(actual, pred)}

    return results

def figures5_6_7(fom_weights=(0.7, 0.2, 0.1), outdir='figures'):
    plt, sns = _lazy_plotting_imports()
    df = create_synthetic_dataset(fom_weights=fom_weights, seed=DATA_SEED)
    res_with = _run_one_split(df, exclude_features=[])
    res_without = _run_one_split(df, exclude_features=['bandwidth_mhz'])

    models = [m for m in res_with if m in res_without]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    r2_with = [max(0, res_with[m]['r2']) for m in models]
    r2_without = [max(0, res_without[m]['r2']) for m in models]
    b1 = ax.bar(x - width / 2, r2_with, width, label='With shortcut feature', color='#e07a5f')
    b2 = ax.bar(x + width / 2, r2_without, width, label='Without shortcut feature', color='#3d5a80')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylabel('Test R²')
    ax.set_title(f'Model Performance With vs Without Shortcut Feature (split_seed={REPRESENTATIVE_SPLIT_SEED})')
    ax.legend()
    
    # Complete text layout printing to avoid truncation errors
    for bars in (b1, b2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, 
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/figure5_shortcut_comparison.png')
    plt.close()


# =============================================================================
# MAIN EXECUTION ROUTINE
# =============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("STARTING CIRCUIT LEAKAGE STUDY PIPELINE")
    print("=" * 80)
    
    outdir = 'figures'
    os.makedirs(outdir, exist_ok=True)
    
    # 1. Real Dataset Verification
    if os.path.exists(REAL_DATA_PATH):
        print(f"\n[1/4] Found local real dataset at '{REAL_DATA_PATH}'. Analyzing...")
        real_analysis = analyze_real_cktgnn_dataset(REAL_DATA_PATH)
        if real_analysis:
            print(f" -> Real Target Formula R²: {real_analysis['formula_r2']:.6f}")
            print(" -> Estimated weights: ", real_analysis['formula_coefs'])
            print(" -> Saving Figure 2 (Original Dataset Heatmap & Scatter)...")
            figure2_real_data(REAL_DATA_PATH, outdir=outdir)
    else:
        print(f"\n[1/4] Real dataset '{REAL_DATA_PATH}' not detected. Skipping real-world checks.")

    # 2. Synthetic Dataset Pipeline
    print(f"\n[2/4] Generating Synthetic Dataset (n=10,000, Data Seed={DATA_SEED})...")
    df_synthetic = create_synthetic_dataset(seed=DATA_SEED)
    print(" -> Data successfully initialized.")
    print(" -> Saving Figures 3 & 4 (Synthetic Correlations & Shortcut Scatter)...")
    figure3_4_synthetic(outdir=outdir)
    
    # 3. Model Training & Comparison
    print("\n[3/4] Preparing Features and Running Repeated Experiments (10 Splits)...")
    X_with, y_with = prepare_features(df_synthetic, exclude_features=[])
    X_without, y_without = prepare_features(df_synthetic, exclude_features=['bandwidth_mhz'])
    
    print(" -> Running models with shortcut feature...")
    results_with = run_repeated_split_experiments(X_with, y_with, X_with.columns, "Synthetic-With")
    
    print(" -> Running models without shortcut feature...")
    results_without = run_repeated_split_experiments(X_without, y_without, X_without.columns, "Synthetic-Without")
    
    # 4. Statistical Analysis Output
    print("\n[4/4] Conducting Robust Statistical Signficance Comparison...")
    comparison_stats = compare_scenarios(results_with, results_without)
    
    # Pretty print summary
    print("\n" + "=" * 90)
    print("                               EXPERIMENT RESULTS SUMMARY                                ")
    print("=" * 90)
    print(comparison_stats[['model', 'r2_with_mean', 'r2_without_mean', 'abs_drop', 'cohens_d', 'wilcoxon_p']].to_string(index=False))
    print("=" * 90)
    
    # Generate Visual R2 Comparison Chart
    print("\nSaving Figure 5 (Shortcut Comparison Bar Chart)...")
    figures5_6_7(outdir=outdir)
    
    print(f"\n[SUCCESS] Pipeline executed completely! Visualizations saved in './{outdir}/'.")
    print("=" * 80)
