import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

# --- STEP 1: Download and prepare market data ---
tickers = ['SPY', 'QQQ']
raw = yf.download(tickers, start="2022-01-01", end="2023-01-01", auto_adjust=True)

# --- STEP 1B: Safely extract adjusted close prices ---
try:
    data = raw['Adj Close']
except KeyError:
    try:
        data = raw['Close']
    except KeyError:
        raise ValueError("Neither 'Adj Close' nor 'Close' found. Try re-downloading without auto_adjust.")

data.columns = tickers

# Compute daily returns
returns = data.pct_change().dropna()

# --- STEP 2: Classify daily outcomes as strategy pairs ---
def classify(row):
    if row['SPY'] > 0 and row['QQQ'] > 0:
        return 'LL'
    elif row['SPY'] > 0 and row['QQQ'] < 0:
        return 'LS'
    elif row['SPY'] < 0 and row['QQQ'] > 0:
        return 'SL'
    else:
        return 'SS'

returns['Outcome'] = returns.apply(classify, axis=1)

# Ensure all 4 outcomes are present
all_outcomes = ['LL', 'LS', 'SL', 'SS']
freqs = returns['Outcome'].value_counts(normalize=True).reindex(all_outcomes, fill_value=0)
joint_probs = freqs.to_numpy()

if np.count_nonzero(joint_probs) < 3:
    raise ValueError("Too few distinct outcomes to infer preferences. Try more data.")

# --- STEP 3: QRE payoff inference ---
def qre_probs(u, lam=1.5):
    EU_I_L = u[0]*joint_probs[0] + u[1]*joint_probs[1]
    EU_I_S = u[2]*joint_probs[2] + u[3]*joint_probs[3]
    pI_L = np.exp(lam * EU_I_L)
    pI_S = np.exp(lam * EU_I_S)
    pI_L /= (pI_L + pI_S)

    EU_II_L = u[4]*joint_probs[0] + u[6]*joint_probs[2]
    EU_II_S = u[5]*joint_probs[1] + u[7]*joint_probs[3]
    pII_L = np.exp(lam * EU_II_L)
    pII_S = np.exp(lam * EU_II_S)
    pII_L /= (pII_L + pII_S)

    return np.array([
        pI_L * pII_L,             # LL
        pI_L * (1 - pII_L),       # LS
        (1 - pI_L) * pII_L,       # SL
        (1 - pI_L) * (1 - pII_L)  # SS
    ])

def kl_divergence(p_obs, p_model):
    return np.sum(p_obs * np.log((p_obs + 1e-8) / (p_model + 1e-8)))

def loss(u):
    return kl_divergence(joint_probs, qre_probs(u))

# Optimize payoffs (8 parameters: 4 per player)
res = minimize(loss, np.random.randn(8), method='BFGS')
u_opt = res.x
payoffs_I = dict(zip(['LL', 'LS', 'SL', 'SS'], u_opt[:4]))
payoffs_II = dict(zip(['LL', 'LS', 'SL', 'SS'], u_opt[4:]))

# --- STEP 4: Rank preferences and match against game types ---
def rank(p_dict):
    return [k for k, _ in sorted(p_dict.items(), key=lambda x: -x[1])]

rank_I = rank(payoffs_I)
rank_II = rank(payoffs_II)

# Canonical game rankings
games = {
    'Prisoners Dilemma': (['SL', 'LL', 'SS', 'LS'], ['LS', 'LL', 'SS', 'SL']),
    'Chicken':           (['LS', 'LL', 'SL', 'SS'], ['SL', 'LL', 'LS', 'SS']),
    'Battle of the Sexes': (['LL', 'SS', 'LS', 'SL'], ['SS', 'LL', 'SL', 'LS']),
}

def score_match(r1, r2, g1, g2):
    return sum([r1[i] == g1[i] for i in range(4)]) + sum([r2[i] == g2[i] for i in range(4)])

scores = {name: score_match(rank_I, rank_II, g1, g2) for name, (g1, g2) in games.items()}
best_game = max(scores.items(), key=lambda x: x[1])

# --- STEP 5: Probabilistic interpretation ---
score_vals = np.array(list(scores.values()), dtype=float)
score_probs = softmax(score_vals)
score_names = list(scores.keys())

# --- STEP 6: Output ---
print("\n=== Inferred Preference Ranking ===")
print("Player I:", rank_I)
print("Player II:", rank_II)

print("\n=== Best Matching Game ===")
print("Game Type:", best_game[0])
print("Match Score:", best_game[1], "/ 8")

print("\n=== Game Type Probabilities (Softmax) ===")
for name, prob in zip(score_names, score_probs):
    print(f"{name}: {prob:.3f}")
