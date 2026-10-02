# Market Game State

This project applies inverse game theory and Quantal Response Equilibrium (QRE) inference to financial markets to infer the strategic game underlying observed market behavior. The inferred game is then compared with canonical 2×2 games such as the Prisoner's Dilemma, Chicken, and Battle of the Sexes.

## What This Does

- Classifies daily returns of SPY and QQQ into strategy outcomes:
  - `LL` → SPY Long, QQQ Long
  - `LS` → SPY Long, QQQ Short
  - `SL` → SPY Short, QQQ Long
  - `SS` → SPY Short, QQQ Short
- Infers **payoff structures** for two players based on observed market behavior.
- Uses **Quantal Response Equilibrium (QRE)** to estimate **strategic preferences** under noisy decision-making.
- Matches these preferences to canonical 2x2 games:
  - Prisoner's Dilemma
  - Chicken
  - Battle of the Sexes
- Outputs:
  - Inferred payoff matrices for both players.
  - Preference rankings.
  - Closest canonical game and match score.
  - Pure Nash equilibrium of the inferred game.

## How It Works

1. **Data**:
   - Downloads daily price data using `yfinance` (e.g., SPY, QQQ)
   - Allows the user to specify the lookback period for market analysis.
   - Computes **daily returns** as % changes
   - Classifies outcomes into one of four action pairs: `LL`, `LS`, `SL`, `SS`
2. **Model**:
   - Uses observed frequencies of outcomes
   - Estimates latent payoffs via **KL divergence minimization**
   - Applies **QRE** to account for bounded rationality
3. **Game Inference**:
   - Ranks inferred preferences
   - Matches against known game types
   - Assigns match scores to canonical games and computes the pure Nash equilibrium of the inferred game.

## What Do `LL`, `LS`, `SL`, `SS` Mean?

These represent the strategy combinations of two market players (or proxies like SPY and QQQ):

| Label | Meaning              | Interpretation                  |
|-------|----------------------|----------------------------------|
| LL    | Long SPY, Long QQQ   | Both assets had positive returns |
| LS    | Long SPY, Short QQQ  | SPY up, QQQ down                 |
| SL    | Short SPY, Long QQQ  | SPY down, QQQ up                 |
| SS    | Short SPY, Short QQQ | Both assets had negative returns |

**Important**:  
Short returns aren’t inherently negative.  
The **sign** of a return only indicates whether the asset’s price went **up or down**.  
A trader holding a short position profits when the return is negative, but in the data, the negative value simply means the price fell.

## Example Output

```
=== Inferred Preference Ranking ===
Player I: ['SL', 'LL', 'SS', 'LS']
Player II: ['LS', 'LL', 'SS', 'SL']

=== Best Matching Game ===
Game Type: Prisoners Dilemma
Match Score: 6 / 8

=== Game Type Probabilities (Softmax) ===
Prisoners Dilemma: 0.735
Chicken: 0.194
Battle of the Sexes: 0.071
```

### Understanding the Output

The model infers which game-theoretic regime the market is currently behaving like (e.g., Prisoner's Dilemma).

- **Match Score**: Measures how closely the observed payoff rankings match classic game templates. Max score is 8 (4 rankings × 2 players).
- **Closest Canonical Game**: The canonical game whose preference ordering most closely matches the inferred game.


## How To Run

**Google Colab (recommended)**:
-[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FShahKhan/market-game-state/blob/main/market_game_state.ipynb)


## License

This project is under a **Non-Commercial License**:  
You are free to use, modify, and extend this code for research or personal use —  
but **not for commercial redistribution** without permission.
