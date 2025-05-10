import MetaTrader5 as mt5
import os

# Création du dossier output s'il n'existe pas
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Paramètre de risque
INITIAL_BALANCE = 10000
MIN_LOT_SIZE = 0.01
PROFIT_WEIGHT = 1.0
PROFIT_FACTOR_WEIGHT = 2.0
WIN_RATE_WEIGHT = 1.5
DRAWDOWN_WEIGHT = 2.0
SHARPE_WEIGHT = 1.5
TRADES_WEIGHT = 0.5
RISK_REWARD_WEIGHT = 2.0
MAX_DRAWDOWN = 50
MIN_TRADES = 20
DRAWDOWN_PENALTY = 10
TRADES_PENALTY = 0.2

# Paramètres de la stratégie
N_STRATEGIES = 1000
N_BARS = 1000
CURRENCY_PAIR = 'EURUSD'
TIMEFRAME = mt5.TIMEFRAME_H1

# Autres paramètres
PIP_VALUE = 0.0001
DATA_FILE = os.path.join(OUTPUT_DIR, "historical_data.csv")
EQUITY_CURVE_FILE = os.path.join(OUTPUT_DIR, "equity_curve.png")
BEST_STRATEGY_PARAMS_FILE = os.path.join(OUTPUT_DIR, "best_strategy_params.csv")
BEST_STRATEGY_TRADES_FILE = os.path.join(OUTPUT_DIR, "best_strategy_trades.csv")
ALL_STRATEGIES_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "all_strategies_summary.csv")