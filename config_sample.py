# config_sample.py
# Template configuration file for the Random Forest Trading Framework

VOLUME = 1.00

# Example threshold probabilities for model execution
THRESHOLDS = {
    "EURUSD": 0.65,
    "GBPUSD": 0.65,
}

# Initial protective Stop Loss distances in points
INITIAL_SL_POINTS = {
    "EURUSD": 250,
    "GBPUSD": 300,
}

# Profit distances in points required to lock in Break-Even protection
BE_TRIGGER_POINTS = {
    "EURUSD": 120,
    "GBPUSD": 150,
}

# Trailing stop distances in points behind current market price
TRAILING_STOP_POINTS = {
    "EURUSD": 150,
    "GBPUSD": 200,
}
