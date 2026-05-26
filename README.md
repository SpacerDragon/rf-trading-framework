# Multi-Timeframe Algorithmic Trading Framework

An enterprise-grade, object-oriented, single-threaded execution framework for automated quantitative trading via the MetaTrader 5 API. This repository showcases advanced state-tracking, clean modular configuration layers, and asynchronous-emulated execution loops designed to run predictive machine learning models across multiple asset assets and timeframes concurrently.

---

## Architectural Highlights & Design Philosophy

Traditional multi-asset algorithmic bots heavily rely on multi-threading. However, network I/O bounds, state synchronization issues, and race conditions within shared API contexts often introduce fragile failure points. This framework completely addresses those hazards through the following architecture:

* Deterministic Single-Threaded Processing: Utilizes a highly optimized, sequential execution matrix. The system loops over individual asset-timeframe engine instances, performing lightning-fast status ticks. This completely eliminates data races and thread collisions while keeping CPU overhead minimal.
* Object-Oriented Symmetrical Scale: Built around a robust parent class (BaseTimeframeTrader). Implementing a completely new timeframe Strategy variant (e.g., a 30-Minute or 1-Hour model) requires writing less than 10 lines of declarative child class configuration code.
* Persistent Trade State Tracking (Crash Resilience): Implements an internal memory matrix tracking individual positions across distinct states (entry -> break-even -> trailing). If the core script execution blinks or loses connection, an auto-healing evaluation layer automatically interrogates active terminal tickets to rebuild the state matrix perfectly upon reboot.
* Decoupled Configuration Boundaries: Hard strategy logic is strictly separated from execution thresholds. System parameters (lot-sizing, directional thresholds, stop levels) are kept in standalone configuration files, maintaining high readability across the repository.

---

## Repository Structure

├── .gitignore               # Strict safety barriers tracking credentials and artifacts
├── README.md                # Project documentation and engineering blueprint
├── config_sample.py         # Mock structural template demonstrating configuration format
└── framework_engine.py      # Core OOP engine containing the parent base & execution loop

---

## How the Risk Management State Machine Works

Every position spawned by the framework is isolated via specific Magic Number offsets calculated at boot. Once an order execution deal is confirmed, the position flows through a rigid, 3-stage protective pipeline:

[State: entry] --> Price hits BE_TRIGGER --> [State: be] --> Price moves past TRAILING_STOP --> [State: trail]

1. entry (Initial Protection): A hard Stop Loss is immediately placed at a set distance (INITIAL_SL_POINTS) from the deal execution price to isolate risk.
2. be (Break-Even Lock): If the market moves in favor of the trade by a distance of BE_TRIGGER_POINTS, the stop-loss is updated to the opening price plus a minor spread_cushion to eliminate financial liability on the trade.
3. trail (Dynamic Profit Capture): As price expands further, the engine flags the trade state as trail, smoothly tightening the stop-loss behind the current market price according to your defined TRAILING_STOP_POINTS configuration.

---

## Getting Started

### 1. Prerequisites
Ensure your local trading or backtesting node runs Python 3.10+ alongside a functioning MetaTrader 5 terminal instance. Install framework dependencies via pip:

pip install MetaTrader5 pandas joblib python-dotenv

### 2. Infrastructure Setup
Create a secure environment file named .env in the root folder of your project workspace to feed terminal authentication arrays safely into the framework wrapper:

MT5_LOGIN=12345678
MT5_PASSWORD=YourSecurePasswordHere
MT5_SERVER=YourBroker-LiveOrDemoServer

*(Note: .env is explicitly trapped by .gitignore and will never be pushed to your public version control repository).*

### 3. Model Provisions
The parent architecture expects an optimized structure for tracking models and pipeline metrics. Ensure you populate your local workspace paths to align structurally with your asset allocation choices:

└── models/
    ├── 5m_models/
    │   ├── EURUSD_rf.joblib
    │   └── EURUSD_stats.json
    └── 15m_models/
        ...

---

## Technical Feature Engineering Pipeline

The execution heartbeat queries live OHLCV terminal arrays on every candle change, constructing uniform input matrices used to poll predictive classification pipelines:
* Relative Exponential Divergence: Calculates closing deviation relative to structural moving averages: (Close - EMA50) / EMA50.
* Normalized Volatility Index: Computes Average True Range percentages over historical windows relative to underlying asset pricing scales.
* Oscillator Vectors: Extracts relative RSI momentum boundaries and normalized MACD-to-Signal delta arrays.

---

## Disclaimer

This repository is shared strictly as an open-source software engineering architectural framework. It does not contain live algorithmic trading models, optimized model statistics, or financial advice. All execution metrics shown within sample parameters utilize arbitrary placeholder variables intended for structural validation only.
