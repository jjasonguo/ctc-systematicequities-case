# Bot Configuration Guide

## Quick Start

### Run with preset modes:
```bash
# Conservative (default) - safest, smallest positions
python bot.py --loop --hz 1 --mode conservative

# Moderate - balanced risk/reward
python bot.py --loop --hz 2 --mode moderate

# Aggressive - maximum trading
python bot.py --loop --hz 3 --mode aggressive
```

### Override specific parameters:
```bash
# Conservative mode but with higher signal threshold
python bot.py --loop --hz 1 --mode conservative --signal-threshold 0.60

# Moderate mode but with lower position limits
python bot.py --loop --hz 1 --mode moderate --position-limit 250

# Conservative mode with more aggressive ETF arb
python bot.py --loop --hz 1 --mode conservative --etf-arb-threshold 0.07
```

---

## Configuration Parameters

### CONSERVATIVE (Default) ⚠️ Safest
- **Position Limits**: 200 max per symbol, 30 max per trade
- **Signal Threshold**: 0.50 (only trades very strong signals)
- **ETF Arb**: 0.10 threshold (selective arbitrage)
- **Spreads**: AAA=0.15, BBB=0.18, CCC=0.20
- **Trade Sizes**: 15 base, 20 ETF

### MODERATE ⚡ Balanced
- **Position Limits**: 350 max per symbol, 50 max per trade
- **Signal Threshold**: 0.35 (trades moderate signals)
- **ETF Arb**: 0.08 threshold (more frequent arb)
- **Spreads**: AAA=0.12, BBB=0.14, CCC=0.16
- **Trade Sizes**: 25 base, 30 ETF

### AGGRESSIVE 🔥 Maximum
- **Position Limits**: 500 max per symbol, 80 max per trade
- **Signal Threshold**: 0.25 (trades more frequently)
- **ETF Arb**: 0.06 threshold (aggressive arb)
- **Spreads**: AAA=0.10, BBB=0.12, CCC=0.14
- **Trade Sizes**: 35 base, 45 ETF

---

## All Tunable Parameters

Edit these in the `BotConfig` class if you want custom configurations:

### Risk Management
- `hard_position_limit` - Max absolute position per symbol
- `per_symbol_soft_limit` - Soft warning limit
- `per_trade_max` - Max quantity per order

### Signal Thresholds
- `signal_threshold` - Min signal strength to trade (higher = fewer trades)
- `aaa_shock_zscore` - Z-score threshold for AAA shock detection

### Position Sizing
- `base_trade_size` - Base size for AAA/BBB/CCC trades
- `etf_trade_size` - Base size for ETF trades

### ETF Arbitrage
- `etf_arb_threshold` - Min price difference to arb (higher = safer)
- `etf_arb_qty` - Quantity per arb trade

### Market Making
- `mm_aaa_spread` - AAA bid-ask spread width
- `mm_bbb_spread` - BBB bid-ask spread width
- `mm_ccc_spread` - CCC bid-ask spread width
- `mm_etf_spread` - ETF bid-ask spread width
- `mm_quote_size` - Quote size per side
- `inventory_skew_divisor` - Inventory flattening aggressiveness (lower = more aggressive)

### Alpha Strategy Parameters
- `aaa_window` - AAA rolling stats window size
- `bbb_ema_fast` - BBB fast EMA period
- `bbb_ema_slow` - BBB slow EMA period
- `ccc_window` - CCC mean reversion window
- `ccc_deviation_threshold` - CCC deviation to trigger signal

### Operational
- `cancel_frequency` - Cancel orders every N ticks

---

## Tips for Tuning

### If you're losing money:
1. Increase `signal_threshold` (0.50 → 0.60)
2. Increase `etf_arb_threshold` (0.10 → 0.12)
3. Widen spreads (`mm_*_spread` values)
4. Lower position limits
5. Run at lower Hz (--hz 0.5)

### If you're not trading enough:
1. Decrease `signal_threshold` (0.50 → 0.40)
2. Decrease `etf_arb_threshold` (0.10 → 0.08)
3. Tighten spreads
4. Increase position limits
5. Run at higher Hz (--hz 2)

### If positions are too large:
1. Lower `hard_position_limit`
2. Lower `per_trade_max`
3. Lower `base_trade_size` and `etf_trade_size`

### If you want more ETF arb opportunities:
1. Lower `etf_arb_threshold`
2. Increase `etf_arb_qty`

---

## Recommended Starting Points

### For Consistent Small Profits:
```bash
python bot.py --loop --hz 0.5 --mode conservative
```

### For Balanced Trading:
```bash
python bot.py --loop --hz 1 --mode moderate
```

### For Maximum Action (risky):
```bash
python bot.py --loop --hz 2 --mode aggressive
```

