#!/usr/bin/env python3
import os, time, json, argparse, math, statistics as stats
from typing import Any, Optional, Dict, List, Tuple
import requests
from collections import deque, defaultdict
import statistics as stats

# sensible defaults; tweak as you like
DEFAULT_MAX_ABS_SPREAD = {"ETF": 0.30, "BBB": 0.30, "CCC": 0.30, "AAA": 0.50}
DEFAULT_SPREAD_MULT    = 6.0   # news burst = spread > 6x median
DEFAULT_MIN_TOP_DEPTH  = 1     # require at least 1 lot on each side

# =========================
# Configuration
# =========================
class BotConfig:
    """Centralized configuration for easy tuning of bot aggressiveness."""
    
    def __init__(self, aggressiveness: str = "conservative"):
        """
        aggressiveness: 'conservative', 'moderate', or 'aggressive'
        """

        if aggressiveness == "conservative":
            # RISK MANAGEMENT
            self.hard_position_limit = 2000     # Max position per symbol
            self.per_symbol_soft_limit = 1500   # Soft limit per symbol
            self.per_trade_max = 300            # Max size per order
            
            # SIGNAL THRESHOLDS (higher = fewer trades)
            self.signal_threshold = 0.70        # Min signal strength to trade
            self.aaa_shock_zscore = 3.0         # Z-score needed for AAA shock
            
            # POSITION SIZING
            self.base_trade_size = 150          # Base size for leg trades
            self.etf_trade_size = 200           # Base size for ETF
            
            # ETF ARBITRAGE
            self.etf_arb_threshold = 0.20       # Min edge for ETF arb
            self.etf_arb_qty = 200              # ETF arb quantity
            
            # MARKET MAKING
            self.mm_aaa_spread = 0.15           # AAA bid-ask spread
            self.mm_bbb_spread = 0.18           # BBB bid-ask spread
            self.mm_ccc_spread = 0.20           # CCC bid-ask spread
            self.mm_etf_spread = 0.10           # ETF bid-ask spread
            self.mm_quote_size = 100            # Quote size per side
            self.inventory_skew_divisor = 30000 # Lower = more aggressive flatten
            
            # ALPHA PARAMETERS
            self.aaa_window = 30                # AAA rolling window
            self.bbb_ema_fast = 8               # BBB fast EMA
            self.bbb_ema_slow = 25              # BBB slow EMA
            self.ccc_window = 40                # CCC mean reversion window
            self.ccc_deviation_threshold = 0.015 # CCC deviation to trigger
            
            # OPERATIONAL
            self.cancel_frequency = 90          # Cancel orders every N ticks
            
        elif aggressiveness == "moderate":
            # RISK MANAGEMENT
            self.hard_position_limit = 3500
            self.per_symbol_soft_limit = 2500
            self.per_trade_max = 500
            
            # SIGNAL THRESHOLDS
            self.signal_threshold = 0.5
            self.aaa_shock_zscore = 2.5
            
            # POSITION SIZING
            self.base_trade_size = 250
            self.etf_trade_size = 300
            
            # ETF ARBITRAGE
            self.etf_arb_threshold = 0.08
            self.etf_arb_qty = 300
            
            # MARKET MAKING
            self.mm_aaa_spread = 0.12
            self.mm_bbb_spread = 0.14
            self.mm_ccc_spread = 0.16
            self.mm_etf_spread = 0.08
            self.mm_quote_size = 150
            self.inventory_skew_divisor = 40000
            
            # ALPHA PARAMETERS
            self.aaa_window = 30
            self.bbb_ema_fast = 8
            self.bbb_ema_slow = 25
            self.ccc_window = 40
            self.ccc_deviation_threshold = 0.012
            
            # OPERATIONAL
            self.cancel_frequency = 60
            
        elif aggressiveness == "aggressive":
            # RISK MANAGEMENT
            self.hard_position_limit = 5000
            self.per_symbol_soft_limit = 3500
            self.per_trade_max = 800
            
            # SIGNAL THRESHOLDS
            self.signal_threshold = 0.25
            self.aaa_shock_zscore = 2.0
            
            # POSITION SIZING
            self.base_trade_size = 350
            self.etf_trade_size = 450
            
            # ETF ARBITRAGE
            self.etf_arb_threshold = 0.06
            self.etf_arb_qty = 400
            
            # MARKET MAKING
            self.mm_aaa_spread = 0.10
            self.mm_bbb_spread = 0.12
            self.mm_ccc_spread = 0.14
            self.mm_etf_spread = 0.06
            self.mm_quote_size = 200
            self.inventory_skew_divisor = 50000
            
            # ALPHA PARAMETERS
            self.aaa_window = 30
            self.bbb_ema_fast = 8
            self.bbb_ema_slow = 25
            self.ccc_window = 40
            self.ccc_deviation_threshold = 0.010
            
            # OPERATIONAL
            self.cancel_frequency = 20
        else:
            raise ValueError(f"Unknown aggressiveness level: {aggressiveness}")
    
    def print_config(self):
        """Print current configuration."""
        print("\n=== BOT CONFIGURATION ===")
        print(f"Position Limits: hard={self.hard_position_limit}, per_trade={self.per_trade_max}")
        print(f"Signal Threshold: {self.signal_threshold}")
        print(f"ETF Arb: threshold={self.etf_arb_threshold}, qty={self.etf_arb_qty}")
        print(f"Spreads: AAA={self.mm_aaa_spread}, BBB={self.mm_bbb_spread}, CCC={self.mm_ccc_spread}")
        print(f"Trade Sizes: base={self.base_trade_size}, etf={self.etf_trade_size}")
        print("========================\n")

# =========================
# HTTP helpers
# =========================
def build_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key, "Content-Type": "application/json"}

import time, random

def _raise(resp: requests.Response) -> None:
    if 200 <= resp.status_code < 300: 
        return
    try:
        detail = resp.json().get("detail","")
    except Exception:
        detail = resp.text
    raise RuntimeError(f"HTTP {resp.status_code}: {detail}")

def api_get(base: str, path: str, key: str, params: Optional[dict[str, Any]]=None) -> Any:
    # retry with exponential backoff on 429
    for attempt in range(4):
        r = requests.get(f"{base}{path}", headers=build_headers(key), params=params, timeout=10)
        if r.status_code == 429:
            time.sleep(0.2 * (2**attempt) + random.random() * 0.1)
            continue
        _raise(r)
        return r.json()
    # last attempt
    _raise(r); return r.json()

def api_post(base: str, path: str, key: str, body: dict[str, Any]) -> Any:
    for attempt in range(4):
        r = requests.post(f"{base}{path}", headers=build_headers(key), data=json.dumps(body), timeout=10)
        if r.status_code == 429:
            time.sleep(0.2 * (2**attempt) + random.random() * 0.1)
            continue
        _raise(r)
        return r.json()
    _raise(r); return r.json()

def api_delete(base: str, path: str, key: str) -> Any:
    for attempt in range(4):
        r = requests.delete(f"{base}{path}", headers=build_headers(key), timeout=10)
        if r.status_code == 429:
            time.sleep(0.2 * (2**attempt) + random.random() * 0.1)
            continue
        _raise(r)
        try: 
            return r.json()
        except Exception: 
            return {}
    _raise(r); 
    try: return r.json()
    except Exception: return {}


# =========================
# Exchange wrappers
# =========================
def get_symbols(api, key) -> List[str]:
    data = api_get(api, "/api/v1/symbols", key)
    return [s["symbol"] for s in data.get("symbols", [])]

def get_orderbook(api, key, symbol) -> dict:
    return api_get(api, f"/api/v1/orderbook/{symbol}", key)

def get_market_trades(api, key, symbol: Optional[str]=None, limit: int=50) -> List[dict]:
    params = {"symbol": symbol, "limit": limit} if symbol else {"limit": limit}
    data = api_get(api, "/api/v1/trades/market", key, params=params)
    return data.get("trades", [])

def place_order(api, key, symbol, side, qty, order_type="limit", price: Optional[float]=None) -> Optional[str]:
    body = {"symbol": symbol, "side": side, "order_type": order_type, "quantity": int(qty)}
    if price is not None: body["price"] = round(float(price), 2)
    try:
        res = api_post(api, "/api/v1/orders", key, body)
        oid = res.get("order_id")
        print(f"[ORDER] {side.upper():4} {symbol:4} x{qty} {order_type} @{price if price else 'MKT'} | id={oid}")
        return oid
    except Exception as e:
        print(f"[ERR] order {symbol} {side} {qty}: {e}")
        return None

def cancel_all(api, key):
    try:
        api_delete(api, "/api/v1/orders/all", key)
        print("[CANCEL] all")
    except Exception as e:
        print(f"[ERR] cancel_all: {e}")

def get_positions(api, key) -> Dict[str,int]:
    data = api_get(api, "/api/v1/positions", key)
    out = {}
    for row in data.get("positions", []):
        sym = row.get("symbol")
        pos = row.get("position", row.get("qty", row.get("quantity", 0)))
        try:
            out[sym] = int(pos)
        except Exception:
            out[sym] = 0
    return out


def etf_create(api, key, symbol, quantity: int) -> bool:
    try:
        api_post(api, f"/api/v1/etf/{symbol}/create", key, {"quantity": int(quantity)})
        print(f"[ETF] CREATE {quantity} {symbol}")
        return True
    except Exception as e:
        print(f"[ERR] ETF create: {e}")
        return False

def etf_redeem(api, key, symbol, quantity: int) -> bool:
    try:
        api_post(api, f"/api/v1/etf/{symbol}/redeem", key, {"quantity": int(quantity)})
        print(f"[ETF] REDEEM {quantity} {symbol}")
        return True
    except Exception as e:
        print(f"[ERR] ETF redeem: {e}")
        return False

# =========================
# Pricing utilities
# =========================
def best_mid(ob: dict) -> Optional[float]:
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    if not bids or not asks: return None
    bp = float(bids[0]["price"]); ap = float(asks[0]["price"])
    if ap <= 0: return None
    return round((bp + ap) / 2.0, 2)

def last_trade_price(trades: List[dict]) -> Optional[float]:
    if not trades: return None
    return round(float(trades[0]["price"]), 2)

def safe_price(api, key, symbol) -> Optional[float]:
    # prefer NBBO mid, fallback to last trade
    try:
        ob = get_orderbook(api, key, symbol)  # if you can, swap this call to Trader.get_orderbook_cached
        mid = best_mid(ob)
        if mid is not None: return mid
    except Exception as e:
        print(f"[WARN] orderbook {symbol}: {e}")
    try:
        lt = last_trade_price(get_market_trades(api, key, symbol, 1))
        return lt
    except Exception as e:
        print(f"[WARN] last trade {symbol}: {e}")
        return None



# =========================
# Signal engines
# =========================
class RollingStats:
    def __init__(self, n: int):
        self.n=n; self.buf=deque(maxlen=n)
    def push(self, x: float):
        self.buf.append(x)
    def mean(self)->Optional[float]:
        return (sum(self.buf)/len(self.buf)) if self.buf else None
    def std(self)->Optional[float]:
        if len(self.buf) < 2: return None
        return stats.pstdev(self.buf)

def ema(prev: Optional[float], x: float, alpha: float) -> float:
    return x if prev is None else (alpha*x + (1-alpha)*prev)

class AlphaAAA:
    """Shock detector for AAA: return zscore & direction on jumps."""
    def __init__(self, w=30, zscore_threshold=3.0):
        self.ret = RollingStats(w)
        self.vol = RollingStats(w)
        self.last = None
        self.signal_decay = 0
        self.zscore_threshold = zscore_threshold

    def update(self, px: Optional[float]) -> float:
        if px is None: return 0.0
        if self.last is not None and self.last > 0:
            r = (px - self.last)/self.last
            self.ret.push(r)
            self.vol.push(abs(r))
            mu = self.ret.mean() or 0.0
            sd = self.ret.std() or (self.vol.mean() or 1e-4)
            z = 0.0 if sd==0 else (r - mu)/sd
            # Keep a short-lived impulse
            if abs(z) > self.zscore_threshold:
                self.signal_decay = 30  # ~30 ticks
                sgn = 1.0 if z>0 else -1.0
                self.last = px
                return 2.0*sgn  # strong one-shot
            if self.signal_decay>0:
                self.signal_decay -= 1
                return 0.6 if r>0 else -0.6
        self.last = px
        return 0.0

class AlphaBBB:
    """Improved momentum with stronger signals."""
    def __init__(self, fast=8, slow=25):
        self.ema_f=None; self.ema_s=None
        self.fast=fast; self.slow=slow

    def update(self, px: Optional[float]) -> float:
        if px is None: return 0.0
        af=2/(self.fast+1); aslow=2/(self.slow+1)
        self.ema_f = ema(self.ema_f, px, af)
        self.ema_s = ema(self.ema_s, px, aslow)
        if self.ema_f is None or self.ema_s is None: return 0.0
        slope = self.ema_f - self.ema_s
        # Follow momentum more aggressively
        sig = max(-1.5, min(1.5, slope / max(0.03*self.ema_s, 0.01)))
        return sig  # Full signal strength

class AlphaCCC:
    """Mean reversion on volatility expansion."""
    def __init__(self, window=40, deviation_threshold=0.015):
        self.window=deque(maxlen=window)
        self.vol_window=deque(maxlen=20)
        self.last_px=None
        self.deviation_threshold = deviation_threshold

    def update(self, px: Optional[float]) -> float:
        if px is None: return 0.0
        self.window.append(px)
        if len(self.window) < 20: return 0.0
        
        # Track volatility
        if self.last_px is not None:
            ret = abs(px - self.last_px) / self.last_px if self.last_px > 0 else 0
            self.vol_window.append(ret)
        self.last_px = px
        
        # Mean reversion signal when far from moving average
        ma = sum(self.window) / len(self.window)
        deviation = (px - ma) / ma if ma > 0 else 0
        
        # Fade extremes (mean revert)
        if deviation > self.deviation_threshold:  # Price too high
            return -1.2  # Sell signal
        elif deviation < -self.deviation_threshold:  # Price too low
            return 1.2  # Buy signal
        return 0.0

# =========================
# Risk & sizing
# =========================
class Risk:
    def __init__(self, config: BotConfig):
        self.hard = config.hard_position_limit
        self.soft = config.per_symbol_soft_limit
        self.per_trade = config.per_trade_max

    def clamp_qty(self, desired: int, pos: int) -> int:
        # Keep within hard limit and pull toward 0 inventory
        max_buy = self.hard - pos
        max_sell = self.hard + pos  # if pos negative, can buy more; if positive, can sell more
        qty = desired
        if qty>0: qty = min(qty, max(1, max_buy))
        if qty<0: qty = -min(-qty, max(1, max_sell))
        qty = max(-self.per_trade, min(self.per_trade, qty))
        return qty

# =========================
# Trader
# =========================
class Trader:
    def __init__(self, api, key, config: BotConfig):
        self.api = api
        self.key = key
        self.config = config
        self.symbols = ["AAA", "BBB", "CCC", "ETF"]
        self.alpha = {
            "AAA": AlphaAAA(w=config.aaa_window, zscore_threshold=config.aaa_shock_zscore),
            "BBB": AlphaBBB(fast=config.bbb_ema_fast, slow=config.bbb_ema_slow),
            "CCC": AlphaCCC(window=config.ccc_window, deviation_threshold=config.ccc_deviation_threshold),
        }
        self.risk = Risk(config)
        self.positions = defaultdict(int)
        self.tick = 0

        # Add these right after your existing init variables
        from collections import deque
        import statistics as stats
        self.spreads = {s: deque(maxlen=50) for s in self.symbols}
        self.cooldown = {s: 0 for s in self.symbols}
        self.max_abs_spread = {"ETF": 0.40, "BBB": 0.60, "CCC": 0.60, "AAA": 1.0}
        self.spread_mult = 12
        self.min_top_depth = 0
        self.cooldown_ticks = 10  # ~20s if hz=2

        self.book_cache = {}  # symbol -> {"ob": dict, "ts": float}
        self.book_ttl = 0.20  # seconds to reuse an orderbook snapshot

        # quote aging & improvement
        self.last_quote_age = {s: 0 for s in self.symbols}  # ticks we’ve held same quote
        self.improve_after_ticks = 3   # nudge after ~5s if hz=2
        self.max_improve_ticks   = 12    # don’t chase more than 5 ticks total
        self.tick_size = 0.01


        import time

        # remember last posted quotes and pacing
        self.last_quote = {s: None for s in self.symbols}      # symbol -> (bid, ask, qty)
        self.last_order_ts = defaultdict(float)                 # symbol -> last send time
        self.global_last_ts = 0.0
        self.min_symbol_interval = 0.20   # seconds between orders per symbol (tune 0.3–0.6)
        self.min_global_interval = 0.08   # global pacing (tune 0.05–0.15)

    def refresh_positions(self):
        """Refresh self.positions from the exchange; be tolerant of schema."""
        try:
            data = api_get(self.api, "/api/v1/positions", self.key)
            out = {}
            for row in data.get("positions", []):
                sym = row.get("symbol")
                pos = row.get("position", row.get("qty", row.get("quantity", 0)))
                try:
                    out[sym] = int(pos)
                except Exception:
                    out[sym] = 0
            self.positions.update(out)
        except Exception as e:
            print(f"[WARN] get positions: {e}")

    def basket_fair(self) -> Optional[float]:
        """
        Compute ETF basket fair as AAA + BBB + CCC with sanity guards so a bad leg
        doesn't trigger dumb ETF arb. Returns None if prices look broken.
        """
        # Case anchors (initial fair levels)
        anchors = {"AAA": 50.0, "BBB": 25.0, "CCC": 75.0}

        # Pull a reference price per leg (NBBO mid preferred; falls back to last trade)
        pa = safe_price(self.api, self.key, "AAA")
        pb = safe_price(self.api, self.key, "BBB")
        pc = safe_price(self.api, self.key, "CCC")

        legs = {"AAA": pa, "BBB": pb, "CCC": pc}
        if any(v is None or v <= 0 for v in legs.values()):
            return None

        # Sanity band: accept only prices within 0.2×–5× their anchors
        for sym, px in legs.items():
            lo = 0.2 * anchors[sym]
            hi = 5.0 * anchors[sym]
            if not (lo <= px <= hi):
                return None  # skip this tick if any leg looks absurd

        return round(legs["AAA"] + legs["BBB"] + legs["CCC"], 2)


    def place_basket_orders(self, side: str, qty: int):
        """
        Place orders for 1 unit each of AAA, BBB, CCC to hedge ETF arbitrage.
        Uses passive limit orders at favorable prices.
        """
        for symbol in ["AAA", "BBB", "CCC"]:
            px = safe_price(self.api, self.key, symbol)
            if px is not None:
                # Clamp quantity to respect position limits
                pos = self.positions.get(symbol, 0)
                clamped_qty = self.risk.clamp_qty(qty if side == "buy" else -qty, pos)
                
                if abs(clamped_qty) > 0:
                    # Passive execution: offer below market when buying, above when selling
                    limit_px = px - 0.01 if side == "buy" else px + 0.01
                    limit_px = max(0.01, round(limit_px, 2))
                    place_order(self.api, self.key, symbol, side, abs(clamped_qty), "limit", limit_px)

    def etf_arb(self):
        """
        Enhanced ETF arbitrage: simultaneously trade ETF and basket components
        to create a true market-neutral arbitrage position.
        """
        # only consider ETF arb every 5 ticks to avoid spam
        if self.tick % 5 != 0:
            return
        # and also pace by time
        if not self._should_send("ETF"):
            return

        fair = self.basket_fair()
        etf = safe_price(self.api, self.key, "ETF")
        if fair is None or etf is None:
            return

        edge = fair - etf
        threshold = self.config.etf_arb_threshold

        # Determine arbitrage quantity, respecting risk limits
        arb_qty = min(self.config.etf_arb_qty, self.risk.per_trade)

        # Post at the touch *passively*, never chase
        if edge > threshold:
            # ETF cheap vs basket → BUY ETF, SELL basket
            etf_qty = self.risk.clamp_qty(+arb_qty, self.positions.get("ETF", 0))
            if etf_qty > 0:
                # Place ETF buy order
                px = max(0.01, round(etf - 0.01, 2))
                place_order(self.api, self.key, "ETF", "buy", etf_qty, "limit", px)
                # Simultaneously sell the basket components
                self.place_basket_orders("sell", etf_qty)
                
        elif edge < -threshold:
            # ETF rich → SELL ETF, BUY basket
            etf_qty = self.risk.clamp_qty(-arb_qty, self.positions.get("ETF", 0))
            if etf_qty < 0:
                # Place ETF sell order
                px = round(etf + 0.01, 2)
                place_order(self.api, self.key, "ETF", "sell", -etf_qty, "limit", px)
                # Simultaneously buy the basket components
                self.place_basket_orders("buy", -etf_qty)
        
        self._mark_sent("ETF")


    def inventory_maker(self, symbol: str, target_spread: Optional[float] = None):
        """
        Maker that joins touch, adapts to inventory, and nudges tighter if quote ages.
        """
        mid, spread, healthy = self.orderbook_health(symbol)
        if mid is None or not healthy:
            return

        # Pull current book for touch prices from cache (no extra API calls)
        ob = self.get_orderbook_cached(symbol)
        bids = ob.get("bids", []); asks = ob.get("asks", [])
        if not bids or not asks:
            return
        bp = round(float(bids[0]["price"]), 2)
        ap = round(float(asks[0]["price"]), 2)

        pos = self.positions.get(symbol, 0)

        # Base target spread from config
        if target_spread is None:
            target_spread = getattr(self.config, f"mm_{symbol.lower()}_spread", 0.12)

        # Widen for inventory (your existing idea)
        inv_adj = 0.05 * (abs(pos) / self.config.per_symbol_soft_limit + 1.0)
        target_spread += inv_adj

        # Start from mid +/- half, then ensure we at least JOIN the touch
        half = max(self.tick_size, target_spread / 2.0)
        skew = max(-0.06, min(0.06, -pos / self.config.inventory_skew_divisor))
        desired_bid = round(mid - half + skew, 2)
        desired_ask = round(mid + half + skew, 2)

        # Join touch: don’t sit inside the book if touch is better for fills
        bid = max(bp, desired_bid)
        ask = min(ap, desired_ask)
        if ask <= bid:
            ask = round(bid + 2 * self.tick_size, 2)

        # Age-based improvement: if we’ve sat at the same quote too long, step 1 tick toward the other side
        prev = self.last_quote.get(symbol)
        if prev is not None:
            pb, pa, pq = prev
            if abs(pb - bid) < self.tick_size and abs(pa - ask) < self.tick_size:
                self.last_quote_age[symbol] += 1
            else:
                self.last_quote_age[symbol] = 0
        else:
            self.last_quote_age[symbol] = 0

        if self.last_quote_age[symbol] >= self.improve_after_ticks:
            # improve by 1 tick toward the market to grab queue priority
            bid = min(bid + self.tick_size, ap - self.tick_size)
            ask = max(ask - self.tick_size, bp + self.tick_size)
            # cap total improvements so we don’t chase indefinitely
            self.last_quote_age[symbol] = min(self.last_quote_age[symbol], self.improve_after_ticks + self.max_improve_ticks)

        # Size down when market is wide
        qty = self.config.mm_quote_size
        if spread is not None and len(self.spreads[symbol]) >= 5:
            import statistics as statistics
            med = statistics.median(self.spreads[symbol])
            if spread > 4.0 * med:
                qty = max(1, int(0.25 * qty))
            elif spread > 2.0 * med:
                qty = max(1, int(0.5 * qty))

        # De-dup & pacing
        if not self._quote_changed(symbol, bid, ask, qty):
            return
        if not self._should_send(symbol):
            return

        place_order(self.api, self.key, symbol, "buy", qty, "limit", bid)
        place_order(self.api, self.key, symbol, "sell", qty, "limit", ask)
        self.last_quote[symbol] = (bid, ask, qty)
        self._mark_sent(symbol)



    def signal_trade(self, symbol: str):
        """
        For signals:
        - Gate by orderbook health & cooldown
        - If |sig| is strong, take a small slice to ensure entry
        - Post the remainder passively at/near the touch (join)
        """
        mid, spread, healthy = self.orderbook_health(symbol)
        if not healthy:
            return

        # Reference price and current touch
        ob = self.get_orderbook_cached(symbol)
        if not ob or not ob.get("bids") or not ob.get("asks"):
            return
        bp = round(float(ob["bids"][0]["price"]), 2)
        ap = round(float(ob["asks"][0]["price"]), 2)
        px = round((bp + ap) / 2.0, 2)

        # Cooldown from earlier blowouts
        if self.cooldown[symbol] > 0:
            self.cooldown[symbol] -= 1
            return

        # Update alpha & gate by threshold
        sig = self.alpha[symbol].update(px)
        if abs(sig) < self.config.signal_threshold:
            # maintain makers only
            self.inventory_maker(symbol)
            return

        side = "buy" if sig > 0 else "sell"
        pos = self.positions.get(symbol, 0)
        base = self.config.base_trade_size if symbol != "ETF" else self.config.etf_trade_size

        # Desired size proportional to signal, lightly capped
        desired = max(1, int(base * min(1.5, abs(sig))))

        # Scale size down in wide spreads
        if spread is not None and len(self.spreads[symbol]) >= 5:
            import statistics as statistics
            med = statistics.median(self.spreads[symbol])
            if spread > 4.0 * med:
                desired = max(1, int(0.25 * desired))
            elif spread > 2.0 * med:
                desired = max(1, int(0.5 * desired))

        qty = self.risk.clamp_qty(desired if side == "buy" else -desired, pos)
        if qty == 0:
            return

        # 🔥 Partial taker only when signal is much stronger than threshold
        strong = abs(sig) >= (self.config.signal_threshold + 0.2)
        taker_slice = int(0.2 * abs(qty))
        if strong:
            taker_slice = max(0, min(abs(qty), int(0.2 * abs(qty))))  # 20% slice, up to qty
            if taker_slice >= 1 and self._should_send(symbol):
                place_order(self.api, self.key, symbol, side, taker_slice, "market")
                self._mark_sent(symbol)
                qty -= taker_slice if side == "buy" else -taker_slice
                if qty == 0:
                    return

        # Post remainder by JOINING touch (best chance to get picked up next)
        if not self._should_send(symbol):
            return

        if side == "buy":
            limit_px = min(ap - self.tick_size, bp)  # join best bid; never cross
            place_order(self.api, self.key, symbol, "buy", abs(qty), "limit", round(limit_px, 2))
        else:
            limit_px = max(bp + self.tick_size, ap)  # join best ask; never cross
            place_order(self.api, self.key, symbol, "sell", abs(qty), "limit", round(limit_px, 2))

        self._mark_sent(symbol)

        # Keep makers running to maintain presence
        self.inventory_maker(symbol)




    

    def step(self):
        self.tick += 1
        self.refresh_positions()

        # ETF arb every few ticks to avoid spam (tune as needed)
        if self.tick % 3 == 0:
            self.etf_arb()

        # Run signals on AAA *and* BBB (BBB has a momentum alpha wired already)
        self.signal_trade("AAA")
        self.signal_trade("BBB")

        # Always keep makers up on all three legs
        self.inventory_maker("AAA")
        self.inventory_maker("BBB")
        self.inventory_maker("CCC")

        # Periodic cleanup (not too frequent so we can hold queue priority)
        if self.tick % self.config.cancel_frequency == 0:
            cancel_all(self.api, self.key)

        # Optional: debug
        if hasattr(self, "debug_every") and self.tick % self.debug_every == 0:
            print("[DBG] skips:", getattr(self, "debug_counters", {}))
            if hasattr(self, "debug_counters"):
                self.debug_counters = {k: 0 for k in self.debug_counters}




    def _should_send(self, symbol: str) -> bool:
        now = time.time()
        if now - self.global_last_ts < self.min_global_interval:
            return False
        if now - self.last_order_ts[symbol] < self.min_symbol_interval:
            return False
        return True

    def _mark_sent(self, symbol: str) -> None:
        now = time.time()
        self.last_order_ts[symbol] = now
        self.global_last_ts = now

    def _quote_changed(self, symbol: str, bid: float, ask: float, qty: int) -> bool:
        prev = self.last_quote.get(symbol)
        # only re-post if price moved >= 1 tick (0.01) or size changed
        if prev is None:
            return True
        pb, pa, pq = prev
        return (abs(bid - pb) >= 0.01) or (abs(ask - pa) >= 0.01) or (qty != pq)
    
    def get_orderbook_cached(self, symbol: str) -> Optional[dict]:
        now = time.time()
        ent = self.book_cache.get(symbol)
        if ent and (now - ent["ts"] <= self.book_ttl):
            return ent["ob"]
        try:
            ob = get_orderbook(self.api, self.key, symbol)
            self.book_cache[symbol] = {"ob": ob, "ts": now}
            return ob
        except Exception as e:
            print(f"[WARN] orderbook_cached {symbol}: {e}")
            return ent["ob"] if ent else None
    
    def orderbook_health(self, symbol: str):
        """
        Returns (mid, spread, healthy).
        Healthy if:
        - both sides present
        - absolute spread <= per-symbol cap
        - relative spread <= spread_mult * rolling median (when we have history)
        - top-of-book has some depth (can be 0 if you set min_top_depth = 0)
        Uses get_orderbook_cached() to avoid spamming the API.
        """
        ob = self.get_orderbook_cached(symbol)
        if not ob:
            return None, None, False

        bids = ob.get("bids", []); asks = ob.get("asks", [])
        if not bids or not asks:
            return None, None, False

        try:
            bp = float(bids[0]["price"]); ap = float(asks[0]["price"])
            bsz = int(bids[0].get("quantity", bids[0].get("size", 0)) or 0)
            asz = int(asks[0].get("quantity", asks[0].get("size", 0)) or 0)
        except Exception:
            return None, None, False

        if ap <= 0 or bp <= 0 or ap <= bp:
            return None, None, False

        spread = round(ap - bp, 2)
        mid = round((ap + bp) / 2.0, 2)

        # Track rolling spread
        self.spreads[symbol].append(spread)
        import statistics as statistics
        if len(self.spreads[symbol]) >= 5:
            med = statistics.median(self.spreads[symbol])
        else:
            med = spread  # bootstrap until we have history

        # Compute checks BEFORE any early-return branches
        abs_ok   = spread <= self.max_abs_spread.get(symbol, 0.50)
        rel_ok   = (spread <= self.spread_mult * med)  # only meaningful once we have history
        depth_ok = (bsz >= self.min_top_depth and asz >= self.min_top_depth)

        # If we have little history, be permissive: use only absolute & depth checks
        if len(self.spreads[symbol]) < 5:
            healthy = abs_ok and depth_ok
            return mid, spread, healthy

        # Normal path: require all three
        healthy = abs_ok and rel_ok and depth_ok
        return mid, spread, healthy




# =========================
# CLI / main loop
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="CTC Systematic Equities Bot")
    p.add_argument("--api-url", default=os.environ.get("CTC_API_URL", "https://cornelltradingcompetition.org"))
    p.add_argument("--api-key", default=os.environ.get("CTC_API_KEY", "ISNeO-lq0Mrnm3zLDuMqUejrkXc5CMVq7wAEAjIDh20") or os.environ.get("X_API_KEY"))
    p.add_argument("--loop", action="store_true", help="Run continuously")
    p.add_argument("--hz", type=float, default=2.0, help="ticks per second")
    p.add_argument("--mode", choices=["conservative", "moderate", "aggressive"], 
                   default="conservative", help="Trading aggressiveness level")
    # Allow fine-grained overrides
    p.add_argument("--signal-threshold", type=float, help="Override signal threshold")
    p.add_argument("--position-limit", type=int, help="Override hard position limit")
    p.add_argument("--etf-arb-threshold", type=float, help="Override ETF arb threshold")
    p.add_argument("--mm-quote-size", type=int, help="Override MM quote size")
    p.add_argument("--base-trade-size", type=int, help="Override base signal size")
    p.add_argument("--etf-arb-qty", type=int, help="Override ETF arb order size")
    p.add_argument("--per-trade-max", type=int, help="Override max size per individual order")

    return p.parse_args()

def main():
    args = parse_args()
    api=args.api_url
    key=args.api_key or input("API key: ").strip()
    if not key:
        print("API key required."); return 1
    
    # Create config
    config = BotConfig(aggressiveness=args.mode)
    
    # Apply command-line overrides
    if args.signal_threshold is not None:
        config.signal_threshold = args.signal_threshold
    if args.position_limit is not None:
        config.hard_position_limit = args.position_limit
    if args.etf_arb_threshold is not None:
        config.etf_arb_threshold = args.etf_arb_threshold
    if args.mm_quote_size is not None:
        config.mm_quote_size = args.mm_quote_size
    if args.base_trade_size is not None:
        config.base_trade_size = args.base_trade_size
    if args.etf_arb_qty is not None:
        config.etf_arb_qty = args.etf_arb_qty
    if args.per_trade_max is not None:
        config.per_trade_max = args.per_trade_max


    
    config.print_config()
    
    trader = Trader(api, key, config)
    print("[INIT] symbols:", get_symbols(api, key))
    try:
        if args.loop:
            period = 1.0/args.hz
            while True:
                t0=time.time()
                trader.step()
                dt=time.time()-t0
                time.sleep(max(0.0, period-dt))
        else:
            trader.step()
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl-C")
    finally:
        cancel_all(api, key)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
