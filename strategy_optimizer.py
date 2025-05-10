import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm import tqdm
import env

# Ajout de l'import MetaTrader5
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Template de base pour le fichier MQ5
MQ5_TEMPLATE = """
//+------------------------------------------------------------------+
//|                                              OptimizedStrategy.mq5 |
//|                                  Copyright 2024, Votre Nom         |
//|                                                                   |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

CTrade trade;

// Paramètres d'entrée
input double   LotSize = {lot_size};           // Taille du lot
input int      StopLoss = {stop_loss};         // Stop Loss en pips
input int      TakeProfit = {take_profit};     // Take Profit en pips
input bool     UseTrailingStop = {trailing_stop};  // Utiliser Trailing Stop
input int      TrailingDistance = {trailing_distance};  // Distance du Trailing Stop
input int      MaxTradesPerDay = {max_trades_per_day};  // Nombre max de trades par jour
input int      StartHour = {entry_hour_min};    // Heure de début de trading
input int      EndHour = {entry_hour_max};      // Heure de fin de trading
input bool     WeekendTrading = {weekend_trading};  // Trading le weekend
input double   MaxDrawdownPercent = {max_drawdown_exit};  // Drawdown max en %
input int      MaxConsecutiveLosses = {consecutive_losses_exit};  // Pertes consécutives max
input double   DailyProfitTarget = {profit_target_daily};  // Objectif de profit quotidien en %

// Variables globales
int handle;
int tradesToday = 0;
double initialBalance;
datetime lastTradeDate;
int consecutiveLosses = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{{
    // Initialisation
    initialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    lastTradeDate = 0;
    
    return(INIT_SUCCEEDED);
}}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{{
    // Nettoyage
}}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{{
    // Vérifier si on peut trader
    if(!CanTrade()) return;
    
    // Vérifier les conditions d'entrée
    if(CanOpenNewPosition())
    {{
        if({direction_condition})
        {{
            OpenPosition();
        }}
    }}
    
    // Gérer les positions existantes
    ManageOpenPositions();
}}

//+------------------------------------------------------------------+
//| Vérifie si on peut trader                                         |
//+------------------------------------------------------------------+
bool CanTrade()
{{
    datetime currentTime = TimeCurrent();
    MqlDateTime time;
    TimeToStruct(currentTime, time);
    
    // Vérifier l'heure de trading
    if(time.hour < StartHour || time.hour > EndHour)
        return false;
        
    // Vérifier le weekend
    if(!WeekendTrading && (time.day_of_week == 0 || time.day_of_week == 6))
        return false;
        
    // Vérifier le nombre de trades par jour
    MqlDateTime lastTime;
    TimeToStruct(lastTradeDate, lastTime);
    if(time.day != lastTime.day)
    {{
        tradesToday = 0;
    }}
    
    if(tradesToday >= MaxTradesPerDay)
        return false;
        
    // Vérifier le drawdown
    double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double drawdown = (initialBalance - currentBalance) / initialBalance * 100;
    if(drawdown > MaxDrawdownPercent)
        return false;
        
    return true;
}}

//+------------------------------------------------------------------+
//| Vérifie si on peut ouvrir une nouvelle position                   |
//+------------------------------------------------------------------+
bool CanOpenNewPosition()
{{
    if(PositionsTotal() > 0)
        return false;
        
    if(consecutiveLosses >= MaxConsecutiveLosses)
        return false;
        
    return true;
}}

//+------------------------------------------------------------------+
//| Conditions de signal                                              |
//+------------------------------------------------------------------+
bool IsLongSignal()
{{
    // À implémenter selon votre stratégie
    return false;
}}

bool IsShortSignal()
{{
    // À implémenter selon votre stratégie
    return false;
}}

//+------------------------------------------------------------------+
//| Ouvre une nouvelle position                                       |
//+------------------------------------------------------------------+
void OpenPosition()
{{
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    if({direction} == "long")
    {{
        double sl = ask - StopLoss * _Point;
        double tp = ask + TakeProfit * _Point;
        trade.Buy(LotSize, _Symbol, ask, sl, tp, "Optimized Strategy");
    }}
    else if({direction} == "short")
    {{
        double sl = bid + StopLoss * _Point;
        double tp = bid - TakeProfit * _Point;
        trade.Sell(LotSize, _Symbol, bid, sl, tp, "Optimized Strategy");
    }}
    
    tradesToday++;
    lastTradeDate = TimeCurrent();
}}

//+------------------------------------------------------------------+
//| Gère les positions ouvertes                                       |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{{
    if(!UseTrailingStop) return;
    
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {{
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {{
            if(PositionGetString(POSITION_SYMBOL) == _Symbol)
            {{
                double currentSL = PositionGetDouble(POSITION_SL);
                double currentTP = PositionGetDouble(POSITION_TP);
                double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
                double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
                
                if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                {{
                    double newSL = currentPrice - TrailingDistance * _Point;
                    if(newSL > currentSL)
                    {{
                        trade.PositionModify(PositionGetTicket(i), newSL, currentTP);
                    }}
                }}
                else
                {{
                    double newSL = currentPrice + TrailingDistance * _Point;
                    if(newSL < currentSL || currentSL == 0)
                    {{
                        trade.PositionModify(PositionGetTicket(i), newSL, currentTP);
                    }}
                }}
            }}
        }}
    }}
}}
"""

class StrategyOptimizer:
    def __init__(self, data_file, initial_balance=env.INITIAL_BALANCE, n_strategies=env.N_STRATEGIES, currency_pair=env.CURRENCY_PAIR):
        """
        Initialise l'optimiseur de stratégie avec les données et paramètres de base
        
        Args:
            data_file (str): Chemin vers le fichier de données historiques (csv)
            initial_balance (float): Solde initial du compte
            n_strategies (int): Nombre de stratégies aléatoires à tester
            currency_pair (str): Paire de devises utilisée
        """
        self.data_file = data_file
        self.initial_balance = initial_balance
        self.n_strategies = n_strategies
        self.currency_pair = currency_pair
        self.best_strategy = None
        self.best_performance = -float('inf')
        self.results = []
        
        # Chargement des données
        self.load_data()
        
    def load_data(self):
        """Charge les données historiques depuis le fichier CSV, ou récupère depuis MT5 si le fichier n'existe pas"""
        if os.path.exists(self.data_file):
            try:
                self.data = pd.read_csv(self.data_file)
                print(f"Données chargées avec succès: {len(self.data)} lignes")

                # Gestion du format MT5
                if 'Time' in self.data.columns:
                    self.data.rename(columns={'Time': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'}, inplace=True)
                    # Conversion du format de date MT5
                    try:
                        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%Y.%m.%d %H:%M')
                    except Exception as e:
                        print(f"Erreur conversion date MT5: {e}")

                # Vérification des colonnes requises
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
                for col in required_columns:
                    if col not in self.data.columns:
                        # Si les colonnes n'existent pas, on essaie de les déduire ou de les renommer
                        if 'time' in self.data.columns:
                            self.data['Date'] = self.data['time']
                        if 'open' in self.data.columns:
                            self.data['Open'] = self.data['open']
                        if 'high' in self.data.columns:
                            self.data['High'] = self.data['high']
                        if 'low' in self.data.columns:
                            self.data['Low'] = self.data['low']
                        if 'close' in self.data.columns:
                            self.data['Close'] = self.data['close']

                # Convertir la colonne Date au format datetime si nécessaire
                if 'Date' in self.data.columns and not pd.api.types.is_datetime64_any_dtype(self.data['Date']):
                    try:
                        self.data['Date'] = pd.to_datetime(self.data['Date'])
                    except Exception as e:
                        print(f"Impossible de convertir la colonne Date: {e}")

                # Afficher un aperçu des données
                print(self.data.head())
            except Exception as e:
                print(f"Erreur lors du chargement des données: {e}")
                print("Création de données de test synthétiques...")
                self.create_synthetic_data()
        else:
            print(f"Fichier {self.data_file} introuvable. Tentative de récupération depuis MetaTrader 5...")
            self.get_mt5_data()
    
    def get_mt5_data(self, symbol=None, timeframe=None, n_bars=env.N_BARS):
        """Récupère les données historiques depuis MetaTrader 5 et les sauvegarde en CSV"""
        if mt5 is None:
            print("Le module MetaTrader5 n'est pas installé. Installez-le avec 'pip install MetaTrader5'.")
            self.create_synthetic_data()
            return
        if symbol is None:
            symbol = self.currency_pair
        if timeframe is None:
            timeframe = env.TIMEFRAME  # 1 heure par défaut
        # Initialiser la connexion
        if not mt5.initialize():
            print(f"Erreur lors de la connexion à MetaTrader 5: {mt5.last_error()}")
            self.create_synthetic_data()
            return
        print(f"Connexion à MT5 réussie. Téléchargement de {n_bars} bougies pour {symbol}...")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
        if rates is None or len(rates) == 0:
            print("Impossible de récupérer les données depuis MT5. Création de données synthétiques.")
            self.create_synthetic_data()
            mt5.shutdown()
            return
        df = pd.DataFrame(rates)
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
        self.data = df[['Date', 'Open', 'High', 'Low', 'Close']]
        # Sauvegarder pour la prochaine fois
        self.data.to_csv(self.data_file, index=False)
        print(f"Données MT5 récupérées et sauvegardées dans {self.data_file}")
        print(self.data.head())
        mt5.shutdown()
    
    def create_synthetic_data(self, days=365, start_date='2023-01-01'):
        """Crée des données synthétiques pour tester l'algorithme"""
        start_date = pd.to_datetime(start_date)
        dates = pd.date_range(start=start_date, periods=days)
        
        # Générer des prix aléatoires avec une tendance
        close = [100]
        for i in range(1, days):
            close.append(close[-1] + random.uniform(-2, 2) + 0.05)  # légère tendance haussière
        
        # Créer high, low, open basés sur close
        high = [c + random.uniform(0.5, 1.5) for c in close]
        low = [c - random.uniform(0.5, 1.5) for c in close]
        open_prices = [(h + l) / 2 for h, l in zip(high, low)]
        
        # Créer le DataFrame
        self.data = pd.DataFrame({
            'Date': dates,
            'Open': open_prices,
            'High': high,
            'Low': low,
            'Close': close
        })
        
        print("Données synthétiques créées avec succès")
        print(self.data.head())
    
    def generate_random_strategy(self):
        """
        Génère une stratégie aléatoire avec tous les paramètres
        
        Returns:
            dict: Dictionnaire contenant les paramètres de la stratégie
        """
        strategy = {
            # Paramètres de prise de décision
            'entry_type': random.choice(['market', 'limit', 'stop']),
            'direction': random.choice(['long', 'short', 'both']),
            'time_frame': random.choice([1, 5, 15, 30, 60, 240, 1440]),  # minutes
            
            # Paramètres de risque et récompense
            'take_profit': random.uniform(10, 200),  # pips
            'stop_loss': random.uniform(10, 150),  # pips
            'trailing_stop': random.choice([True, False]),
            'trailing_distance': random.uniform(10, 100) if random.random() > 0.5 else None,  # pips
            
            # Paramètres de taille de position
            'lot_size': round(random.uniform(0.01, 1.0), 2),
            'risk_percent': random.uniform(0.5, 5.0),  # pourcentage du capital
            'position_sizing': random.choice(['fixed', 'risk_based']),
            
            # Paramètres d'entrée de marché
            'entry_hour_min': random.randint(0, 23),
            'entry_hour_max': random.randint(0, 23),
            'max_trades_per_day': random.randint(1, 10),
            'weekend_trading': random.choice([True, False]),
            
            # Paramètres supplémentaires
            'max_drawdown_exit': random.uniform(5, 30),  # pourcentage
            'consecutive_losses_exit': random.randint(3, 15),
            'profit_target_daily': random.uniform(0.5, 5.0),  # pourcentage
        }
        
        # S'assurer que entry_hour_max est supérieur à entry_hour_min
        if strategy['entry_hour_min'] > strategy['entry_hour_max']:
            strategy['entry_hour_min'], strategy['entry_hour_max'] = strategy['entry_hour_max'], strategy['entry_hour_min']
            
        return strategy
    
    def backtest_strategy(self, strategy):
        """
        Effectue un backtest de la stratégie sur les données historiques
        
        Args:
            strategy (dict): Paramètres de la stratégie à tester
            
        Returns:
            dict: Résultats du backtest
        """
        # Initialisation des variables de backtest
        balance = self.initial_balance
        initial_balance = balance
        trades = []
        current_position = None
        max_balance = balance
        min_balance = balance
        consecutive_losses = 0
        daily_trades = {}
        
        pip_value = env.PIP_VALUE
        
        # Pour chaque barre de données (simulation)
        for i in range(1, len(self.data) - 1):
            current_bar = self.data.iloc[i]
            next_bar = self.data.iloc[i + 1]
            date = current_bar['Date']
            
            # Vérifier si c'est un jour de semaine (si weekend_trading=False)
            if not strategy['weekend_trading'] and pd.to_datetime(date).weekday() >= 5:
                continue
                
            # Vérifier l'heure de trading
            trade_hour = pd.to_datetime(date).hour
            if not (strategy['entry_hour_min'] <= trade_hour <= strategy['entry_hour_max']):
                continue
                
            # Vérifier le nombre max de trades par jour
            day_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            if day_str in daily_trades and daily_trades[day_str] >= strategy['max_trades_per_day']:
                continue
                
            # Vérifier si on a atteint le max drawdown
            if balance < max_balance * (1 - strategy['max_drawdown_exit'] / 100):
                break
                
            # Vérifier si on a atteint le nombre max de pertes consécutives
            if consecutive_losses >= strategy['consecutive_losses_exit']:
                break
                
            # Si aucune position n'est ouverte, vérifier si on doit en ouvrir une
            if current_position is None:
                # Déterminer la direction du trade
                trade_direction = None
                if strategy['direction'] == 'both':
                    trade_direction = random.choice(['long', 'short'])
                else:
                    trade_direction = strategy['direction']
                
                # Calculer la taille du lot
                lot = strategy['lot_size']
                if strategy['position_sizing'] == 'risk_based':
                    # Calculer la taille du lot basée sur le risque
                    risk_amount = balance * (strategy['risk_percent'] / 100)
                    pip_risk = strategy['stop_loss']
                    # Taille du lot = Montant risqué / (pip_risk * valeur du pip)
                    lot = min(round(risk_amount / (pip_risk * pip_value * env.INITIAL_BALANCE), 2), 1.0)
                    lot = max(lot, env.MIN_LOT_SIZE)
                
                # Simuler une entrée en position
                entry_price = current_bar['Close']
                tp_price = entry_price + (strategy['take_profit'] * pip_value) if trade_direction == 'long' else entry_price - (strategy['take_profit'] * pip_value)
                sl_price = entry_price - (strategy['stop_loss'] * pip_value) if trade_direction == 'long' else entry_price + (strategy['stop_loss'] * pip_value)
                
                current_position = {
                    'entry_date': date,
                    'entry_price': entry_price,
                    'direction': trade_direction,
                    'lot_size': lot,
                    'take_profit': tp_price,
                    'stop_loss': sl_price,
                    'trailing_stop': strategy['trailing_stop'],
                    'trailing_distance': strategy['trailing_distance'],
                    'max_price': entry_price if trade_direction == 'long' else float('inf'),
                    'min_price': entry_price if trade_direction == 'short' else float('-inf')
                }
                
                # Enregistrer ce trade dans la comptabilité quotidienne
                if day_str not in daily_trades:
                    daily_trades[day_str] = 1
                else:
                    daily_trades[day_str] += 1
            
            # Si une position est ouverte, vérifier si elle doit être fermée
            else:
                # Mise à jour du trailing stop si activé
                if current_position['trailing_stop'] and current_position['trailing_distance'] is not None:
                    if current_position['direction'] == 'long' and current_bar['High'] > current_position['max_price']:
                        current_position['max_price'] = current_bar['High']
                        current_position['stop_loss'] = current_position['max_price'] - (current_position['trailing_distance'] * pip_value)
                    elif current_position['direction'] == 'short' and current_bar['Low'] < current_position['min_price']:
                        current_position['min_price'] = current_bar['Low']
                        current_position['stop_loss'] = current_position['min_price'] + (current_position['trailing_distance'] * pip_value)
                
                # Vérifier si le SL ou TP est touché
                exit_price = None
                exit_type = None
                profit_loss = 0
                
                if current_position['direction'] == 'long':
                    # Pour une position longue
                    if next_bar['Low'] <= current_position['stop_loss']:
                        exit_price = current_position['stop_loss']
                        exit_type = 'SL'
                    elif next_bar['High'] >= current_position['take_profit']:
                        exit_price = current_position['take_profit']
                        exit_type = 'TP'
                else:
                    # Pour une position courte
                    if next_bar['High'] >= current_position['stop_loss']:
                        exit_price = current_position['stop_loss']
                        exit_type = 'SL'
                    elif next_bar['Low'] <= current_position['take_profit']:
                        exit_price = current_position['take_profit']
                        exit_type = 'TP'
                
                # Si une condition de sortie est atteinte
                if exit_price is not None:
                    # Calculer le P&L
                    if current_position['direction'] == 'long':
                        profit_loss = (exit_price - current_position['entry_price']) / pip_value
                    else:
                        profit_loss = (current_position['entry_price'] - exit_price) / pip_value
                    
                    profit_loss *= current_position['lot_size'] * 10  # Convertir en dollars (approximatif)
                    
                    # Mettre à jour le solde
                    balance += profit_loss
                    
                    # Mettre à jour les statistiques
                    max_balance = max(max_balance, balance)
                    min_balance = min(min_balance, balance)
                    
                    if profit_loss < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    
                    # Enregistrer le trade
                    trades.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': next_bar['Date'],
                        'direction': current_position['direction'],
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'profit_loss': profit_loss,
                        'exit_type': exit_type,
                        'lot_size': current_position['lot_size']
                    })
                    
                    # Fermer la position
                    current_position = None
                    
                    # Vérifier l'objectif quotidien
                    daily_profit = sum(t['profit_loss'] for t in trades if pd.to_datetime(t['exit_date']).strftime('%Y-%m-%d') == day_str)
                    if daily_profit >= initial_balance * (strategy['profit_target_daily'] / 100):
                        # Si l'objectif quotidien est atteint, ne plus trader ce jour
                        daily_trades[day_str] = strategy['max_trades_per_day']
        
        # Calculer les statistiques de performance
        total_trades = len(trades)
        if total_trades == 0:
            return {
                'total_profit': 0,
                'profit_factor': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'strategy': strategy,
                'trades': [],
                'balance': balance,
                'roi': 0,
                'total_trades': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'risk_reward_ratio': 0
            }
        
        winning_trades = [t for t in trades if t['profit_loss'] > 0]
        losing_trades = [t for t in trades if t['profit_loss'] <= 0]
        
        total_profit = sum(t['profit_loss'] for t in trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = sum(t['profit_loss'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['profit_loss'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(t['profit_loss'] for t in winning_trades) / sum(t['profit_loss'] for t in losing_trades)) if sum(t['profit_loss'] for t in losing_trades) != 0 else float('inf')
        
        # Calculer le drawdown maximum
        balances = [self.initial_balance]
        for trade in trades:
            balances.append(balances[-1] + trade['profit_loss'])
        
        cummax = np.maximum.accumulate(balances)
        drawdowns = (cummax - balances) / cummax * 100
        max_drawdown = max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calculer le ratio de Sharpe (simplifié)
        if len(trades) > 1:
            returns = [t['profit_loss'] / self.initial_balance for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        roi = (balance - self.initial_balance) / self.initial_balance * 100
        
        return {
            'total_profit': total_profit,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': total_trades,
            'strategy': strategy,
            'trades': trades,
            'balance': balance,
            'roi': roi,
            'risk_reward_ratio': avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
        }
    
    def run_optimization(self):
        """
        Lance l'optimisation en testant plusieurs stratégies aléatoires
        """
        print(f"Début de l'optimisation avec {self.n_strategies} stratégies aléatoires...")
        
        for i in tqdm(range(self.n_strategies)):
            strategy = self.generate_random_strategy()
            result = self.backtest_strategy(strategy)
            self.results.append(result)
            
            # Mettre à jour la meilleure stratégie
            performance_score = self.calculate_performance_score(result)
            if performance_score > self.best_performance:
                self.best_performance = performance_score
                self.best_strategy = result
                
        print("Optimisation terminée.")
        return self.best_strategy
    
    def calculate_performance_score(self, result):
        """
        Calcule un score de performance global pour une stratégie
        
        Args:
            result (dict): Résultats du backtest
            
        Returns:
            float: Score de performance
        """
        # Ne pas considérer les stratégies avec trop peu de trades
        if result['total_trades'] < 5:
            return -float('inf')
        
        # Facteurs importants (ajustez selon vos priorités)
        profit_weight = env.PROFIT_WEIGHT
        profit_factor_weight = env.PROFIT_FACTOR_WEIGHT
        win_rate_weight = env.WIN_RATE_WEIGHT
        drawdown_weight = env.DRAWDOWN_WEIGHT
        sharpe_weight = env.SHARPE_WEIGHT
        trades_weight = env.TRADES_WEIGHT
        risk_reward_weight = env.RISK_REWARD_WEIGHT
        
        # Pénalités pour les mauvais résultats
        if result['max_drawdown'] > env.MAX_DRAWDOWN:  # Drawdown trop élevé
            drawdown_penalty = env.DRAWDOWN_PENALTY
        else:
            drawdown_penalty = 0
            
        if result['total_trades'] < env.MIN_TRADES:  # Trop peu de trades
            trades_penalty = (env.MIN_TRADES - result['total_trades']) * env.TRADES_PENALTY
        else:
            trades_penalty = 0
            
        # Calcul du score
        score = (
            profit_weight * result['roi'] / 100 +
            profit_factor_weight * min(result['profit_factor'], 5) +
            win_rate_weight * result['win_rate'] +
            drawdown_weight * (1 - result['max_drawdown'] / 100) +
            sharpe_weight * min(result['sharpe_ratio'], 3) +
            trades_weight * min(result['total_trades'] / 100, 1) +
            risk_reward_weight * min(result.get('risk_reward_ratio', 1), 3) -
            drawdown_penalty -
            trades_penalty
        )
        
        return score
    
    def plot_equity_curve(self, result=None):
        """
        Trace la courbe d'équité de la meilleure stratégie
        
        Args:
            result (dict, optional): Résultat du backtest à tracer. Si None, utilise la meilleure stratégie.
        """
        if result is None:
            if self.best_strategy is None:
                print("Aucune stratégie optimisée trouvée. Exécutez run_optimization() d'abord.")
                return
            result = self.best_strategy
        
        # Créer la courbe d'équité
        trades = result['trades']
        if not trades:
            print("Aucun trade dans cette stratégie.")
            return
            
        equity = [self.initial_balance]
        dates = [trades[0]['entry_date']]
        
        for trade in trades:
            equity.append(equity[-1] + trade['profit_loss'])
            dates.append(trade['exit_date'])
        
        # Convertir les dates en format datetime si nécessaire
        if not isinstance(dates[0], datetime):
            dates = [pd.to_datetime(d) for d in dates]
        
        # Tracer la courbe
        plt.figure(figsize=(12, 6))
        plt.plot(dates, equity, label='Équité')
        
        # Ajouter une ligne pour le solde initial
        plt.axhline(y=self.initial_balance, color='r', linestyle='-', alpha=0.3, label='Solde initial')
        
        # Ajouter les informations de la stratégie
        strategy = result['strategy']
        plt.title(f"Courbe d'équité - Profit total: {result['total_profit']:.2f} USD, Win Rate: {result['win_rate']*100:.1f}%")
        plt.xlabel('Date')
        plt.ylabel('Solde ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Sauvegarder le graphique
        plt.tight_layout()
        plt.savefig(env.EQUITY_CURVE_FILE)
        plt.close()
        
        print(f"Courbe d'équité enregistrée sous '{env.EQUITY_CURVE_FILE}'")
    
    def generate_mq5_file(self, result=None):
        """
        Génère un fichier MQ5 avec les paramètres optimisés
        
        Args:
            result (dict, optional): Résultat du backtest à utiliser. Si None, utilise la meilleure stratégie.
        """
        if result is None:
            if self.best_strategy is None:
                print("Aucune stratégie optimisée trouvée. Exécutez run_optimization() d'abord.")
                return
            result = self.best_strategy
            
        strategy = result['strategy']
        
        # Préparer les paramètres pour le template
        params = {
            'lot_size': strategy['lot_size'],
            'stop_loss': int(strategy['stop_loss']),
            'take_profit': int(strategy['take_profit']),
            'trailing_stop': 'true' if strategy['trailing_stop'] else 'false',
            'trailing_distance': int(strategy['trailing_distance']) if strategy['trailing_distance'] is not None else 0,
            'max_trades_per_day': strategy['max_trades_per_day'],
            'entry_hour_min': strategy['entry_hour_min'],
            'entry_hour_max': strategy['entry_hour_max'],
            'weekend_trading': 'true' if strategy['weekend_trading'] else 'false',
            'max_drawdown_exit': strategy['max_drawdown_exit'],
            'consecutive_losses_exit': strategy['consecutive_losses_exit'],
            'profit_target_daily': strategy['profit_target_daily'],
            'direction': strategy['direction'],
            'direction_condition': self._get_direction_condition(strategy['direction'])
        }
        
        # Générer le fichier MQ5
        mq5_content = MQ5_TEMPLATE.format(**params)
        
        # Sauvegarder le fichier
        output_file = os.path.join(env.OUTPUT_DIR, "OptimizedStrategy.mq5")
        with open(output_file, 'w') as f:
            f.write(mq5_content)
            
        print(f"Fichier MQ5 généré avec succès : {output_file}")
        
    def _get_direction_condition(self, direction):
        """Retourne la condition de direction pour le template MQ5"""
        if direction == 'long':
            return "IsLongSignal()"  # À implémenter selon votre stratégie
        elif direction == 'short':
            return "IsShortSignal()"  # À implémenter selon votre stratégie
        else:  # both
            return "IsLongSignal() || IsShortSignal()"  # À implémenter selon votre stratégie

    def summarize_results(self):
        """
        Résume les résultats de l'optimisation et génère le fichier MQ5
        """
        if self.best_strategy is None:
            print("Aucune stratégie optimisée trouvée. Exécutez run_optimization() d'abord.")
            return
            
        strategy = self.best_strategy['strategy']
        
        print("\n" + "="*80)
        print(f"MEILLEURE STRATÉGIE - Score de performance: {self.best_performance:.2f}")
        print("="*80)
        
        print("\nPARAMÈTRES DE LA STRATÉGIE:")
        print(f"Direction: {strategy['direction']}")
        print(f"Time Frame: {strategy['time_frame']} minutes")
        print(f"Type d'entrée: {strategy['entry_type']}")
        print(f"\nTake Profit: {strategy['take_profit']:.1f} pips")
        print(f"Stop Loss: {strategy['stop_loss']:.1f} pips")
        print(f"Trailing Stop: {'Activé' if strategy['trailing_stop'] else 'Désactivé'}")
        if strategy['trailing_stop'] and strategy['trailing_distance'] is not None:
            print(f"Distance du Trailing: {strategy['trailing_distance']:.1f} pips")
        
        print(f"\nTaille de lot: {strategy['lot_size']:.2f}")
        print(f"Méthode de dimensionnement: {'Basé sur le risque ' + str(strategy['risk_percent']) + '%' if strategy['position_sizing'] == 'risk_based' else 'Fixe'}")
        
        print(f"\nHeures de trading: {strategy['entry_hour_min']}h - {strategy['entry_hour_max']}h")
        print(f"Max trades par jour: {strategy['max_trades_per_day']}")
        print(f"Trading le weekend: {'Oui' if strategy['weekend_trading'] else 'Non'}")
        
        print(f"\nSortie sur drawdown max: {strategy['max_drawdown_exit']:.1f}%")
        print(f"Sortie après pertes consécutives: {strategy['consecutive_losses_exit']}")
        print(f"Objectif de profit quotidien: {strategy['profit_target_daily']:.1f}%")
        
        print("\nRÉSULTATS DE PERFORMANCE:")
        print(f"Profit total: {self.best_strategy['total_profit']:.2f} USD")
        print(f"ROI: {self.best_strategy['roi']:.2f}%")
        print(f"Facteur de profit: {self.best_strategy['profit_factor']:.2f}")
        print(f"Taux de réussite: {self.best_strategy['win_rate']*100:.1f}%")
        print(f"Drawdown maximum: {self.best_strategy['max_drawdown']:.2f}%")
        print(f"Ratio Sharpe: {self.best_strategy['sharpe_ratio']:.2f}")
        print(f"Ratio risque/récompense: {self.best_strategy.get('risk_reward_ratio', 0):.2f}")
        print(f"Gain moyen: {self.best_strategy['avg_win']:.2f} USD")
        print(f"Perte moyenne: {self.best_strategy['avg_loss']:.2f} USD")
        print(f"Nombre total de trades: {self.best_strategy['total_trades']}")
        print(f"Solde final: {self.best_strategy['balance']:.2f} USD")
        
        # Générer le graphique
        self.plot_equity_curve()
        
        # Générer le fichier MQ5
        self.generate_mq5_file()
        
        # Sauvegarder les résultats détaillés
        self.save_detailed_results()
    
    def save_detailed_results(self):
        """Sauvegarde les résultats détaillés dans un fichier CSV"""
        if self.best_strategy is None:
            return
            
        # Sauvegarder les paramètres de la stratégie
        strategy_params = pd.DataFrame([self.best_strategy['strategy']])
        strategy_params.to_csv(env.BEST_STRATEGY_PARAMS_FILE, index=False)
        print(f"Paramètres de la stratégie sauvegardés dans '{env.BEST_STRATEGY_PARAMS_FILE}'")
        
        # Sauvegarder les trades
        if self.best_strategy['trades']:
            trades_df = pd.DataFrame(self.best_strategy['trades'])
            trades_df.to_csv(env.BEST_STRATEGY_TRADES_FILE, index=False)
            print(f"Historique des trades sauvegardé dans '{env.BEST_STRATEGY_TRADES_FILE}'")
        
        # Sauvegarder un résumé de toutes les stratégies testées
        if self.results:
            summary = []
            for r in self.results:
                s = r['strategy']
                summary.append({
                    'profit': r['total_profit'],
                    'roi': r['roi'],
                    'win_rate': r['win_rate'],
                    'profit_factor': r['profit_factor'],
                    'max_drawdown': r['max_drawdown'],
                    'sharpe_ratio': r['sharpe_ratio'],
                    'total_trades': r['total_trades'],
                    'take_profit': s['take_profit'],
                    'stop_loss': s['stop_loss'],
                    'direction': s['direction'],
                    'lot_size': s['lot_size'],
                    'time_frame': s['time_frame']
                })
                
            summary_df = pd.DataFrame(summary)
            summary_df = summary_df.sort_values('roi', ascending=False)
            summary_df.to_csv(env.ALL_STRATEGIES_SUMMARY_FILE, index=False)
            print(f"Résumé de toutes les stratégies sauvegardé dans '{env.ALL_STRATEGIES_SUMMARY_FILE}'")
            
            # Afficher un résumé des 5 meilleures stratégies
            print("\nTOP 5 DES MEILLEURES STRATÉGIES:")
            for i, row in summary_df.head(5).iterrows():
                print(f"{i+1}. ROI: {row['roi']:.2f}%, Win Rate: {row['win_rate']*100:.1f}%, TP: {row['take_profit']:.1f}, SL: {row['stop_loss']:.1f}, Direction: {row['direction']}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Utiliser un fichier de données historiques si disponible, sinon utiliser des données synthétiques
    data_file = env.DATA_FILE  # Remplacez par votre fichier de données
    
    print("Initialisation de l'optimiseur de stratégie...")
    optimizer = StrategyOptimizer(
        data_file=data_file,
        initial_balance=env.INITIAL_BALANCE,
        n_strategies=env.N_STRATEGIES,  # Nombre de stratégies aléatoires à tester
        currency_pair=env.CURRENCY_PAIR
    )
    
    # Lancer l'optimisation
    best_strategy = optimizer.run_optimization()
    
    # Afficher les résultats
    optimizer.summarize_results()
