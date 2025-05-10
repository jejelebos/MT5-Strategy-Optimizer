# Strategy Optimizer

Un outil puissant pour optimiser les stratégies de trading algorithmique en utilisant des données historiques de MetaTrader 5 ou des données synthétiques.

## 🚀 Fonctionnalités

- Optimisation de stratégies de trading avec paramètres personnalisables
- Support pour MetaTrader 5 (données historiques en temps réel)
- Génération de données synthétiques pour les tests
- Backtesting complet avec métriques de performance
- Visualisation des résultats avec graphiques
- Export des résultats en CSV pour analyse approfondie

## 📋 Prérequis

- Python 3.8 ou supérieur
- MetaTrader 5 (optionnel, pour les données en temps réel)
- Les dépendances listées dans `requirements.txt`

## 🛠️ Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/jejelebos/MT5-Strategy-Optimizer.git
cd MT5-Strategy-Optimizer
```

2. Créez un environnement virtuel et activez-le :
```bash
python -m venv venv
# Sur Windows
venv\Scripts\activate
# Sur Linux/Mac
source venv/bin/activate
```

## ⚙️ Configuration

Le fichier `env.py` contient tous les paramètres configurables :

- Paramètres de risque et de trading
- Configuration de l'optimisation
- Paramètres de MetaTrader 5
- Chemins des fichiers de sortie

## 🎯 Utilisation

1. Configurez vos paramètres dans `env.py`
2. Lancez l'optimisation :
```bash
python strategy_optimizer.py
```

Les résultats seront sauvegardés dans le dossier `output/` :
- `equity_curve.png` : Graphique de la courbe d'équité
- `best_strategy_params.csv` : Paramètres de la meilleure stratégie
- `best_strategy_trades.csv` : Historique des trades
- `all_strategies_summary.csv` : Résumé de toutes les stratégies testées

## 📊 Métriques de Performance

L'optimiseur évalue les stratégies selon plusieurs critères :
- Profit total et ROI
- Facteur de profit
- Taux de réussite
- Drawdown maximum
- Ratio de Sharpe
- Ratio risque/récompense

## 🔧 Personnalisation

Vous pouvez ajuster les poids des différentes métriques dans `env.py` :
- `PROFIT_WEIGHT`
- `PROFIT_FACTOR_WEIGHT`
- `WIN_RATE_WEIGHT`
- `DRAWDOWN_WEIGHT`
- `SHARPE_WEIGHT`
- `TRADES_WEIGHT`
- `RISK_REWARD_WEIGHT`

## 📝 Structure du Projet

```
strategy-optimizer/
├── strategy_optimizer.py  # Script principal
├── env.py                # Configuration
├── README.md            # Documentation
└── output/              # Dossier des résultats
    ├── equity_curve.png
    ├── best_strategy_params.csv
    ├── best_strategy_trades.csv
    └── all_strategies_summary.csv
```

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## ⚠️ Avertissement

Ce logiciel est fourni à des fins éducatives uniquement. Le trading comporte des risques. Utilisez ce logiciel à vos propres risques. 
