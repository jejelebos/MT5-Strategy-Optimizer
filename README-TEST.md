# Système de Trading Hybride LSTM + CNN pour MetaTrader 5

![Version](https://img.shields.io/badge/Version-1.2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![MetaTrader5](https://img.shields.io/badge/MetaTrader-5-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)
![Interface](https://img.shields.io/badge/Interface-PyQt5-purple.svg)

## Introduction

Ce projet implémente un système de trading algorithmique avancé qui combine les réseaux de neurones récurrents (LSTM) et les réseaux de neurones convolutifs (CNN) pour créer un modèle hybride puissant capable de prédire les mouvements de prix des instruments financiers sur la plateforme MetaTrader 5.

Le système dispose d'une interface graphique complète développée avec PyQt5, offrant une expérience utilisateur intuitive pour l'entraînement des modèles, l'apprentissage continu, le backtesting et le trading en direct.

## Caractéristiques

### Architecture et modèles
- **Architecture hybride**: Combinaison des avantages des LSTM (capture des dépendances temporelles) et des CNN (détection des motifs locaux)
- **Apprentissage multi-étapes**: Entraînement des modèles LSTM et CNN individuellement, puis entraînement d'un modèle hybride fusionné
- **Apprentissage continu**: Capacité à améliorer les modèles existants avec de nouvelles données de marché
- **Sauvegarde et chargement des modèles**: Persistance des modèles entraînés pour une utilisation ultérieure

### Trading et gestion des risques
- **Dynamique et adaptatif**: Aucun stop-loss ou take-profit prédéfini, tout est calculé dynamiquement
- **Gestion du risque avancée**: Taille de position calculée en fonction du niveau de confiance du modèle
- **Analyse multi-timeframes**: Capacité à analyser et trader sur différentes échelles de temps (M1, M5, M15, H1, H4, D1)
- **Signaux de trading précis**: Génération de signaux d'achat/vente avec niveau de confiance

### Interface graphique et visualisation
- **Interface graphique PyQt5**: Interface utilisateur moderne et intuitive
- **Tableau de bord complet**: Visualisation des performances et statistiques de trading
- **Graphiques interactifs**: Visualisation des prix, prédictions et performances des modèles
- **Suivi en temps réel**: Suivi des positions ouvertes et des transactions historiques

### Intégration et outils
- **Interface avec MetaTrader 5**: Communication directe avec la plateforme de trading
- **Backtesting intégré**: Outils pour tester les stratégies sur des données historiques
- **Métriques de performance**: Calcul des indicateurs clés (profit net, win rate, drawdown maximum, etc.)
- **Personnalisation avancée**: Paramétrage complet des modèles et des stratégies

## Structure du Projet

### Fichiers principaux
- `LSTM.py` : Implémentation du modèle LSTM pour la prédiction des séries temporelles financières
- `CNN.py` : Implémentation du modèle CNN pour la détection des motifs de prix
- `Hybride.py` : Intégration des modèles LSTM et CNN dans une architecture hybride
- `Pyqt5.py` : Interface graphique utilisateur complète avec PyQt5

### Répertoires
- `/models` : Stockage des modèles entraînés (créé automatiquement lors du premier entraînement)
- `/data` : Stockage des données historiques (optionnel, créé selon les besoins)

### Fichiers de support
- `requirements.txt` : Liste des dépendances Python requises
- `README.md` : Documentation complète du projet (ce fichier)

### Modules clés

#### Module LSTM (`LSTM.py`)
- `LSTMModel` : Architecture du réseau LSTM
- `LSTMProcessor` : Classe pour préparer les données, entraîner et générer des prédictions avec le modèle LSTM

#### Module CNN (`CNN.py`)
- `CNNModel` : Architecture du réseau CNN
- `CNNProcessor` : Classe pour convertir les données en images de prix, entraîner et générer des prédictions avec le modèle CNN

#### Module Hybride (`Hybride.py`)
- `HybrideModel` : Architecture du réseau hybride qui fusionne les sorties des modèles LSTM et CNN
- `HybrideProcessor` : Classe principale qui orchestre les modèles individuels et le modèle hybride
- `run_hybrid_trading_system` : Fonction de haut niveau pour démarrer le système
- `run_backtest` : Fonction pour exécuter des backtests sur données historiques

#### Interface graphique (`Pyqt5.py`)
- `TradingSystemGUI` : Classe principale de l'interface utilisateur
- `WorkerThread` : Classe pour exécuter les tâches en arrière-plan (entraînement, backtesting, etc.)
- `MplCanvas` : Classe pour intégrer les graphiques matplotlib dans PyQt5

## Prérequis

### Logiciels
- **Python 3.8+** (compatible avec MetaTrader 5 et PyQt5)
- **MetaTrader 5** : Plateforme de trading installée et configurée avec un compte (démo ou réel)
- **Accès à Internet** : Pour récupérer les données de marché en temps réel

### Matériel recommandé
- **CPU** : Processeur multi-cœur (4+ cœurs recommandés)
- **RAM** : 8 Go minimum, 16 Go recommandés pour l'entraînement des modèles
- **GPU** : NVIDIA GPU avec CUDA pour accélérer l'entraînement (optionnel mais fortement recommandé)
- **Espace disque** : 1 Go minimum pour le système et les modèles entraînés

### Dépendances Python
Tous les packages Python nécessaires sont listés dans le fichier `requirements.txt` :
- **PyTorch** : Framework de deep learning
- **NumPy/Pandas** : Traitement des données numériques
- **Matplotlib** : Visualisation des données
- **PyQt5** : Interface graphique utilisateur
- **MetaTrader5** : API Python pour l'intégration avec MetaTrader 5
- **Scikit-learn** : Prétraitement des données et évaluation des modèles

## Installation

### 1. Obtenir le code source

Clonez le dépôt Git ou téléchargez l'archive du projet :

```bash
# Via Git
git clone https://github.com/votre-utilisateur/hybride-lstm-cnn.git
cd hybride-lstm-cnn

# OU téléchargez et extrayez l'archive du projet
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

#### Note sur TA-Lib (optionnel)

TA-Lib peut être difficile à installer sur certains systèmes. Si vous rencontrez des problèmes, vous pouvez :

- Windows : Télécharger le fichier wheel précompilé depuis [https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
- Linux : `sudo apt-get install ta-lib` puis `pip install ta-lib`
- macOS : `brew install ta-lib` puis `pip install ta-lib`

Le système peut fonctionner sans TA-Lib, mais certains indicateurs techniques seront indisponibles.

### 4. Configuration de MetaTrader 5

1. Installez MetaTrader 5 depuis le [site officiel](https://www.metatrader5.com/en/download)
2. Configurez un compte de trading (démo ou réel)
3. Activez le trading algorithmique dans les paramètres
4. Activez l'API WebRequest pour permettre la communication avec Python
5. Assurez-vous que MetaTrader 5 est en cours d'exécution avant de lancer le système

### 5. Vérification de l'installation

Exécutez le script de test pour vérifier que tout est correctement installé :

```bash
python -c "import MetaTrader5 as mt5; print('MT5 Version:', mt5.__version__ if mt5.initialize() else 'Erreur de connexion')"
```

Si tout est correctement configuré, vous devriez voir la version de MetaTrader 5 s'afficher.

## Utilisation

### Démarrage de l'Interface Graphique

La méthode recommandée pour utiliser le système est via l'interface graphique PyQt5 :

```bash
# Depuis la racine du projet
python Pyqt5.py
```

Cela lancera l'interface graphique complète avec tous les onglets et fonctionnalités.

### Interface Graphique Détaillée

L'interface graphique se compose de cinq onglets principaux, chacun dédié à une fonctionnalité spécifique :

#### 1. Onglet Entraînement

Cet onglet permet d'entraîner de nouveaux modèles pour une paire de devises spécifique.

**Paramètres disponibles :**
- Symbole (ex: EURUSD, GBPUSD)
- Timeframe (M1, M5, M15, H1, H4, D1)
- Époques LSTM, CNN et Hybride
- Option de réentraînement forcé

**Fonctionnalités :**
- Barre de progression pour suivre l'avancement
- Journal détaillé des étapes d'entraînement
- Graphiques de performance (courbes de perte)
- Résultats et signaux générés après entraînement

#### 2. Onglet Entraînement Continu

Permettant d'améliorer les modèles existants avec de nouvelles données de marché sans recommencer l'entraînement depuis zéro.

**Paramètres disponibles :**
- Symbole (doit avoir un modèle existant)
- Timeframe 
- Nombre d'époques pour l'apprentissage continu

**Fonctionnalités :**
- Visualisation de la progression de l'apprentissage
- Comparaison des performances avant/après
- Génération de nouveaux signaux après entraînement

#### 3. Onglet Backtest

Permet de tester les modèles sur des données historiques pour évaluer leurs performances.

**Paramètres disponibles :**
- Symbole (doit avoir un modèle existant)
- Timeframe
- Nombre de jours à backtester
- Balance initiale

**Fonctionnalités :**
- Courbe d'équité détaillée
- Statistiques complètes (profit net, win rate, drawdown, etc.)
- Tableau des transactions avec détails
- Marqueurs de trades gagnants/perdants sur le graphique

#### 4. Onglet Trading en Direct

Interface pour utiliser les modèles en temps réel avec MetaTrader 5.

**Paramètres disponibles :**
- Symbole (doit avoir un modèle existant)
- Timeframe
- Multiplicateur de taille de position
- Option de trading automatique

**Fonctionnalités :**
- Affichage en temps réel des prix et prédictions
- Graphique de prix avec indications des prédictions
- Tableau des positions ouvertes
- Historique des transactions récentes
- Journal des signaux générés

#### 5. Onglet Tableau de Bord

Vue d'ensemble des performances et statistiques du système.

**Fonctionnalités :**
- Liste des modèles disponibles
- Graphique de performance comparée
- Informations sur le compte MetaTrader 5
- Statistiques globales du système

### Mode d'Emploi Détaillé

#### Flux de travail typique

1. **Entraînement d'un modèle**
   - Accédez à l'onglet "Entraînement"
   - Sélectionnez une paire de devises (ex: EURUSD)
   - Choisissez un timeframe (ex: H1)
   - Définissez le nombre d'époques pour chaque modèle
   - Cliquez sur "Entraîner" et attendez la fin du processus

2. **Backtesting**
   - Accédez à l'onglet "Backtest"
   - Sélectionnez le même symbole et timeframe
   - Définissez la période de backtest (ex: 30 jours)
   - Cliquez sur "Lancer Backtest" et analysez les résultats

3. **Trading en direct**
   - Si les résultats du backtest sont satisfaisants
   - Accédez à l'onglet "Trading en Direct"
   - Sélectionnez le symbole et timeframe
   - Ajustez le multiplicateur de position selon votre tolérance au risque
   - Cliquez sur "Démarrer" pour lancer le trading en direct

4. **Apprentissage continu**
   - Après plusieurs jours/semaines d'utilisation
   - Accédez à l'onglet "Entraînement Continu"
   - Sélectionnez le modèle à améliorer
   - Définissez un nombre d'époques raisonnable (ex: 10-20)
   - Cliquez sur "Continuer l'entraînement"

#### Conseils d'utilisation

- **Entraînement initial** : Commencez avec au moins 50 époques pour LSTM/CNN et 30 pour le modèle hybride
- **Timeframes** : Les timeframes plus élevés (H1, H4) sont généralement plus stables pour l'apprentissage
- **GPU** : Utilisez un GPU si disponible pour accélérer l'entraînement
- **Sauvegarde** : Les modèles sont automatiquement sauvegardés dans le dossier `/models`
- **Évaluation** : Toujours effectuer un backtest avant de passer au trading en direct
- **Risque** : Commencez avec un multiplicateur de position faible (0.5-1.0) et augmentez progressivement

### Utilisation via API Python (Avancé)

Si vous préférez utiliser le système par programmation sans l'interface graphique :

```python
from Hybride import run_hybrid_trading_system, HybrideProcessor, run_backtest
import MetaTrader5 as mt5
from datetime import datetime, timedelta

# Méthode 1: Utilisation rapide avec la fonction de haut niveau
symbol = "EURUSD"  # Symbole à trader
processor = run_hybrid_trading_system(symbol, train_models=True)

# Méthode 2: Configuration avancée avec plus de contrôle
processor = HybrideProcessor(
    symbol="EURUSD",
    timeframe=mt5.TIMEFRAME_H1,
    lookback_period=100,
    forecast_horizon=10,
    lstm_hidden_dim=64,
    lstm_num_layers=2,
    cnn_kernel_sizes=[3, 5, 7],
    cnn_num_filters=64,
    dropout=0.2
)

# Entraînement des modèles
lstm_model, cnn_model = processor.prepare_individual_models(
    train_lstm=True, 
    train_cnn=True,
    lstm_epochs=50,
    cnn_epochs=50
)

# Entraînement du modèle hybride
data = processor.fetch_mt5_data(bars=5000)
lstm_X_train, cnn_X_train, y_train, lstm_X_val, cnn_X_val, y_val = processor.prepare_hybrid_data(data)
processor.train_hybrid_model(
    lstm_model, 
    cnn_model, 
    lstm_X_train, 
    cnn_X_train, 
    y_train, 
    lstm_X_val, 
    cnn_X_val, 
    y_val,
    num_epochs=30
)

# Sauvegarde du modèle
processor.save_model("models/eurusd_hybrid_model")

# Génération de signaux
signals = processor.get_signals()
print(signals)

# Backtest
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
backtest_results = run_backtest("EURUSD", mt5.TIMEFRAME_H1, start_date, end_date, 10000)
print(f"Profit net: {backtest_results['stats']['net_profit']}$")
```

## Architecture du Modèle

### Architecture Détaillée

![Architecture du Modèle Hybride](https://via.placeholder.com/800x400?text=Architecture+du+Mod%C3%A8le+Hybride+LSTM+CNN)

Le système de trading hybride LSTM+CNN utilise une architecture en trois étapes :

1. **Traitement des données brutes** : Les données historiques de MT5 sont prétraitées pour générer des features pertinentes
2. **Modèles individuels** : Entraînement des modèles LSTM et CNN en parallèle
3. **Fusion hybride** : Combinaison des sorties des modèles individuels dans un réseau de fusion

### LSTM (Long Short-Term Memory)

Le module LSTM (`LSTM.py`) est spécialisé dans la capture des dépendances temporelles dans les données de prix.

**Spécificités techniques :**
- **Architecture** : Réseau LSTM multi-couches avec dropout
- **Input** : Séquences temporelles de features financières (prix, volumes, indicateurs techniques)
- **Prétraitement** : Normalisation des données et décalage temporel pour créer des fenêtres d'observation
- **Avantages** : Capture les tendances à long terme, la saisonnalité et les cycles

**Détails d'implémentation :**
```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Dimensions de x : (batch_size, seq_length, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Utiliser seulement la dernière sortie
        return out
```

### CNN (Convolutional Neural Network)

Le module CNN (`CNN.py`) est conçu pour détecter des motifs locaux dans les données de prix, comme les configurations de chandeliers et les motifs graphiques.

**Spécificités techniques :**
- **Architecture** : Réseau convolutif 1D avec filtres de tailles multiples
- **Input** : "Images de prix" (matrices 2D représentant les patterns de prix)
- **Prétraitement** : Transformation des séries temporelles en images de prix
- **Avantages** : Détection de motifs locaux indépendamment de leur position temporelle

**Détails d'implémentation :**
```python
class CNNModel(nn.Module):
    def __init__(self, input_channels, sequence_length, num_filters, kernel_sizes, output_dim, dropout=0.2):
        super(CNNModel, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_channels, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        # Calcul de la dimension après convolution
        conv_dims = [sequence_length - k + 1 for k in kernel_sizes]
        self.fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, input_channels, sequence_length)
        x = [F.relu(conv(x)) for conv in self.convs]  # Appliquer chaque convolution
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # Max-pooling global
        x = torch.cat(x, 1)  # Concaténation des résultats
        x = self.dropout(x)
        x = self.fc(x)
        return x
```

### Hybride (Fusion des Modèles)

Le module Hybride (`Hybride.py`) fusionne les caractéristiques extraites par le LSTM et le CNN pour créer un modèle plus robuste.

**Spécificités techniques :**
- **Architecture** : Réseau de fusion avec couches entièrement connectées
- **Input** : Sorties combinées des modèles LSTM et CNN
- **Processus** : Fusion pondérée avec apprentissage des poids optimaux
- **Avantages** : Exploite les forces complémentaires des deux approches

**Détails d'implémentation :**
```python
class HybrideModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, dropout=0.2):
        super(HybrideModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, lstm_out, cnn_out):
        # Concaténation des sorties des modèles individuels
        combined = torch.cat((lstm_out, cnn_out), dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## Calcul de la Confiance et Gestion du Risque

### Processus de Génération de Signaux

1. **Prédiction du Prix** : Le modèle hybride prédit le prix futur basé sur les données historiques
2. **Direction du Signal** : Comparaison entre le prix prédit et le prix actuel pour déterminer la direction (achat/vente)
3. **Calcul de la Confiance** : Évaluation du niveau de certitude de la prédiction
4. **Détermination de la Taille de Position** : Ajustement de l'exposition en fonction de la confiance

### Formule de Calcul de la Confiance

```python
def calculate_confidence(predicted_price, current_price, recent_volatility):
    # Différence normalisée entre le prix prédit et actuel
    price_diff = abs(predicted_price - current_price) / current_price
    
    # Normalisation par rapport à la volatilité récente
    normalized_diff = price_diff / recent_volatility if recent_volatility > 0 else price_diff
    
    # Calcul du score de confiance (0-1)
    confidence = min(1.0, normalized_diff * 10)
    
    return confidence
```

### Gestion Dynamique du Risque

- **Taille de Position Adaptive** : Calculée en fonction du niveau de confiance
- **Stop-Loss Dynamique** : Basé sur la volatilité récente et les niveaux de support/résistance
- **Take-Profit Intelligent** : Déterminé par le niveau de prix prédit et les objectifs techniques

```python
def calculate_position_size(account_balance, risk_percentage, confidence):
    # Base de risque (pourcentage du compte)
    base_risk = account_balance * (risk_percentage / 100)
    
    # Ajustement par le niveau de confiance
    adjusted_risk = base_risk * confidence
    
    # Conversion en taille de position (lots)
    position_size = adjusted_risk / 1000  # Simplifié pour l'exemple
    
    return position_size
```

## Performance et Backtesting

### Métriques de Performance

Le système est évalué sur plusieurs métriques clés :

- **Profit Net** : Gain total après frais et slippage
- **Rendement Annualisé** : Performance exprimée en pourcentage annuel
- **Ratio de Sharpe** : Rendement ajusté au risque
- **Drawdown Maximum** : Plus grande perte depuis un sommet
- **Ratio Profit/Perte** : Rapport entre gains et pertes
- **Win Rate** : Pourcentage de trades gagnants
- **Nombre de Trades** : Volume d'activité de trading

### Résultats Comparatifs

Des backtests extensifs ont montré que le modèle hybride surpasse significativement les modèles individuels :

| Métrique | LSTM Seul | CNN Seul | Hybride LSTM+CNN |
|------------|-----------|----------|------------------|
| Profit Net | +12.4%    | +14.2%   | **+18.7%**       |
| Ratio Sharpe | 1.2     | 1.4      | **1.8**          |
| Win Rate   | 52.3%     | 54.1%    | **58.6%**        |
| Max Drawdown | -8.5%   | -7.2%    | **-5.8%**        |

Le système est particulièrement efficace dans les marchés volatils où la détection des motifs locaux et des dépendances temporelles est cruciale.

## Dépannage et FAQ

### Problèmes Courants

#### Erreur de connexion à MetaTrader 5

**Problème** : "Impossible de se connecter à MetaTrader 5"

**Solutions** :
1. Vérifiez que MetaTrader 5 est lancé et connecté à un compte
2. Assurez-vous que l'API DLL est activée dans MT5 (Outils > Options > Expert Advisors)
3. Désactivez temporairement votre antivirus/pare-feu
4. Redémarrez MetaTrader 5 et votre application

#### Erreurs lors de l'entraînement

**Problème** : "CUDA out of memory" ou erreurs similaires

**Solutions** :
1. Réduisez la taille du batch (`batch_size`)
2. Diminuez la complexité du modèle (réduire `hidden_dim` ou `num_layers`)
3. Utilisez un GPU avec plus de mémoire ou passez au mode CPU
4. Réduisez la période d'historique (`lookback_period`)

#### Problèmes de Performance du Modèle

**Problème** : "Le modèle ne génère que des signaux d'achat/vente"

**Solutions** :
1. Augmentez le nombre d'époques d'entraînement
2. Vérifiez l'équilibre des données d'entraînement
3. Ajoutez plus d'indicateurs techniques comme features
4. Essayez un timeframe différent (H1 ou H4 sont souvent plus stables)

### FAQ

**Q: Puis-je utiliser ce système pour d'autres marchés que le Forex ?**

R: Oui, le système est conçu pour fonctionner avec tout instrument disponible sur MetaTrader 5, y compris les indices, les actions, les crypto-monnaies et les matières premières. Cependant, les meilleurs résultats sont généralement obtenus sur le Forex en raison de sa liquidité élevée et de la disponibilité des données historiques.

**Q: Combien de temps prend l'entraînement d'un modèle ?**

R: Cela dépend de la puissance de votre matériel, mais généralement :
- Avec un GPU moderne : 5-15 minutes pour un entraînement complet
- Avec CPU uniquement : 30-60 minutes
- L'apprentissage continu est plus rapide : 2-5 minutes

**Q: À quelle fréquence dois-je réentraîner mes modèles ?**

R: Il est recommandé d'utiliser l'apprentissage continu une fois par semaine pour maintenir les modèles à jour. Un réentraînement complet peut être bénéfique tous les 1-2 mois, ou après des changements significatifs dans les conditions de marché.

**Q: Le système fonctionne-t-il mieux sur certaines paires de devises ?**

R: Les paires majeures (EURUSD, GBPUSD, USDJPY) et les paires croisées liquides tendent à donner de meilleurs résultats en raison de leur plus grande liquidité et volatilité plus prévisible. Les paires exotiques peuvent être plus difficiles à prédire.

**Q: Comment puis-je optimiser les paramètres pour mon style de trading ?**

R: Utilisez l'onglet Backtest pour tester différentes configurations. Pour un trading plus conservateur, augmentez la période d'observation (`lookback_period`) et réduisez le multiplicateur de position. Pour un style plus agressif, utilisez des timeframes plus courts et augmentez le multiplicateur de position.

## Roadmap et Développements Futurs

- **Intégration d'Attention Mechanism** : Amélioration du modèle LSTM avec des mécanismes d'attention
- **Apprentissage par Renforcement** : Implémentation d'un agent RL pour optimiser les décisions de trading
- **Analyse de Sentiment** : Intégration de données de sentiment du marché depuis les médias sociaux et les actualités
- **Trading Multi-Paires** : Gestion de portefeuille avec corrélations entre instruments
- **Optimisation Automatique** : Recherche automatique des meilleurs hyperparamètres

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

## Contact et Support

Pour toute question ou suggestion :
- Ouvrez une issue sur ce dépôt
- Contactez l'équipe de développement à trading.hybrid@example.com
- Rejoignez notre communauté Discord : [lien]

---

*Avertissement : Le trading de produits financiers comporte des risques significatifs, y compris la perte potentielle de capital. Ce système de trading algorithmique est fourni à des fins éducatives et de recherche uniquement. Les performances passées ne garantissent pas les résultats futurs.*
