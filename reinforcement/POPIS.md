- V souboru `valueIterationAgents.py` jsem doplnil agenta pro **value iteration**:
  - batch výpočet hodnot stavů,
  - výpočet Q-hodnot ze spočítaných hodnot,
  - odvození nejlepší akce (policy) pro daný stav.

- V souboru `qlearningAgents.py` jsem doplnil **Q-learning**:
  - ukládání a čtení Q-hodnot,
  - výpočet nejlepší hodnoty a nejlepší akce,
  - aktualizaci Q-hodnot podle přechodu a odměny,
  - epsilon-greedy výběr akce,
  - approximate Q-learning pomocí vah a feature extractorů.

- V souboru `analysis.py` jsem doplnil odpovědi na otázku 2, tedy konkrétní nastavení parametrů:
  - discount,
  - noise,
  - living reward.
