# Změny v `models.py`

Do souboru `models.py` jsem doplnil chybějící implementace pro všechny hodnocené části zadání.

- `PerceptronModel`: inicializace vah, výpočet skóre, predikce a trénování do konvergence
- `RegressionModel`: neuronová síť pro aproximaci funkce, výpočet loss a trénovací smyčka
- `DigitClassificationModel`: klasifikátor číslic, výpočet logits, loss a trénování
- `LanguageIDModel`: rekurentní model pro rozpoznání jazyka slov, výpočet loss a trénování
- `Convolve`: vlastní implementace 2D konvoluce
- `DigitConvolutionalModel`: konvoluční klasifikátor číslic, výpočet loss a trénování
- `Attention`: doplnění masked scaled dot-product attention

