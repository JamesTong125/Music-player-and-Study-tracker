import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("session_results.csv")

df['smooth'] = df['focus_score'].rolling(window=30).mean() * 100 

plt.fill_between(range(len(df)), df['smooth'], color="limegreen", alpha=0.3)
plt.plot(df['smooth'], color="green", linewidth=2, label="Focus Level %")
plt.title("Your Neural Flow State Session"); plt.ylabel("Focus %"); plt.legend(); plt.show()