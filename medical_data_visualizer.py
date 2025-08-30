import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('medical_examination.csv')

df['overweight'] = (df['weight'] / (df['height']/100)**2 > 25).astype(int)

df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1

def draw_cat_plot():
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    graph = sns.catplot(data=df_cat, kind='count', x='variable', hue='value', col='cardio')
    graph.set_ylabels('total', fontsize=10)

    fig = plt.show()

    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    df_heat = df.loc[(df.ap_lo <= df.ap_hi) & (df.height >= df.height.quantile(0.025)) & (df.height < df.height.quantile(0.975)) & (df.weight >= df.weight.quantile(0.025)) & (df.weight < df.weight.quantile(0.975)), :]

    f, ax = plt.subplots(figsize=(15, 10))
    corr = df_heat.corr()
    mask = np.triu(corr)
    ax = sns.heatmap(corr, mask=mask, annot=True, cmap='mako', fmt='.1f')
    fig = plt.show()

    fig.savefig('heatmap.png')
    return fig
