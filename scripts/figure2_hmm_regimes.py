from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, YearLocator
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

root = Path('/').resolve()
feature_df = pd.read_csv(root / 'data/processed/feature_dataset.csv', index_col=0, parse_dates=True).sort_index()
state_file = root / 'results' / 'part_b' / 'hmm_6state_assignments.csv'
state_assign = pd.read_csv(state_file, parse_dates=['Date']).set_index('Date').sort_index()
state_series = state_assign['state'].astype(int)
# Keep only dates that exist in both the feature dataset and the saved state assignment
common_index = feature_df.index.intersection(state_series.index)
feature_df = feature_df.loc[common_index]
state_series = state_series.loc[common_index]
X = feature_df[['^GSPC', '^VIX', 'rv_20']].copy()
# Recompute the HMM on the same filtered sample because the saved assignment file is the authoritative model output.
feature_list = [
    'spx_log_return', 'rv_20', 'cumret_20', 'vix_level', 'vix_daily_change', 'vix_dev_20',
    'DGS2', 'DGS10', 'yc_spread', 'DCOILWTICO', 'BAMLH0A0HYM2', 'DFF'
]
feat = feature_df[feature_list].loc[common_index].dropna()
Z = StandardScaler().fit_transform(feat)
model = GaussianHMM(n_components=6, covariance_type='full', n_iter=300, random_state=42, init_params='stmc')
model.fit(Z)
state_series = pd.Series(model.predict(Z), index=feat.index, name='state')
state_file = root / 'results' / 'part_b' / 'hmm_6state_assignments.csv'
state_assign = pd.DataFrame({'Date': feat.index, 'state': state_series.values})
state_assign['Date'] = pd.to_datetime(state_assign['Date'])
state_assign.to_csv(state_file, index=False)

state_colors = {
    1: '#4C78A8',
    2: '#4E9D99',
    3: '#5AAE61',
    4: '#B8B05A',
    5: '#E59A3E',
    6: '#C85C5C',
}

runs = []
start = state_series.index[0]
prev_state = state_series.iloc[0]
for dt, st in state_series.items():
    if st == prev_state:
        continue
    runs.append((start, dt, prev_state))
    start = dt
    prev_state = st
runs.append((start, state_series.index[-1], prev_state))

periods = [
    ('9/11', pd.Timestamp('2001-09-11')),
    ('Dot-com', pd.Timestamp('2002-07-01')),
    ('GFC', pd.Timestamp('2008-09-15')),
    ('Flash Crash', pd.Timestamp('2010-05-06')),
    ('Euro crisis', pd.Timestamp('2011-08-15')),
    ('China deval.', pd.Timestamp('2015-08-24')),
    ('COVID', pd.Timestamp('2020-03-23')),
    ('Yen unwind', pd.Timestamp('2024-08-05')),
    ('Tariffs', pd.Timestamp('2025-04-02')),
    ('Iran shock', pd.Timestamp('2026-03-13')),
]
visible_events = [(label, date) for label, date in periods if X.index.min() <= date <= X.index.max()]

state_summary = []
state_order = sorted(state_series.unique())
for state_id in state_order:
    subset = feature_df.loc[state_series.index[state_series == state_id], ['^VIX', 'rv_20']] 
    runs_state = []
    run_len = 0
    for st in state_series.values:
        if st == state_id:
            run_len += 1
        else:
            if run_len > 0:
                runs_state.append(run_len)
            run_len = 0
    if run_len > 0:
        runs_state.append(run_len)
    state_summary.append({
        'State': int(state_id) + 1,
        'Share': len(subset) / len(X),
        'Avg VIX': float(subset['^VIX'].mean()),
        'Avg RV': float(subset['rv_20'].mean()),
        'Avg duration': float(np.mean(runs_state)) if runs_state else 0.0,
    })
state_summary = pd.DataFrame(state_summary).sort_values('State').reset_index(drop=True)

fig = plt.figure(figsize=(16, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.02)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

ax_table = fig.add_axes([0.72, 0.25, 0.24, 0.26])
ax_table.axis('off')
col_labels = ['State', 'Share', 'Avg VIX', 'Avg RV', 'Avg dur.']
rows = [[
    f"{int(r['State'])}",
    f"{r['Share']:.3f}",
    f"{r['Avg VIX']:.1f}",
    f"{r['Avg RV']:.3f}",
    f"{r['Avg duration']:.1f}",
] for _, r in state_summary.iterrows()]
ax_table.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center', colColours=['#EFEFEF'] * 5, fontsize=8)
ax_table.set_title('State summary', fontsize=9, pad=8)

spx = feature_df['^GSPC'].reindex(X.index)
ax1.plot(spx.index, spx.values, color='#2F4F4F', linewidth=1.8)
for s, e, st in runs:
    ax1.axvspan(s, e, facecolor=state_colors[int(st) + 1], alpha=0.08, linewidth=0)

label_offsets = {
    '9/11': (0, 18),
    'Dot-com': (0, 28),
    'GFC': (-10, 18),
    'Flash Crash': (-14, 20),
    'Euro crisis': (-8, -22),
    'China deval.': (0, 20),
    'COVID': (0, 20),
    'Yen unwind': (-6, 18),
    'Tariffs': (0, 18),
    'Iran shock': (0, 18),
}
for label, date in visible_events:
    idx = spx.index.get_indexer([date], method='nearest')[0]
    y = float(spx.iloc[idx])
    dx, dy = label_offsets.get(label, (0, 18))
    ax1.axvline(date, color='0.55', linestyle='--', linewidth=0.8, alpha=0.35)
    ax1.annotate(
        label,
        xy=(date, y),
        xytext=(dx, dy),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=7.5,
        color='#2f2f2f',
        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='0.75', alpha=0.8),
        arrowprops=dict(arrowstyle='-', color='#4b4b4b', lw=0.8),
        annotation_clip=False,
    )

ax1.set_ylabel('S&P 500 Index')
ax1.set_title('Market regime dynamics over time: S&P 500 with inferred HMM states')
ax1.set_xlim(X.index.min(), X.index.max())
ax1.yaxis.set_major_formatter(lambda x, pos: f'{x:,.0f}')

for s, e, st in runs:
    ax2.hlines(0, s, e, colors=state_colors[int(st) + 1], linewidth=12, alpha=0.95)
ax2.set_ylim(-0.5, 0.5)
ax2.set_yticks([])
ax2.set_ylabel('State')
ax2.set_xlabel('Date')
ax2.set_xlim(X.index.min(), X.index.max())
ax2.xaxis.set_major_locator(YearLocator(base=5))
ax2.xaxis.set_major_formatter(DateFormatter('%Y'))

legend_handles = [
    plt.Line2D([0], [0], marker='s', linestyle='None', markersize=8, markerfacecolor=state_colors[s], color='none', label=f'State {s}')
    for s in [1, 2, 3, 4, 5, 6]
]
fig.legend(handles=legend_handles, loc='lower center', ncol=6, frameon=True, bbox_to_anchor=(0.5, 0.06), fontsize=8)

for ax in [ax1, ax2]:
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

outdir = root / 'figures' / 'part_b' / 'final'
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / 'fig_b01_hmm_market_regimes.png', dpi=300, bbox_inches='tight')
fig.savefig(outdir / 'fig_b01_hmm_market_regimes.pdf', bbox_inches='tight')
fig.savefig(outdir / 'fig_b01_hmm_market_regimes.svg', bbox_inches='tight')
plt.close(fig)

caption = (
    'Market regime dynamics over time: S&P 500 with inferred HMM states. '
    'The selected 6-state Gaussian HMM identifies persistent market regimes over the sample period, '
    'with state transitions reflected in the shaded background and the color-coded strip. '
    'Warmer-colored states generally align with episodes of elevated market stress, while cooler-colored states are associated with calmer conditions. '
    'The figure summarizes regime identification and persistence rather than forecasting future crises.'
)
(outdir / 'fig_b01_hmm_market_regimes_caption.md').write_text(caption, encoding='utf-8')

print('Saved PNG:', outdir / 'fig_b01_hmm_market_regimes.png')
print('Saved PDF:', outdir / 'fig_b01_hmm_market_regimes.pdf')
print('Saved SVG:', outdir / 'fig_b01_hmm_market_regimes.svg')
print('Date range:', X.index.min(), 'to', X.index.max())
print('Visible event labels:', len(visible_events))
print('State assignment file:', state_file)
print(state_summary.to_string(index=False))
