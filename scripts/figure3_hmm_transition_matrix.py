from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path('/').resolve()
RESULTS_PATH = ROOT / 'results' / 'part_b'
FIGURE_CONFIG = {
    'title': 'HMM Transition Matrix: State Transition Probabilities and Regime Persistence',
    'panel_title_heatmap': 'Estimated transition matrix',
    'panel_title_table': 'State persistence',
    'base_name': 'fig_b02_hmm_transition_matrix',
    'output_dir': ROOT / 'figures' / 'part_b' / 'final',
    'state_colors': {
        1: '#4C78A8',
        2: '#4E9D99',
        3: '#5AAE61',
        4: '#B8B05A',
        5: '#E59A3E',
        6: '#C85C5C',
    },
    'highlight_cells': {(0, 3), (3, 0)},
    'state_labels': [f'State {i}' for i in range(1, 7)],
}


def load_transition_matrix() -> pd.DataFrame:
    tm = pd.read_csv(RESULTS_PATH / 'hmm_transition_matrix.csv', index_col=0)
    # The saved HMM transition matrix uses the model's native indexing (0..5), but the report labels these as State 1..6.
    if list(tm.index) == FIGURE_CONFIG['state_labels']:
        tm = tm.copy()
    else:
        tm.index = [f'State {i}' for i in range(1, len(tm) + 1)]
        tm.columns = [f'State {i}' for i in range(1, len(tm.columns) + 1)]
    return tm.reindex(index=FIGURE_CONFIG['state_labels'], columns=FIGURE_CONFIG['state_labels'])


def load_state_sequence() -> pd.Series:
    state_series = pd.read_csv(RESULTS_PATH / 'hmm_6state_assignments.csv', parse_dates=['Date']).set_index('Date').sort_index()
    state_series.index = pd.to_datetime(state_series.index)
    state_series = state_series['state'].astype(int)
    return state_series


def build_empirical_transition_matrix(state_series: pd.Series) -> pd.DataFrame:
    states = state_series.to_numpy()
    counts = np.zeros((6, 6), dtype=float)
    for cur, nxt in zip(states[:-1], states[1:]):
        counts[int(cur) - 1, int(nxt) - 1] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)
    df = pd.DataFrame(matrix, index=FIGURE_CONFIG['state_labels'], columns=FIGURE_CONFIG['state_labels'])
    return df


def compute_state_persistence(state_series: pd.Series, transition_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state_id in range(1, 7):
        mask = state_series == state_id
        run_lengths = []
        current_run = 0
        for state in state_series.astype(int):
            if state == state_id:
                current_run += 1
            else:
                if current_run > 0:
                    run_lengths.append(current_run)
                current_run = 0
        if current_run > 0:
            run_lengths.append(current_run)
        self_prob = float(transition_matrix.iloc[state_id - 1, state_id - 1])
        expected_duration = 1.0 / (1.0 - self_prob) if self_prob < 1.0 else np.inf
        rows.append({
            'State ID': state_id,
            'State': f'State {state_id}',
            'Self-transition probability': self_prob,
            'Expected duration (days)': float(expected_duration) if np.isfinite(expected_duration) else np.nan,
            'Observed mean run duration (days)': float(np.mean(run_lengths)) if run_lengths else 0.0,
            'Observed run count': int(len(run_lengths)),
            'Observation share': float(mask.mean()),
        })
    return pd.DataFrame(rows)


def compute_alternation_stats(state_series: pd.Series) -> dict:
    states = state_series.to_numpy()
    alt_14 = int(np.sum((states[:-1] == 1) & (states[1:] == 4)))
    alt_41 = int(np.sum((states[:-1] == 4) & (states[1:] == 1)))
    total = alt_14 + alt_41
    transition_matrix = load_transition_matrix()
    prob_14 = float(transition_matrix.iloc[0, 3])
    prob_41 = float(transition_matrix.iloc[3, 0])
    return {'fitted_p_14': prob_14, 'fitted_p_41': prob_41, 'decoded_14': alt_14, 'decoded_41': alt_41, 'decoded_total': total}


def compute_state_similarity(feature_df: pd.DataFrame, state_series: pd.Series) -> pd.DataFrame:
    merged = feature_df.loc[state_series.index, ['^VIX', 'rv_20']].copy()
    merged['state'] = state_series.values
    summary = merged.groupby('state').agg({'^VIX': 'mean', 'rv_20': 'mean'}).reset_index()
    summary['State'] = [f'State {int(s)}' for s in summary['state']]
    summary = summary.set_index('State')
    return summary


def draw_figure() -> None:
    transition_matrix = load_transition_matrix()
    state_series = load_state_sequence()
    empirical_matrix = build_empirical_transition_matrix(state_series)
    persistence = compute_state_persistence(state_series, transition_matrix)
    alternation = compute_alternation_stats(state_series)

    feature_df = pd.read_csv(ROOT / 'data' / 'processed' / 'feature_dataset.csv', index_col=0, parse_dates=True).sort_index()
    similarity = compute_state_similarity(feature_df, state_series)

    fig, (ax_heatmap, ax_rhs) = plt.subplots(1, 2, figsize=(11.6, 4.8), gridspec_kw={'width_ratios': [3.35, 1.20]})
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.86, wspace=0.03)
    ax_rhs.set_position([0.74, 0.10, 0.22, 0.68])
    ax_rhs.axis('off')

    values = transition_matrix.to_numpy()
    im = ax_heatmap.imshow(values, cmap='Blues', vmin=0, vmax=1)
    ax_heatmap.set_xticks(range(6))
    ax_heatmap.set_yticks(range(6))
    ax_heatmap.set_xticklabels(FIGURE_CONFIG['state_labels'])
    ax_heatmap.set_yticklabels(FIGURE_CONFIG['state_labels'])
    ax_heatmap.set_xlabel('Next state', fontsize=9)
    ax_heatmap.set_ylabel('Current state', fontsize=9)
    ax_heatmap.set_title(FIGURE_CONFIG['panel_title_heatmap'], fontsize=10.5, pad=8)

    for i in range(6):
        for j in range(6):
            val = values[i, j]
            ax_heatmap.text(j, i, f'{val:.4f}', ha='center', va='center', fontsize=8,
                            color='white' if val >= 0.55 else '#1e1e1e')

    for i in range(6):
        for j in range(6):
            if (i + 1, j + 1) in FIGURE_CONFIG['highlight_cells']:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='#9a3f3f', linewidth=1.8, alpha=0.9)
                ax_heatmap.add_patch(rect)

    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_heatmap, pad=0.03, shrink=0.9)
    cbar.set_label('Transition probability', fontsize=9)

    fig.text(0.785, 0.80, 'State persistence', ha='center', va='center', fontsize=10.5)
    ax_table = ax_rhs.inset_axes([0.03, 0.40, 0.94, 0.45])
    ax_table.axis('off')
    table_rows = [[str(row['State ID']), f"{row['Self-transition probability']:.4f}", f"{row['Expected duration (days)']:.1f}"] for _, row in persistence.iterrows()]
    table = ax_table.table(
        cellText=table_rows,
        colLabels=['State', 'P(i→i)', 'E[duration]'],
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2', '#f2f2f2', '#f2f2f2'],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.08, 1.30)

    row_state_1 = persistence.index[persistence['State ID'] == 1][0]
    row_state_4 = persistence.index[persistence['State ID'] == 4][0]
    for r_idx in [row_state_1 + 1, row_state_4 + 1]:
        for c_idx in range(3):
            table[r_idx, c_idx].set_facecolor('#eef3ff')
            table[r_idx, c_idx].set_edgecolor('#d0d8ea')

    ax_note = ax_rhs.inset_axes([0.03, 0.05, 0.94, 0.22])
    ax_note.axis('off')
    note = (
        'Interpretation: States 2–3 are the most persistent calm regimes; '
        'States 1 and 4 are short-lived but high-turnover, with 26 observed 1↔4 switches. '
        'Fitted P(1→4)=' + f'{alternation["fitted_p_14"]:.4f}' + ' and P(4→1)=' + f'{alternation["fitted_p_41"]:.4f}' + '. '
        'States 5–6 are the rare, stressed states.'
    )
    ax_note.text(
        0.0,
        0.95,
        note,
        va='top',
        ha='left',
        fontsize=7.3,
        color='#2f2f2f',
        wrap=True,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#f5f7fb', edgecolor='#d9e1ed', linewidth=0.8),
    )

    fig.suptitle(FIGURE_CONFIG['title'], fontsize=12.5, y=0.97)

    out_dir = Path(FIGURE_CONFIG['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{FIGURE_CONFIG['base_name']}.png"
    pdf_path = out_dir / f"{FIGURE_CONFIG['base_name']}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print('Saved PNG:', png_path)
    print('Saved PDF:', pdf_path)
    print('Transition matrix row sums:', transition_matrix.sum(axis=1).round(12).to_dict())
    print('P(1→4) fitted:', alternation['fitted_p_14'])
    print('P(4→1) fitted:', alternation['fitted_p_41'])
    print('Decoded 1↔4 alternations:', alternation['decoded_total'])
    print('State 1 vs 4 VIX/RV:')
    print(similarity.loc[['State 1', 'State 4'], ['^VIX', 'rv_20']].to_string())
    print('State persistence table:')
    print(persistence.to_string(index=False))


if __name__ == '__main__':
    draw_figure()
