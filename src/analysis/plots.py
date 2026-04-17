import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from .config import PALETTE, FIG_DIR, MIN_MARKETS, HIT_RATE_THRESHOLD


def _register_template():
    pio.templates['whalewatch'] = go.layout.Template(
        layout=dict(
            font=dict(family='Helvetica, Arial, sans-serif', color=PALETTE['fignavy']),
            colorway=[PALETTE['fignavy'], PALETTE['figblue'], PALETTE['figred'],
                      PALETTE['figgreen'], PALETTE['figlblue'], PALETTE['figgrey']],
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='#e6e7eb', zerolinecolor='#b8bcc4'),
            yaxis=dict(gridcolor='#e6e7eb', zerolinecolor='#b8bcc4'),
        ),
    )
    pio.templates.default = 'whalewatch'


_register_template()
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_wallet_activity(wallet_activity, save=True):
    fig = px.histogram(
        wallet_activity, x='trade_count', nbins=100, log_y=True,
        title='Wallet activity distribution (log scale)',
        labels={'trade_count': 'Number of trades', 'count': 'Number of wallets'},
        color_discrete_sequence=[PALETTE['fignavy']],
    )
    median_trades = wallet_activity['trade_count'].median()
    fig.add_vline(x=median_trades, line_dash='dash', line_color=PALETTE['figred'],
                  annotation_text=f'Median: {median_trades:.0f}')
    fig.update_layout(height=450)
    if save:
        fig.write_image(str(FIG_DIR / 'wallet_activity.png'))
    return fig


def plot_cum_volume(wallet_activity, save=True):
    wa = wallet_activity.sort_values('total_volume', ascending=False).reset_index(drop=True)
    wa['cum_volume_pct'] = wa['total_volume'].cumsum() / wa['total_volume'].sum() * 100
    wa['wallet_rank'] = range(1, len(wa) + 1)
    plot_df = wa.head(500)

    fig = px.line(
        plot_df, x='wallet_rank', y='cum_volume_pct',
        title='Cumulative percentage of total volume by wallet rank',
        labels={'wallet_rank': 'Wallet rank (by volume)', 'cum_volume_pct': 'Cumulative volume %'},
        color_discrete_sequence=[PALETTE['fignavy']],
    )
    for threshold in [50, 80, 90]:
        mask = plot_df['cum_volume_pct'] >= threshold
        if mask.any():
            idx = mask.idxmax()
            rank_at = plot_df.loc[idx, 'wallet_rank']
            fig.add_annotation(
                x=rank_at, y=threshold,
                text=f'{threshold}% volume = top {rank_at} wallets',
                showarrow=True, arrowhead=2, arrowcolor=PALETTE['figred'],
            )
    fig.update_layout(height=450)
    if save:
        fig.write_image(str(FIG_DIR / 'cum_volume_curve.png'))
    return fig


def plot_hit_rate_distribution(experienced, save=True):
    fig = px.histogram(
        experienced, x='hit_rate_pct', nbins=40,
        title=f'Hit-rate distribution (wallets with {MIN_MARKETS}+ resolved markets)',
        labels={'hit_rate_pct': 'Hit rate (%)', 'count': 'Number of wallets'},
        color_discrete_sequence=[PALETTE['fignavy']],
    )
    fig.add_vline(x=50, line_dash='dash', line_color=PALETTE['figgrey'],
                  annotation_text='50% (coin flip)')
    fig.add_vline(x=HIT_RATE_THRESHOLD, line_dash='dash', line_color=PALETTE['figred'],
                  annotation_text=f'{HIT_RATE_THRESHOLD}% threshold')
    n_above = (experienced['hit_rate_pct'] >= HIT_RATE_THRESHOLD).sum()
    fig.add_annotation(
        x=80, y=0, yref='paper', yanchor='bottom',
        text=f'{n_above:,} wallets ≥ {HIT_RATE_THRESHOLD}% hit rate',
        showarrow=False, font=dict(size=14, color=PALETTE['figred']),
    )
    fig.update_layout(height=450)
    if save:
        fig.write_image(str(FIG_DIR / 'hitrate.png'))
    return fig


def plot_volume_vs_hit(experienced, save=True):
    vol_75 = experienced['total_volume'].quantile(0.75)
    fig = px.scatter(
        experienced,
        x='hit_rate_pct', y='total_volume',
        size='markets_traded', size_max=25,
        hover_data=['wallet', 'markets_traded', 'markets_correct', 'total_trades'],
        log_y=True,
        title=f'Volume vs hit rate (wallets with {MIN_MARKETS}+ markets, contracts excluded)',
        labels={
            'hit_rate_pct': 'Hit rate (%)',
            'total_volume': 'Total volume USD (log scale)',
            'markets_traded': 'Markets traded',
        },
        color='hit_rate_pct',
        color_continuous_scale=[PALETTE['figgrey'], PALETTE['figlblue'], PALETTE['fignavy']],
        range_color=[30, 80],
    )
    fig.add_vline(x=HIT_RATE_THRESHOLD, line_dash='dot', line_color=PALETTE['figred'], opacity=0.6)
    fig.add_hline(y=vol_75, line_dash='dot', line_color=PALETTE['figblue'], opacity=0.6,
                  annotation_text=f'Q75 volume: ${vol_75:,.0f}')
    y_top = experienced['total_volume'].quantile(0.95)
    fig.add_annotation(x=78, y=y_top, text='WHALE QUADRANT',
                       showarrow=False, font=dict(color=PALETTE['figgreen'], size=12))
    fig.update_layout(height=550)
    if save:
        fig.write_image(str(FIG_DIR / 'vol_hit.png'))
    return fig


def plot_trade_timing(timing_df, save=True):
    capped = timing_df[timing_df['days_before_resolution'] <= 180]
    fig = px.histogram(
        capped, x='days_before_resolution', nbins=60,
        title='BUY Trade Timing: Days Before Market Resolution',
        labels={'days_before_resolution': 'Days Before Resolution', 'count': 'Number of Trades'},
        color_discrete_sequence=[PALETTE['figred']],
    )
    med = capped['days_before_resolution'].median()
    fig.add_vline(x=med, line_dash='dash', line_color='black',
                  annotation_text=f'Median: {med:.0f}d')
    fig.update_layout(template='plotly_white', height=450)
    if save:
        fig.write_image(str(FIG_DIR / 'days_before.png'))
    return fig


def plot_timing_vs_hit(timing_merged, save=True):
    fig = px.scatter(
        timing_merged,
        x='avg_days_before', y='hit_rate_pct',
        size='total_volume', size_max=20,
        hover_data=['wallet', 'markets_traded', 'total_volume'],
        title='Trade Timing vs Hit Rate — Do Informed Wallets Buy Early?',
        labels={
            'avg_days_before': 'Avg Days Before Resolution (BUY trades)',
            'hit_rate_pct': 'Hit Rate (%)',
        },
        color='hit_rate_pct',
        color_continuous_scale=[PALETTE['figgrey'], PALETTE['figlblue'], PALETTE['fignavy']],
        range_color=[30, 80],
    )
    fig.add_hline(y=HIT_RATE_THRESHOLD, line_dash='dot', line_color='red', opacity=0.5)
    fig.update_layout(template='plotly_white', height=500)
    if save:
        fig.write_image(str(FIG_DIR / 'timing_hit.png'))
    return fig


def plot_specialisation_heatmap(domain_df, specialisation, save=True):
    top15 = specialisation.nlargest(15, 'total_volume')['wallet'].tolist()
    top_tags = (
        domain_df[domain_df['wallet'].isin(top15)]
        .groupby('tag')['tag_volume'].sum()
        .nlargest(10).index.tolist()
    )
    heat = (
        domain_df[
            (domain_df['wallet'].isin(top15))
            & (domain_df['tag'].isin(top_tags))
        ]
        .pivot_table(index='wallet', columns='tag', values='tag_share', fill_value=0)
    )
    heat.index = [
        f'{w[:6]}...{w[-4:]}' if len(str(w)) > 14 else str(w)
        for w in heat.index
    ]
    fig = px.imshow(
        heat.values,
        x=heat.columns.tolist(), y=heat.index.tolist(),
        color_continuous_scale=[[0, 'white'], [1, PALETTE['fignavy']]],
        title='Domain Specialisation: Top 15 Wallets x Market Tags (share of wallet volume)',
        labels={'color': 'Volume Share'},
        aspect='auto',
    )
    fig.update_layout(template='plotly_white', height=500)
    if save:
        fig.write_image(str(FIG_DIR / 'heatmap.png'))
    return fig