from .config import (
    COLD_START_CUTOFF, HIT_RATE_THRESHOLD, SMART_WALLETS_CSV,
)
from .queries import (
    load_wallet_activity, load_hit_rate, load_wallet_edge,
    load_timing, load_domain_volumes, load_post_cutoff_wallets,
)
from .metrics import (
    filter_experienced, threshold_sensitivity, build_whale_quadrant,
    per_wallet_timing, build_early_accurate, build_specialisation,
    build_shortlist, survival_analysis,
    wallet_bootstrap, whale_count_bootstrap, early_accurate_count_bootstrap,
)
from .plots import (
    plot_wallet_activity, plot_cum_volume, plot_hit_rate_distribution,
    plot_volume_vs_hit, plot_trade_timing, plot_timing_vs_hit,
    plot_specialisation_heatmap,
)


def main():
    # ── Section 1: wallet activity and volume concentration ──s
    wallet_activity = load_wallet_activity()
    print(f'Total unique wallets: {len(wallet_activity):,}')
    print(f'Total trades:         {wallet_activity["trade_count"].sum():,}')
    print(f'Total volume:         ${wallet_activity["total_volume"].sum():,.0f}')

    plot_wallet_activity(wallet_activity)
    plot_cum_volume(wallet_activity)

    # ── Section 2: hit rate ──
    hit_rate_df = load_hit_rate()
    print('\nThreshold sensitivity:')
    print(threshold_sensitivity(hit_rate_df).to_string(index=False))

    experienced = filter_experienced(hit_rate_df)
    print(f'\nExperienced wallets: {len(experienced):,}')

    plot_hit_rate_distribution(experienced)

    # ── Section 3: volume × hit rate × realised edge ──
    wallet_edge = load_wallet_edge()
    print(f'\nWallets with realised-edge observation: {len(wallet_edge):,}')
    print(f'Mean realised edge (all wallets): {wallet_edge["mean_realised_edge"].mean():+.4f}')

    plot_volume_vs_hit(experienced)

    whales, vol_75 = build_whale_quadrant(experienced, wallet_edge)
    print(f'\nWhale quadrant: {len(whales)} wallets')
    print(f'Whale mean realised edge: {whales["mean_realised_edge"].mean():+.4f}')

    # Bootstrap CIs
    m, lo, hi = wallet_bootstrap(experienced['hit_rate_pct'], stat='median')
    print(f'\nBootstrap 95% CIs (wallet-clustered):')
    print(f'  Median hit rate:                {m:5.2f}%  [{lo:.2f}, {hi:.2f}]')
    m, lo, hi = wallet_bootstrap(whales['mean_realised_edge'].dropna())
    print(f'  Whale mean realised edge:       {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]')
    m, lo, hi = wallet_bootstrap(wallet_edge['mean_realised_edge'])
    print(f'  All-wallet mean realised edge:  {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]')
    m, lo, hi = whale_count_bootstrap(experienced, vol_75)
    print(f'  Whale-quadrant count:           {m:,}  [{lo:,}, {hi:,}]')

    # ── Section 4: trade timing ──
    timing_df = load_timing()
    print(f'\nBUY trades before resolution: {len(timing_df):,}')

    plot_trade_timing(timing_df)

    wallet_timing = per_wallet_timing(timing_df)
    early_accurate, timing_p75, timing_merged = build_early_accurate(experienced, wallet_timing)
    print(f'Early + accurate wallets: {len(early_accurate)} (threshold {timing_p75:.0f}d)')

    plot_timing_vs_hit(timing_merged)

    m, lo, hi = early_accurate_count_bootstrap(timing_merged, timing_p75)
    print(f'Early-accurate bootstrap CI:     {m:,}  [{lo:,}, {hi:,}]')

    # ── Section 5: domain specialisation ──
    top50 = experienced.nlargest(50, 'total_volume')['wallet'].tolist()
    domain_df = load_domain_volumes(top50)
    specialisation, domain_df = build_specialisation(domain_df, experienced)
    print('\nHerfindahl percentiles (top 50 wallets):')
    for p in [25, 50, 75, 90]:
        print(f'  p{p}: {specialisation["herfindahl"].quantile(p/100):.3f}')

    plot_specialisation_heatmap(domain_df, specialisation)

    # ── Section 6: cold-start survival and shortlist ──
    post_cutoff = load_post_cutoff_wallets(COLD_START_CUTOFF)
    n_whales, n_post, n_survived = survival_analysis(whales, post_cutoff)
    print(f'\nCold-start survival ({COLD_START_CUTOFF}):')
    print(f'  Whale wallets: {n_whales:,}')
    print(f'  Active post-cutoff: {n_post:,}')
    print(f'  Whales surviving: {n_survived:,} ({n_survived/max(n_whales,1)*100:.1f}%)')

    shortlist = build_shortlist(experienced, wallet_timing, specialisation, wallet_edge)
    print(f'\nSmart-wallet shortlist: {len(shortlist)} wallets')

    export_cols = [
        'wallet', 'markets_traded', 'markets_correct', 'hit_rate_pct',
        'mean_realised_edge', 'total_volume', 'total_trades',
        'avg_volume_per_market', 'avg_days_before', 'median_days_before',
        'num_buys', 'herfindahl', 'top_tag', 'top_tag_share',
    ]
    export_cols = [c for c in export_cols if c in shortlist.columns]
    SMART_WALLETS_CSV.parent.mkdir(parents=True, exist_ok=True)
    shortlist[export_cols].to_csv(SMART_WALLETS_CSV, index=False)
    print(f'Exported {len(shortlist)} wallets -> {SMART_WALLETS_CSV}')


if __name__ == '__main__':
    main()