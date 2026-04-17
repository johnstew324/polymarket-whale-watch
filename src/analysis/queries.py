import duckdb

from .config import DB_PATH, EXCLUDED_CONTRACTS

_EXCLUDED_SQL = ', '.join(f"'{a}'" for a in EXCLUDED_CONTRACTS)


def _connect():
    return duckdb.connect(str(DB_PATH), read_only=True)


def load_wallet_activity():
    con = _connect()
    df = con.execute("""
        SELECT
            wallet,
            COUNT(*)                       AS trade_count,
            COUNT(DISTINCT condition_id)   AS markets_traded,
            SUM(usd_amount)                AS total_volume,
            MIN(timestamp)                 AS first_trade,
            MAX(timestamp)                 AS last_trade
        FROM trades
        GROUP BY wallet
        ORDER BY trade_count DESC
    """).fetchdf()
    con.close()
    return df


def load_hit_rate():
    con = _connect()
    df = con.execute(f"""
        WITH wallet_positions AS (
            SELECT
                t.wallet,
                t.condition_id,
                t.outcomes                                        AS outcome_token,
                SUM(CASE WHEN t.side = 'BUY'  THEN t.usd_amount
                         WHEN t.side = 'SELL' THEN -t.usd_amount
                         ELSE 0 END)                              AS net_usd,
                SUM(t.usd_amount)                                 AS gross_usd,
                COUNT(*)                                          AS num_trades
            FROM trades t
            WHERE t.wallet NOT IN ({_EXCLUDED_SQL})
            GROUP BY t.wallet, t.condition_id, t.outcomes
        ),
        best_position AS (
            SELECT
                wallet, condition_id, outcome_token,
                net_usd, gross_usd, num_trades,
                ROW_NUMBER() OVER (
                    PARTITION BY wallet, condition_id
                    ORDER BY net_usd DESC
                ) AS rn
            FROM wallet_positions
            WHERE net_usd != 0
        ),
        wallet_bets AS (
            SELECT
                bp.wallet,
                bp.condition_id,
                bp.outcome_token                                  AS bet_on,
                bp.net_usd,
                bp.gross_usd,
                bp.num_trades,
                m.question,
                m.resolvedOutcome,
                m.closedTime,
                m.endDate,
                CASE
                    WHEN bp.net_usd > 0 AND bp.outcome_token = m.resolvedOutcome THEN 1
                    WHEN bp.net_usd < 0 AND bp.outcome_token != m.resolvedOutcome THEN 1
                    ELSE 0
                END AS hit
            FROM best_position bp
            JOIN markets m ON bp.condition_id = m.conditionId
            WHERE bp.rn = 1
              AND m.resolvedOutcome IS NOT NULL
              AND m.resolvedOutcome != ''
        )
        SELECT
            wallet,
            COUNT(*)                                   AS markets_traded,
            SUM(hit)                                   AS markets_correct,
            ROUND(AVG(hit) * 100, 2)                   AS hit_rate_pct,
            SUM(gross_usd)                             AS total_volume,
            SUM(num_trades)                            AS total_trades,
            ROUND(AVG(gross_usd), 2)                   AS avg_volume_per_market
        FROM wallet_bets
        GROUP BY wallet
        ORDER BY markets_traded DESC, hit_rate_pct DESC
    """).fetchdf()
    con.close()
    return df


def load_wallet_edge():
    con = _connect()
    df = con.execute(f"""
        WITH bet_rows AS (
            SELECT
                t.wallet,
                t.condition_id,
                AVG(CASE WHEN t.side = 'BUY' THEN t.price ELSE NULL END) AS avg_price,
                MAX(CASE
                    WHEN t.side = 'BUY' AND t.outcomes = m.resolvedOutcome THEN 1
                    WHEN t.side = 'BUY' AND t.outcomes != m.resolvedOutcome THEN 0
                    ELSE NULL END) AS outcome_correct
            FROM trades t
            JOIN markets m ON t.condition_id = m.conditionId
            WHERE t.wallet NOT IN ({_EXCLUDED_SQL})
              AND m.resolvedOutcome IS NOT NULL
              AND t.side = 'BUY'
            GROUP BY t.wallet, t.condition_id
        )
        SELECT
            wallet,
            COUNT(*)                                     AS markets_with_edge,
            AVG(outcome_correct - avg_price)             AS mean_realised_edge,
            AVG(outcome_correct)                         AS mean_outcome,
            AVG(avg_price)                               AS mean_avg_price
        FROM bet_rows
        WHERE outcome_correct IS NOT NULL
        GROUP BY wallet
    """).fetchdf()
    con.close()
    return df


def load_timing():
    con = _connect()
    df = con.execute("""
        SELECT
            t.wallet,
            t.condition_id,
            t.timestamp                                                       AS trade_ts,
            COALESCE(m.closedTime, m.endDate)                                 AS resolution_ts,
            t.usd_amount,
            t.side,
            DATEDIFF('day', t.timestamp, COALESCE(m.closedTime, m.endDate))   AS days_before_resolution
        FROM trades t
        JOIN markets m ON t.condition_id = m.conditionId
        WHERE t.side = 'BUY'
          AND COALESCE(m.closedTime, m.endDate) IS NOT NULL
          AND t.timestamp < COALESCE(m.closedTime, m.endDate)
    """).fetchdf()
    con.close()
    return df


def load_domain_volumes(top_wallets):
    wallet_list_sql = ', '.join(f"'{w}'" for w in top_wallets)
    con = _connect()
    df = con.execute(f"""
        WITH trade_markets AS (
            SELECT
                t.wallet,
                t.condition_id,
                m.question,
                m.tags,
                SUM(t.usd_amount) AS volume_in_market
            FROM trades t
            JOIN markets m ON t.condition_id = m.conditionId
            WHERE t.wallet IN ({wallet_list_sql})
            GROUP BY t.wallet, t.condition_id, m.question, m.tags
        )
        SELECT
            tm.wallet,
            u.tag,
            SUM(tm.volume_in_market) AS tag_volume
        FROM trade_markets tm
        CROSS JOIN UNNEST(tm.tags) AS u(tag)
        GROUP BY tm.wallet, u.tag
        ORDER BY tm.wallet, tag_volume DESC
    """).fetchdf()
    con.close()
    return df


def load_post_cutoff_wallets(cutoff):
    con = _connect()
    df = con.execute(f"""
        SELECT DISTINCT wallet
        FROM trades
        WHERE timestamp >= '{cutoff}'
    """).fetchdf()
    con.close()
    return df