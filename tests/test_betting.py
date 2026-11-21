from datetime import datetime, timezone

import pandas as pd
import pytest

from bigdance.betting import (
    BetLog,
    OddsApi,
    OddsApiError,
    american_to_decimal,
    american_to_prob,
    append_snapshot,
    devig_two_way,
    find_edges,
    game_snapshot,
    kelly_stake,
    map_team_names,
    normalize_name,
    parse_odds_response,
    prob_to_american,
    read_quota,
    settle_snapshots,
    snapshot_games_due,
    update_snapshot_closing,
    write_quota,
)

TIP = "2026-11-20T00:00:00Z"  # 7pm ET on Nov 19


def make_events():
    return [
        {
            "id": "g1",
            "commence_time": TIP,
            "home_team": "Duke Blue Devils",
            "away_team": "Michigan State Spartans",
            "bookmakers": [
                {
                    "key": "bookA",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Duke Blue Devils", "price": -200},
                                {"name": "Michigan State Spartans", "price": 170},
                            ],
                        }
                    ],
                },
                {
                    "key": "bookB",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Duke Blue Devils", "price": -180},
                                {"name": "Michigan State Spartans", "price": 160},
                            ],
                        }
                    ],
                },
            ],
        }
    ]


def make_matchups(elo_prob_home):
    return pd.DataFrame(
        {
            "team1": ["Michigan St."],
            "team2": ["Duke"],
            "elo_prob1": [1 - elo_prob_home],
            "elo_prob2": [elo_prob_home],
        }
    )


def test_odds_conversions():
    assert american_to_prob(-200) == pytest.approx(2 / 3)
    assert american_to_prob(150) == pytest.approx(0.4)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert prob_to_american(0.5) == pytest.approx(-100)
    assert american_to_prob(prob_to_american(0.7)) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        prob_to_american(1.0)


def test_devig_sums_to_one():
    a, b = devig_two_way(american_to_prob(-110), american_to_prob(-110))
    assert (a, b) == (pytest.approx(0.5), pytest.approx(0.5))


def test_kelly_stake_caps_and_floors():
    assert kelly_stake(0.4, 2.0) == 0.0  # negative EV
    assert kelly_stake(0.9, 2.0, fraction=1.0, max_fraction=0.05) == 0.05
    assert kelly_stake(0.55, 2.0, fraction=0.5, max_fraction=1.0) == pytest.approx(0.05)


def test_name_matching_prefers_longest_and_handles_st():
    assert normalize_name("St. John's Red Storm") == "saint john s red storm"
    mapping, unmatched = map_team_names(
        ["Michigan Wolverines", "Michigan State Spartans", "Nowhere Tech Owls"],
        ["Michigan", "Michigan St."],
    )
    assert mapping["Michigan Wolverines"] == "Michigan"
    assert mapping["Michigan State Spartans"] == "Michigan St."
    assert unmatched == ["Nowhere Tech Owls"]
    mapping, _ = map_team_names(["Foo Bar"], ["Baz"], overrides={"Foo Bar": "Baz"})
    assert mapping["Foo Bar"] == "Baz"


def test_parse_odds_response():
    df = parse_odds_response(make_events())
    assert len(df) == 2
    assert set(df["book"]) == {"bookA", "bookB"}
    assert df["home_price"].tolist() == [-200.0, -180.0]


def test_find_edges_flags_underpriced_home_team():
    odds = parse_odds_response(make_events())
    edges = find_edges(
        make_matchups(0.85),
        odds,
        model_weight=1.0,
        game_date="2026-11-19",
        max_divergence=1.0,
        max_ev=10.0,
    )
    assert len(edges) == 1
    e = edges.iloc[0]
    assert e["odds_team"] == "Duke Blue Devils"
    assert e["book"] == "bookB"  # best price on Duke
    assert e["ev"] == pytest.approx(0.85 * american_to_decimal(-180) - 1)
    assert 0 < e["stake_frac"] <= 0.05


def test_find_edges_flags_away_underdog_and_respects_blend():
    odds = parse_odds_response(make_events())
    full = find_edges(make_matchups(0.40), odds, model_weight=1.0, max_divergence=1.0, max_ev=10.0)
    assert full.iloc[0]["odds_team"] == "Michigan State Spartans"
    assert full.iloc[0]["book"] == "bookA"  # +170 beats +160
    # With the model fully ignored, there is no edge against a vigged market
    assert find_edges(make_matchups(0.40), odds, model_weight=0.0).empty


def test_find_edges_date_filter_and_validation():
    odds = parse_odds_response(make_events())
    assert find_edges(make_matchups(0.85), odds, model_weight=1.0, game_date="2026-11-20").empty
    with pytest.raises(ValueError):
        find_edges(make_matchups(0.85), odds, model_weight=1.5)
    with pytest.raises(ValueError):
        find_edges(make_matchups(0.85).drop(columns="elo_prob2"), odds)


def test_betlog_clv_and_results(tmp_path):
    path = str(tmp_path / "bets.csv")
    odds = parse_odds_response(make_events())
    edges = find_edges(make_matchups(0.85), odds, model_weight=1.0, max_divergence=1.0, max_ev=10.0)

    log = BetLog(path)
    assert log.add_edges(edges, bankroll=1000) == 1
    assert log.add_edges(edges, bankroll=1000) == 0  # no duplicates

    # Line moves toward Duke before tip: closing price -250/+210 on a single book
    closing = odds.iloc[:1].copy()
    closing["home_price"], closing["away_price"] = -250.0, 210.0
    before_tip = datetime(2026, 11, 19, 23, 0, tzinfo=timezone.utc)
    assert BetLog(path).update_closing(closing, now=before_tip) == 1

    log = BetLog(path)
    assert log.df.iloc[0]["clv"] > 0  # took -180, closed at -250: beat the close
    after_tip = datetime(2026, 11, 20, 1, 0, tzinfo=timezone.utc)
    assert log.update_closing(closing, now=after_tip) == 0  # frozen once the game starts

    bet_id = log.df.iloc[0]["bet_id"]
    log.record_result(bet_id, won=True)
    s = BetLog(path).summary()
    assert s["settled"] == 1 and s["profit"] > 0 and s["avg_clv"] > 0
    with pytest.raises(KeyError):
        log.record_result("nope", won=False)


def test_max_divergence_filters_implausible_edges():
    odds = parse_odds_response(make_events())
    # Elo says 85% vs a ~64% market: a 20+ point gap is treated as model error
    assert find_edges(make_matchups(0.85), odds, model_weight=1.0, max_divergence=0.10).empty
    assert len(find_edges(make_matchups(0.72), odds, model_weight=1.0, max_divergence=0.10)) == 1


def test_max_ev_filters_implausible_edges():
    odds = parse_odds_response(make_events())
    kwargs = {"model_weight": 1.0, "max_divergence": 1.0}
    assert len(find_edges(make_matchups(0.85), odds, max_ev=10.0, **kwargs)) == 1
    assert find_edges(make_matchups(0.85), odds, max_ev=0.25, **kwargs).empty


def _logged(tmp_path):
    path = str(tmp_path / "bets.csv")
    odds = parse_odds_response(make_events())
    edges = find_edges(make_matchups(0.85), odds, model_weight=1.0, max_divergence=1.0, max_ev=10.0)
    log = BetLog(path)
    log.add_edges(edges, bankroll=1000)
    return path, odds, log


def test_due_for_closing_and_windowed_update(tmp_path):
    path, odds, log = _logged(tmp_path)  # tips at 2026-11-20 00:00Z
    assert log.due_for_closing(20, now=datetime(2026, 11, 19, 22, 0, tzinfo=timezone.utc)) == 0
    assert log.due_for_closing(20, now=datetime(2026, 11, 19, 23, 45, tzinfo=timezone.utc)) == 1
    assert log.due_for_closing(20, now=datetime(2026, 11, 20, 0, 5, tzinfo=timezone.utc)) == 0

    early = datetime(2026, 11, 19, 22, 0, tzinfo=timezone.utc)
    assert log.update_closing(odds, now=early, window_minutes=20) == 0  # too early: no-op
    near = datetime(2026, 11, 19, 23, 45, tzinfo=timezone.utc)
    assert log.update_closing(odds, now=near, window_minutes=20) == 1


def test_settle_from_scores(tmp_path):
    path, odds, log = _logged(tmp_path)
    final = pd.DataFrame(
        {
            "team1": ["Michigan St."],
            "team2": ["Duke"],
            "score1": [60],
            "score2": [72],
        }
    )
    seen = []

    def fake_matchups(date):
        seen.append(date)
        return final

    too_soon = datetime(2026, 11, 20, 2, 0, tzinfo=timezone.utc)
    assert log.settle(now=too_soon, matchups_for_date=fake_matchups) == 0
    assert seen == []  # didn't even fetch scores

    later = datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc)
    assert log.settle(now=later, matchups_for_date=fake_matchups) == 1
    assert seen == ["2026-11-19"]  # US/Eastern date of the tip-off
    bet = BetLog(path).df.iloc[0]
    assert bet["result"] == "win" and bet["profit"] > 0

    # Unscored (postponed / not yet updated) games stay open
    path2 = str(tmp_path / "bets2.csv")
    log2 = BetLog(path2)
    log2.add_edges(
        find_edges(make_matchups(0.85), odds, model_weight=1.0, max_divergence=1.0, max_ev=10.0),
        1000,
    )
    unscored = final.assign(score1=0, score2=0)
    assert log2.settle(now=later, matchups_for_date=lambda d: unscored) == 0


def test_game_snapshot_covers_all_games_and_dedupes(tmp_path):
    odds = parse_odds_response(make_events())
    snap = game_snapshot(make_matchups(0.72), odds)
    assert len(snap) == 1
    assert snap.iloc[0]["model_prob_home"] == pytest.approx(0.72)
    assert 0.6 < snap.iloc[0]["market_prob_home"] < 0.7

    path = str(tmp_path / "snap.csv")
    assert append_snapshot(path, snap) == 1
    assert append_snapshot(path, snap) == 1  # same game replaced, not duplicated
    assert len(pd.read_csv(path)) == 1


class _FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else []
        self.headers = {"x-requests-remaining": "123"}

    def json(self):
        return self._payload


def test_odds_api_retries_network_errors_and_hides_key(monkeypatch):
    import requests

    calls = []
    sleeps = []

    def flaky_get(url, params=None, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            raise requests.ConnectionError(f"DNS failed for {url}?apiKey={params['apiKey']}")
        return _FakeResp(payload=make_events())

    monkeypatch.setattr(requests, "get", flaky_get)
    monkeypatch.setattr("bigdance.betting.time.sleep", sleeps.append)
    api = OddsApi(api_key="SECRETKEY")
    assert len(api.fetch_h2h()) == 2
    assert len(calls) == 3 and len(sleeps) == 2 and api.requests_remaining == "123"

    def always_down(url, params=None, timeout=None):
        raise requests.ConnectionError(f"boom {url}?apiKey={params['apiKey']}")

    monkeypatch.setattr(requests, "get", always_down)
    with pytest.raises(OddsApiError) as exc:
        api.fetch_h2h()
    assert "SECRETKEY" not in str(exc.value) and exc.value.__cause__ is None


def test_odds_api_client_errors_fail_fast(monkeypatch):
    import requests

    calls = []

    def unauthorized(url, params=None, timeout=None):
        calls.append(1)
        return _FakeResp(status=401)

    monkeypatch.setattr(requests, "get", unauthorized)
    monkeypatch.setattr("bigdance.betting.time.sleep", lambda s: None)
    with pytest.raises(OddsApiError, match="HTTP 401"):
        OddsApi(api_key="SECRETKEY").fetch_h2h()
    assert len(calls) == 1  # no retry on a 4xx


def test_snapshot_closing_results_and_rerun_safety(tmp_path):
    odds = parse_odds_response(make_events())  # tips 2026-11-20T00:00Z
    path = str(tmp_path / "snap.csv")
    snap = game_snapshot(make_matchups(0.72), odds)
    append_snapshot(path, snap)

    early = datetime(2026, 11, 19, 22, 0, tzinfo=timezone.utc)
    near = datetime(2026, 11, 19, 23, 45, tzinfo=timezone.utc)
    assert snapshot_games_due(path, 35, now=early) == 0
    assert snapshot_games_due(path, 35, now=near) == 1
    assert update_snapshot_closing(path, odds, now=early, window_minutes=35) == 0
    assert update_snapshot_closing(path, odds, now=near, window_minutes=35) == 1
    row = pd.read_csv(path).iloc[0]
    assert 0.6 < row["closing_prob_home"] < 0.7

    # Re-running the morning snapshot must not wipe captured closing data
    append_snapshot(path, snap)
    assert pd.read_csv(path).iloc[0]["closing_prob_home"] == pytest.approx(row["closing_prob_home"])

    final = pd.DataFrame(
        {"team1": ["Michigan St."], "team2": ["Duke"], "score1": [60], "score2": [72]}
    )
    too_soon = datetime(2026, 11, 20, 2, 0, tzinfo=timezone.utc)
    assert settle_snapshots(path, now=too_soon, matchups_for_date=lambda d: final) == 0
    later = datetime(2026, 11, 20, 6, 0, tzinfo=timezone.utc)
    assert settle_snapshots(path, now=later, matchups_for_date=lambda d: final) == 1
    assert pd.read_csv(path).iloc[0]["home_won"] == 1.0  # Duke (home) won


def test_snapshot_file_from_older_version_still_loads(tmp_path):
    path = str(tmp_path / "old.csv")
    pd.DataFrame(
        {
            "snapshot_at": ["x"],
            "game_id": ["g1"],
            "commence_time": [TIP],
            "home_team": ["Duke Blue Devils"],
            "away_team": ["Michigan State Spartans"],
            "model_prob_home": [0.7],
            "market_prob_home": [0.65],
        }
    ).to_csv(path, index=False)
    assert (
        snapshot_games_due(path, 35, now=datetime(2026, 11, 19, 23, 45, tzinfo=timezone.utc)) == 1
    )


def test_quota_roundtrip(tmp_path):
    path = str(tmp_path / "snap.csv")
    assert read_quota(path) is None
    write_quota(path, "367")
    assert read_quota(path) == 367


def test_known_name_overrides_resolve_to_warren_nolan_names():
    wn = [
        "Little Rock", "Cal State Bakersfield", "Cal State Northridge", "George Washington",
        "Mount Saint Mary's", "North Carolina State", "Southeastern Louisiana",
        "Southeast Missouri", "Tennessee-Martin", "UTA", "Connecticut", "Hawaii",
        "California Baptist", "UMass", "LSU New Orleans", "Penn",
    ]  # fmt: skip
    odds = [
        "Arkansas-Little Rock Trojans", "CSU Bakersfield Roadrunners", "CSU Northridge Matadors",
        "GW Revolutionaries", "Mt. St. Mary's Mountaineers", "NC State Wolfpack",
        "SE Louisiana Lions", "SE Missouri St Redhawks", "Tenn-Martin Skyhawks",
        "UT-Arlington Mavericks", "UConn Huskies", "Hawai'i Rainbow Warriors",
        "Cal Baptist Lancers", "Massachusetts Minutemen", "New Orleans Privateers",
        "Pennsylvania Quakers",
    ]  # fmt: skip
    mapping, unmatched = map_team_names(odds, wn)
    assert unmatched == []
    assert mapping["NC State Wolfpack"] == "North Carolina State"
    assert mapping["UT-Arlington Mavericks"] == "UTA"
