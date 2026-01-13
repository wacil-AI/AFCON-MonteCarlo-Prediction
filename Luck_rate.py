# pip install statsbombpy pandas numpy

from statsbombpy import sb
import pandas as pd
import numpy as np
import time
import random
import warnings

# -----------------------------
# CONFIG
# -----------------------------
START_DATE = "2020-01-01"
N_MATCHES_TARGET = 250         # >= 200
LOW_XG_TH = 0.05
RANDOM_SEED = 42
DEBUG_FIRST_MATCH = True     

warnings.filterwarnings("ignore", message="credentials were not supplied*")


# -----------------------------
# HELPERS
# -----------------------------
def _name(x):
    """Return a 'name' whether x is dict({'name':..}) or already a string."""
    if isinstance(x, dict):
        return x.get("name", np.nan)
    if isinstance(x, str):
        return x
    return np.nan


def get_events_with_retry(match_id: int, max_tries: int = 6):
    for attempt in range(max_tries):
        try:
            return sb.events(match_id=match_id)
        except Exception as e:
            if attempt == max_tries - 1:
                raise e
            time.sleep((2 ** attempt) + random.random())


def standardize_events(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()

    # type_name
    if "type.name" in ev.columns:
        ev["type_name"] = ev["type.name"]
    elif "type_name" in ev.columns:
        ev["type_name"] = ev["type_name"]
    elif "type" in ev.columns:
        ev["type_name"] = ev["type"].apply(_name)
    else:
        ev["type_name"] = np.nan

    # team_name
    if "team.name" in ev.columns:
        ev["team_name"] = ev["team.name"]
    elif "team_name" in ev.columns:
        ev["team_name"] = ev["team_name"]
    elif "team" in ev.columns:
        ev["team_name"] = ev["team"].apply(_name)
    else:
        ev["team_name"] = np.nan

    # player_name (optionnel)
    if "player.name" in ev.columns:
        ev["player_name"] = ev["player.name"]
    elif "player_name" in ev.columns:
        ev["player_name"] = ev["player_name"]
    elif "player" in ev.columns:
        ev["player_name"] = ev["player"].apply(_name)
    else:
        ev["player_name"] = np.nan

    # minute
    ev["minute_std"] = ev["minute"] if "minute" in ev.columns else np.nan

    # shot_xg
    xg_col = None
    for c in ["shot.statsbomb_xg", "shot_statsbomb_xg"]:
        if c in ev.columns:
            xg_col = c
            break

    if xg_col is None:
        cand = [c for c in ev.columns if "statsbomb_xg" in c]
        xg_col = cand[0] if cand else None

    if xg_col is not None:
        ev["shot_xg"] = ev[xg_col]
    elif "shot" in ev.columns:
        ev["shot_xg"] = ev["shot"].apply(lambda s: s.get("statsbomb_xg") if isinstance(s, dict) else np.nan)
    else:
        ev["shot_xg"] = np.nan

    # shot_outcome
    if "shot.outcome.name" in ev.columns:
        ev["shot_outcome"] = ev["shot.outcome.name"]
    elif "shot_outcome_name" in ev.columns:
        ev["shot_outcome"] = ev["shot_outcome_name"]
    elif "shot" in ev.columns:
        def out_name(s):
            if isinstance(s, dict):
                o = s.get("outcome")
                if isinstance(o, dict):
                    return o.get("name")
            return np.nan
        ev["shot_outcome"] = ev["shot"].apply(out_name)
    else:
        ev["shot_outcome"] = np.nan

    return ev


def compute_luck_from_events(events: pd.DataFrame, low_xg_th: float = 0.05):
    ev = standardize_events(events)

    shots = ev[ev["type_name"] == "Shot"].copy()
    if shots.empty:
        return None

    shots = shots.dropna(subset=["team_name"])
    if shots.empty:
        return None

    shots["xg"] = pd.to_numeric(shots["shot_xg"], errors="coerce").fillna(0.0)
    shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)

    xg = shots.groupby("team_name")["xg"].sum().sort_values(ascending=False)
    goals = shots.groupby("team_name")["is_goal"].sum()

    if len(xg.index) < 2:
        return None

    # IMPORTANT: on prend les 2 équipes principales
    t1, t2 = xg.index[0], xg.index[1]

    xg1, xg2 = float(xg[t1]), float(xg[t2])
    g1, g2 = int(goals.get(t1, 0)), int(goals.get(t2, 0))

    fv1, fv2 = g1 - xg1, g2 - xg2
    swing = abs(fv1 - fv2)

    denom = (xg1 + xg2) + swing
    luck_match = 0.0 if denom <= 0 else (swing / denom) * 100

    dominance_t1 = 0.5 if (xg1 + xg2) <= 0 else xg1 / (xg1 + xg2)

    low_xg_goal_count = int(((shots["is_goal"] == 1) & (shots["xg"] < low_xg_th)).sum())

    return {
        "team1": t1, "team2": t2,
        "goals1": g1, "goals2": g2,
        "xg1": xg1, "xg2": xg2,
        "dominance_team1_xgshare": dominance_t1,
        "fv1": fv1, "fv2": fv2,
        "luck_match_pct": luck_match,
        "low_xg_goal_count": low_xg_goal_count,
    }


def collect_all_matches_since(start_date: str) -> pd.DataFrame:
    comps = sb.competitions()
    all_matches = []

    for _, c in comps.iterrows():
        comp_id = int(c["competition_id"])
        season_id = int(c["season_id"])

        try:
            m = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception:
            continue

        date_col = "match_date" if "match_date" in m.columns else ("kick_off" if "kick_off" in m.columns else None)
        if date_col is None:
            continue

        m[date_col] = pd.to_datetime(m[date_col], errors="coerce")
        m = m.dropna(subset=[date_col])
        m = m[m[date_col] >= pd.to_datetime(start_date)]
        if m.empty:
            continue

        m = m.rename(columns={date_col: "match_date"})
        m["competition_id"] = comp_id
        m["season_id"] = season_id

        keep = [c for c in ["match_id", "match_date", "home_team", "away_team", "home_score", "away_score", "competition_id", "season_id"] if c in m.columns]
        all_matches.append(m[keep])

    if not all_matches:
        return pd.DataFrame()

    out = pd.concat(all_matches, ignore_index=True)
    out = out.drop_duplicates(subset=["match_id"]).sort_values("match_date").reset_index(drop=True)
    return out


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    matches = collect_all_matches_since(START_DATE)
    if matches.empty:
        print(f"No open-data matches found since {START_DATE}")
        return

    # sample plus large pour survivre aux fails
    n_sample = min(max(N_MATCHES_TARGET * 2, N_MATCHES_TARGET), len(matches))
    sample = matches.sample(n=n_sample, random_state=RANDOM_SEED).reset_index(drop=True)

    rows = []
    failed = 0
    shown_fail = 0

    # DEBUG: test 1 match pour vérifier qu'on voit "Shot"
    if DEBUG_FIRST_MATCH:
        test_id = int(sample.iloc[0]["match_id"])
        ev = get_events_with_retry(test_id)
        sev = standardize_events(ev)
        print("DEBUG first match_id =", test_id)
        print("Columns (first 20):", list(ev.columns)[:20])
        print("Top event types:", sev["type_name"].value_counts().head(10).to_string())
        print("Number of shots detected:", int((sev["type_name"] == "Shot").sum()))
        print("-" * 60)

    for i, row in sample.iterrows():
        match_id = int(row["match_id"])

        try:
            events = get_events_with_retry(match_id)
            r = compute_luck_from_events(events, LOW_XG_TH)

            if r is None:
                failed += 1
            else:
                r["match_id"] = match_id
                r["match_date"] = row.get("match_date")
                r["competition_id"] = row.get("competition_id")
                r["season_id"] = row.get("season_id")
                rows.append(r)

        except Exception as e:
            failed += 1
            if shown_fail < 8:
                shown_fail += 1
                print(f"[FAIL] match_id={match_id} -> {type(e).__name__}: {e}")

        ok = len(rows)
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(sample)} | ok={ok} | failed={failed}")

        if ok >= N_MATCHES_TARGET:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n0 match exploitable. Regarde la section DEBUG au début :")
        print("- Si 'Number of shots detected' = 0, on a encore un problème de parsing (tu colles la sortie et je corrige).")
        print("- Si 'Number of shots detected' > 0, alors c’est le filtrage des équipes / données, et je t’ajuste ça.")
        return

    df.to_csv("luck_rates_2020_today.csv", index=False)

    print("\n=== RESULTS ===")
    print(f"Matches used: {len(df)}")
    print(f"Mean LuckMatch%: {df['luck_match_pct'].mean():.2f}")
    print("Quantiles LuckMatch%:")
    print(df["luck_match_pct"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
    print("\nSaved: luck_rates_2020_today.csv")


if __name__ == "__main__":
    main()




