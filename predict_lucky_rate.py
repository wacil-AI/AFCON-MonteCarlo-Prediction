import pandas as pd
import numpy as np
import glob
import os

# ============================================================
# 0) CONFIG LUCK (Option A: L dépend du gap)
# ============================================================
LUCK_FILE = "luck_rates_2020_today.csv"  
LUCK_MEAN = 0.273  

# Plus GAP_SHRINK est grand => plus on "écrase" la chance quand le gap est grand
GAP_SHRINK = 0.8

# bornes de sécurité sur L_eff
L_EFF_MIN = 0.03
L_EFF_MAX = 0.60


def load_luck_pool(csv_path: str):
    """
    Charge la colonne luck_match_pct.
    Retourne un numpy array de L en [0,1].
    """
    if not os.path.exists(csv_path):
        print(f"[Luck] Fichier '{csv_path}' introuvable -> fallback sur LUCK_MEAN={LUCK_MEAN:.3f}")
        return None

    # essai sep=',', puis sep=';'
    for sep in [",", ";"]:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            if "luck_match_pct" in df.columns:
                pool = (df["luck_match_pct"].dropna().clip(0, 100).values) / 100.0
                pool = pool[np.isfinite(pool)]
                if len(pool) > 10:
                    print(f"[Luck] Chargé {len(pool)} valeurs de chance depuis '{csv_path}' (sep='{sep}')")
                    return pool
        except Exception:
            continue

    print(f"[Luck] Impossible de lire 'luck_match_pct' dans '{csv_path}' -> fallback LUCK_MEAN={LUCK_MEAN:.3f}")
    return None


luck_pool = load_luck_pool(LUCK_FILE)


def sample_luck():
    """Tire une chance brute L en [0,1]."""
    if luck_pool is None or len(luck_pool) == 0:
        return float(LUCK_MEAN)
    return float(np.random.choice(luck_pool))


def shrink_luck_with_gap(L: float, diff_abs: float):
    """
    plus le gap (|diff|) est grand, moins le match est chaotique.
    shrink = 1/(1 + GAP_SHRINK * |diff|)
    """
    shrink = 1.0 / (1.0 + GAP_SHRINK * diff_abs)
    L_eff = L * shrink
    return float(np.clip(L_eff, L_EFF_MIN, L_EFF_MAX))


# ============================================================
# 1) CONFIGURATION & MAPPING DES NOMS
# ============================================================
name_map = {
    # DR Congo
    "Democratic Republic of the Congo": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "RD Congo": "DR Congo",
    "DRC": "DR Congo",

    # Côte d'Ivoire
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Cote D'Ivoire": "Côte d'Ivoire",
    "Côte d’Ivoire": "Côte d'Ivoire",

    # Traductions FR -> EN 
    "Afrique du Sud": "South Africa",
    "AFRIQUE DU SUD": "South Africa",
    "Afrique DS": "South Africa",
    "Cameroun": "Cameroon",
    "Maroc": "Morocco",
    "Algérie": "Algeria",
    "Sénégal": "Senegal",
    "Égypte": "Egypt",
    "Tunisie": "Tunisia",
    "Soudan": "Sudan",
    "Tanzanie": "Tanzania",
}

def clean_name(name: str) -> str:
    n = str(name).strip()
    return name_map.get(n, n)

# ============================================================
# 2) CHARGEMENT (Excel & CSV)
# ============================================================
def load_any_file(prefix: str) -> pd.DataFrame:
    files = glob.glob(f"{prefix}*")
    if not files:
        raise FileNotFoundError(f"Fichier commençant par '{prefix}' introuvable.")
    file_path = files[0]
    print(f"Chargement de : {file_path}")

    if file_path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)

    for encoding in ["utf-8", "latin1", "cp1252", "utf-8-sig"]:
        try:
            return pd.read_csv(file_path, sep=None, engine="python", encoding=encoding)
        except Exception:
            continue

    raise Exception(f"Impossible de lire le fichier {file_path}")

# ============================================================
# 3) WIN RATE (HISTORIQUE RÉCENT)
# ============================================================
def calculate_win_rate(team: str, df: pd.DataFrame) -> float:
    try:
        h_col, a_col = df.columns[1], df.columns[2]
        hs_col, as_col = df.columns[3], df.columns[4]

        matches = df[(df[h_col] == team) | (df[a_col] == team)]
        if len(matches) == 0:
            return 0.5

        wins = 0.0
        for _, row in matches.iterrows():
            if row[h_col] == team:
                if row[hs_col] > row[as_col]:
                    wins += 1.0
                elif row[hs_col] == row[as_col]:
                    wins += 0.5
            else:
                if row[as_col] > row[hs_col]:
                    wins += 1.0
                elif row[as_col] == row[hs_col]:
                    wins += 0.5

        return float(wins / len(matches))
    except Exception:
        return 0.5

# ============================================================
# 4) OUTILS
# ============================================================
def safe_minmax_scale(s: pd.Series) -> pd.Series:
    mn, mx = float(s.min()), float(s.max())
    if np.isclose(mx, mn):
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)

def get_stat(df: pd.DataFrame, team: str, default: float) -> float:
    try:
        row = df[df.iloc[:, 0] == team]
        if len(row) == 0:
            return float(default)
        return float(row.iloc[0, 1])
    except Exception:
        return float(default)

# ============================================================
# 5) EXÉCUTION DU MODÈLE + MONTE CARLO
# ============================================================
try:
    # --- Chargement fichiers ---
    df_history = load_any_file("1")
    df_value   = load_any_file("2")
    df_elo     = load_any_file("3")
    df_fifa    = load_any_file("4")

    # --- Nettoyage noms ---
    df_elo.iloc[:, 0]   = df_elo.iloc[:, 0].apply(clean_name)
    df_fifa.iloc[:, 0]  = df_fifa.iloc[:, 0].apply(clean_name)
    df_value.iloc[:, 0] = df_value.iloc[:, 0].apply(clean_name)
    df_history.iloc[:, 1] = df_history.iloc[:, 1].apply(clean_name)
    df_history.iloc[:, 2] = df_history.iloc[:, 2].apply(clean_name)

    # --- Liste R16 ---
    teams_r16 = [
        "Mali", "Tunisia",
        "Senegal", "Sudan",
        "Egypt", "Benin",
        "Côte d'Ivoire", "Burkina Faso",
        "South Africa", "Cameroon",
        "Morocco", "Tanzania",
        "Algeria", "DR Congo",
        "Nigeria", "Mozambique"
    ]
    teams_r16 = [clean_name(t) for t in teams_r16]

    # --- Construction stats équipe ---
    stats = []
    for t in teams_r16:
        elo  = get_stat(df_elo,  t, default=1400)
        fifa = get_stat(df_fifa, t, default=1200)
        val  = get_stat(df_value, t, default=10)
        wr   = calculate_win_rate(t, df_history)
        stats.append({"Team": t, "ELO": elo, "FIFA": fifa, "Value": val, "WinRate": wr})

    df_stats = pd.DataFrame(stats)

    # --- Normalisation robuste ---
    for c in ["ELO", "FIFA", "Value", "WinRate"]:
        df_stats[f"S_{c}"] = safe_minmax_scale(df_stats[c])

    # --- Power Score ---
    df_stats["PowerScore_raw"] = (
        df_stats["S_ELO"] * 0.38 +
        df_stats["S_FIFA"] * 0.17 +
        df_stats["S_Value"] * 0.23 +
        df_stats["S_WinRate"] * 0.22
    )

    df_stats["PowerScore"] = 0.05 + 0.95 * df_stats["PowerScore_raw"]
    power_dict = df_stats.set_index("Team")["PowerScore"].to_dict()

    # ============================================================
    # 6) MATCH ENGINE (Poisson + Luck conditionnée par gap)
    # ============================================================
    BASE_GOALS   = 1.15
    K_STRENGTH   = 0.35
    SIGMA_DIFF   = 0.25
    SIGMA_LAMBDA = 0.12
    LAMBDA_MIN   = 0.20
    LAMBDA_MAX   = 3.50
    EPS_POWER    = 1e-6
    PEN_TEMP     = 1.00
    SIGMA_PEN    = 0.35

    # Calibration: on fait dépendre les paramètres de L_eff
    SIGMA_DIFF_MEAN   = SIGMA_DIFF
    SIGMA_LAMBDA_MEAN = SIGMA_LAMBDA
    K_STRENGTH_MEAN   = K_STRENGTH

    def sigma_diff_from_luck(L_eff: float) -> float:
        a = 0.10
        b = (SIGMA_DIFF_MEAN - a) / LUCK_MEAN
        return float(a + b * L_eff)

    def sigma_lambda_from_luck(L_eff: float) -> float:
        a = 0.05
        b = (SIGMA_LAMBDA_MEAN - a) / LUCK_MEAN
        return float(a + b * L_eff)

    def k_strength_from_luck(L_eff: float) -> float:
        # L_eff grand => gap compte moins
        return float(K_STRENGTH_MEAN * (1.0 - 0.50 * L_eff))

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def simulate_match(t1: str, t2: str) -> str:
        p1 = float(power_dict.get(t1, 0.50))
        p2 = float(power_dict.get(t2, 0.50))
        p1 = max(p1, EPS_POWER)
        p2 = max(p2, EPS_POWER)

        # diff en log (stable)
        diff = np.log(p1) - np.log(p2)
        diff_abs = abs(diff)

        # ---- L dépend du gap ----
        L = sample_luck()                       # chance brute (StatsBomb)
        L_eff = shrink_luck_with_gap(L, diff_abs)  # moins de chaos si gros gap

        # paramètres match
        sigma_diff   = sigma_diff_from_luck(L_eff)
        sigma_lambda = sigma_lambda_from_luck(L_eff)
        k_strength   = k_strength_from_luck(L_eff)

        # bruit sur diff
        diff_noisy = diff + np.random.normal(0.0, sigma_diff)

        # lambdas
        lam1 = BASE_GOALS * np.exp(k_strength * diff_noisy)
        lam2 = BASE_GOALS * np.exp(-k_strength * diff_noisy)

        # variance multiplicative (luck-driven)
        lam1 *= np.random.lognormal(mean=0.0, sigma=sigma_lambda)
        lam2 *= np.random.lognormal(mean=0.0, sigma=sigma_lambda)

        lam1 = float(np.clip(lam1, LAMBDA_MIN, LAMBDA_MAX))
        lam2 = float(np.clip(lam2, LAMBDA_MIN, LAMBDA_MAX))

        g1 = np.random.poisson(lam1)
        g2 = np.random.poisson(lam2)

        if g1 > g2:
            return t1
        if g2 > g1:
            return t2

        # Egalité -> TAB : un peu plus aléatoire si L_eff est grand
        pen_diff = diff + np.random.normal(0.0, SIGMA_PEN * (1.0 + 0.6 * L_eff))
        p_pen = sigmoid(pen_diff / PEN_TEMP)
        return t1 if np.random.random() < p_pen else t2

    # ============================================================
    # 7) BRACKET OFFICIEL
    # ============================================================
    bracket_r16 = [
        ("Mali", "Tunisia"),
        ("Senegal", "Sudan"),
        ("Egypt", "Benin"),
        ("Côte d'Ivoire", "Burkina Faso"),
        ("South Africa", "Cameroon"),
        ("Morocco", "Tanzania"),
        ("Algeria", "DR Congo"),
        ("Nigeria", "Mozambique"),
    ]
    bracket_r16 = [(clean_name(a), clean_name(b)) for a, b in bracket_r16]

    # ============================================================
    # 8) SIMULATION MONTE CARLO
    # ============================================================
    n_sims = 10_000
    print(f"\nSimulation de {n_sims} tournois à partir des 8èmes (02/01/2026)...")

    stage_counts = {t: {"WIN_R16": 0, "WIN_QF": 0, "WIN_SF": 0, "WIN_F": 0} for t in teams_r16}

    for _ in range(n_sims):
        # R16
        r16_winners = []
        for a, b in bracket_r16:
            w = simulate_match(a, b)
            r16_winners.append(w)
            stage_counts[w]["WIN_R16"] += 1

        # QF
        qf_pairs = [
            (r16_winners[0], r16_winners[1]),
            (r16_winners[2], r16_winners[3]),
            (r16_winners[4], r16_winners[5]),
            (r16_winners[6], r16_winners[7]),
        ]
        qf_winners = []
        for a, b in qf_pairs:
            w = simulate_match(a, b)
            qf_winners.append(w)
            stage_counts[w]["WIN_QF"] += 1

        # SF
        sf_pairs = [
            (qf_winners[0], qf_winners[1]),
            (qf_winners[2], qf_winners[3]),
        ]
        sf_winners = []
        for a, b in sf_pairs:
            w = simulate_match(a, b)
            sf_winners.append(w)
            stage_counts[w]["WIN_SF"] += 1

        # Final
        champion = simulate_match(sf_winners[0], sf_winners[1])
        stage_counts[champion]["WIN_F"] += 1

    # ============================================================
    # 9) TABLEAU FINAL
    # ============================================================
    report_data = []
    for team in teams_r16:
        s = stage_counts[team]
        report_data.append({
            "Pays": team,
            "Power Score": round(float(power_dict[team]), 3),
            "% gagner le 1/8": 100.0 * s["WIN_R16"] / n_sims,
            "% gagner le 1/4": 100.0 * s["WIN_QF"] / n_sims,
            "% gagner le 1/2": 100.0 * s["WIN_SF"] / n_sims,
            "% gagner la Finale": 100.0 * s["WIN_F"] / n_sims,
        })

    df_report = pd.DataFrame(report_data).sort_values("Power Score", ascending=False)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 140)

    print("\n" + "=" * 110)
    print("TABLEAU FINAL - CAN 2025 (Calcul du 02/01/2026) - Monte Carlo + Poisson + Luck(gap-conditioned)")
    print("=" * 110)
    print(df_report.to_string(index=False, formatters={
        "% gagner le 1/8": lambda x: f"{x:6.2f}%",
        "% gagner le 1/4": lambda x: f"{x:6.2f}%",
        "% gagner le 1/2": lambda x: f"{x:6.2f}%",
        "% gagner la Finale": lambda x: f"{x:6.2f}%",
    }))
    print("=" * 110)

except Exception as e:
    print(f"Erreur fatale : {e}")

