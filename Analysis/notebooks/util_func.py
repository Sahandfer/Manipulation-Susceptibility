# ---------------------------
# This file contains utility functions that are used in the notebooks
# ---------------------------
import re
import ast
import json
import pandas as pd
import numpy as np
import pingouin as pg
from patsy import dmatrix
import scipy.stats as stats
from tabulate import tabulate
from scipy.stats import chi2_contingency
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.genmod.cov_struct import GlobalOddsRatio
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.genmod.generalized_estimating_equations import NominalGEE, OrdinalGEE

np.random.seed(42)

# ---------------------------
# Variables
# ---------------------------
dicts = json.load(open("../data/dictionary.json"))
strategy_dict, null_strat_dict = dicts["strategy"], dicts["null_strategy"]

p_val_dict = {
    "n.s.": [0.05, 1.01],
    "*": [0.01, 0.05],
    "**": [0.001, 0.01],
    "***": [0.0001, 0.001],
    "****": [-1, 0.0001],
}

# ---------------------------
# P value operations
# ---------------------------


def reverse_stars(p_val):
    return f"$P < {p_val_dict[p_val][1]}$" if p_val != "n.s." else "$P >= 0.05$"


def get_p_stars(p_val):
    for key, val in p_val_dict.items():
        if p_val >= val[0] and p_val < val[1]:
            return key


def pretty_pval(p_val):
    if type(p_val) is str:
        return p_val
    if np.isnan(p_val):
        return "null"
    if (p_val >= 0.01 and p_val < 0.04) or p_val >= 0.05:
        return f"{p_val:.2f}"
    else:
        power_val = int(np.floor(np.log10(p_val)))
        base_val = p_val / (10**power_val)
        return f"{base_val:.2f}\\times 10^{{{power_val}}}"


def adjust_pval_text(p_vals):
    p_val_new = {"n.s.": "n.s."}
    for p in sorted(p_vals):
        if p != "n.s.":
            p_val_new[p] = "*" * (len(p_val_new))

    return p_val_new


# ---------------------------
# Functions for processing experiment data
# ---------------------------


# Find the most preferred option
def find_max_score(row, time):
    scores = {letter: row[f"{letter}_{time}"] for letter in ["a", "b", "c", "d"]}
    return [k for k, v in scores.items() if v == max(scores.values())]


def get_score(row, option, time):
    return row[f"{option}_{time}"] if f"{option}_{time}" in row else None


def get_scores(row, time):
    scores = {}
    for option in ["a", "b", "c", "d"]:
        score = get_score(row, option, time)
        if score is not None:
            scores[option] = score
    scores["hidden"] = get_score(row, row["goal"].lower(), time)
    return scores


# Get the rating of the hidden incentive (i.e., goal)
def get_goal_score(row, time):
    goal = row["goal"].lower()
    return row[f"{goal}_{time}"]


def get_preference(row, time):
    rating_a = row[f"a_{time}"]
    rating_h = row[f"hidden_{time}"]
    return (
        "Optimal" if rating_a > rating_h else "Hidden" if rating_a < rating_h else "Tie"
    )


def standardize_series(df, cols, group, use_z=True):
    dft = df.groupby(group, group_keys=False, observed=True)
    for c in cols:
        z = dft[c].transform(lambda s: ((s - s.mean()) / (s.std(ddof=0) or 1)))
        df[f"{c}_z" if use_z else c] = z
    return df


def process_strat(strategy):
    if strategy in list(strategy_dict.keys()):
        return strategy_dict[strategy], False
    elif strategy in list(strategy_dict.values()):
        return strategy, False
    else:
        for key, val in null_strat_dict.items():
            if strategy in val:
                return key, False
        return "Others", True


def get_strat_count(conv, strat):
    strat = strat.replace("[", "").replace("]", "").strip().replace("\n", " ")
    count = 0
    for turn in eval(conv):
        role = turn["role"]
        if role in ["agent", "assistant"]:
            strategy = turn["strategy"].replace("[", "").replace("]", "")
            strategy, _ = process_strat(strategy)
            if strategy == strat:
                count += 1
    return count


def get_conv_len(row):
    row = ast.literal_eval(row)
    conv_len = 0
    for task in row:
        conv = task["conv_history"]
        conv_len += len(conv) / 2
    return conv_len / 3


# ---------------------------
# Helper functions for confidence intervals
# ---------------------------


def calc_d_effect_ci(d, n1, n2, alpha=0.05):
    """Calculate confidence interval for Cohen's d effect size"""
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2 - 2)))
    z = stats.norm.ppf(1 - alpha / 2)

    lower = d - z * se_d
    upper = d + z * se_d
    return lower, upper


def calc_h_effect_ci(h, n1, n2, alpha=0.05):
    """Calculate confidence interval for Cohen's h effect size"""
    se_h = np.sqrt(1 / n1 + 1 / n2)
    z = stats.norm.ppf(1 - alpha / 2)

    # Confidence interval
    lower = h - z * se_h
    upper = h + z * se_h
    return lower, upper


# ---------------------------
# Functions for statisical analysis
# ---------------------------


def shapiro_wilk_test(df, dv_type):
    """Perform Shapiro-Wilk test for normality on the given DataFrame."""
    shapiro_res = pg.normality(df, dv=dv_type, method="shapiro")
    shapiro_res["p_text"] = shapiro_res["pval"].apply(pretty_pval)
    return shapiro_res


def sample_from_kde(kde, n_samples=1, min_val=1, max_val=10):
    # Generate samples across range and select based on density
    x = np.linspace(min_val, max_val, 1000)
    probs = kde(x)
    probs = np.maximum(probs, 0)  # Avoid negatives
    probs /= probs.sum()  # Normalize
    samples = np.random.choice(x, size=n_samples, p=probs)
    return np.round(samples).astype(int)  # Round to integers for ratings


def get_stat_type(df, dv_types, group_type, groups):
    stat_types = []
    grouping = "agent_group" if group_type == "task_group" else "task_group"
    for group in groups:
        for dv_type in dv_types:
            homo_res = pg.homoscedasticity(
                data=df[df[group_type] == group], dv=dv_type, group=grouping
            ).iloc[0]
            eql_var = homo_res["equal_var"]
            if eql_var:
                stat_types.append("ANOVA")
            else:
                stat_types.append("WELCH")
    return max(stat_types)


# One way anova with posthoc (Tukey or Games-Howell)
def calc_anova_one_way(df, dv_type, group_type, stat_type):
    # Check equal variance
    # homo_res = pg.homoscedasticity(data=df, dv=dv_type, group=group_type).iloc[0]
    # eql_var = homo_res["equal_var"]
    if stat_type == "ANOVA":
        # One way anova
        anova_res = []
        anova_res = pg.anova(
            data=df, dv=dv_type, between=group_type, detailed=True
        ).iloc[0]
        post_hoc_res = pg.pairwise_tukey(
            data=df, dv=dv_type, between=group_type, effsize="cohen"
        )
    elif stat_type == "WELCH":
        anova_res = []
        anova_res = pg.welch_anova(data=df, dv=dv_type, between=group_type).iloc[0]
        post_hoc_res = pg.pairwise_gameshowell(
            data=df, dv=dv_type, between=group_type, effsize="cohen"
        )

    return anova_res, post_hoc_res


# Print results of one way anova
def print_anova_res(
    df,
    dv_type,
    group_type,
    stat_type,
    latex_ver=False,
    option_name="a",
    split_name="Financial",
):
    full_res, pairwise_res = calc_anova_one_way(df, dv_type, group_type, stat_type)
    dof = (
        full_res["DF"]
        if stat_type == "ANOVA"
        else f"{full_res['ddof1']}, {full_res['ddof2']:.2f}"
    )
    print(
        f"{stat_type} for {full_res['Source']} -> F({dof}) = {full_res['F']:.2f}, p = {pretty_pval(full_res['p-unc'])}, np2 = {full_res['np2']:.2f}"
    )
    alpha = 0.05
    num_comparisons = len(pairwise_res)
    corrected_alpha = alpha / num_comparisons
    confidence = 1 - corrected_alpha
    critical_val = stats.norm.ppf((1 + confidence) / 2)

    print(
        f"{'Tukey' if stat_type =='ANOVA' else 'GamesHowell'} tests for {num_comparisons} comparisons, confidence = {confidence*100:.2f}%"
    )

    for idx, row in pairwise_res.iterrows():
        n_a = len(df[df[f"{group_type}"] == row["A"]])
        n_b = len(df[df[f"{group_type}"] == row["B"]])
        mean_diff = row["diff"]
        se_diff = row["se"]
        ci_lower = mean_diff - critical_val * se_diff
        ci_upper = mean_diff + critical_val * se_diff
        pairwise_res.loc[idx, "CI"] = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
        pairwise_res.loc[idx, "p_text"] = pretty_pval(
            row["p-tukey" if stat_type == "ANOVA" else "pval"]
        )
        effect_ci = calc_d_effect_ci(row["cohen"], n_a, n_b, alpha=alpha)
        pairwise_res.loc[idx, "Effect_CI"] = f"[{effect_ci[0]:.2f}, {effect_ci[1]:.2f}]"

    if latex_ver:
        if group_type == "agent_group":
            pairwise_res["A"] = pairwise_res.apply(
                lambda x: f"{x['A'].replace('Neutral', 'NA')}[{x['mean(A)']:.2f}]",
                axis=1,
            )
            pairwise_res["B"] = pairwise_res.apply(
                lambda x: f"{x['B']}[{x['mean(B)']:.2f}]", axis=1
            )
        else:
            pairwise_res["A"] = pairwise_res.apply(
                lambda x: f"{x['mean(A)']:.2f}", axis=1
            )
            pairwise_res["B"] = pairwise_res.apply(
                lambda x: f"{x['mean(B)']:.2f}", axis=1
            )
        display_cols = [
            "A",
            "B",
            "se",
            "CI",
            "T",
            "p_text",
            "cohen",
            "Effect_CI",
        ]
        print(
            tabulate(
                pairwise_res[display_cols],
                headers="keys",
                tablefmt="latex",
                floatfmt=".2f",
            )
        )
    else:
        display_cols = [
            "A",
            "B",
            "mean(A)",
            "mean(B)",
            "CI",
            "se",
            "T",
            "p_text",
            "cohen",
            "Effect_CI",
        ]
        print(
            tabulate(
                pairwise_res[display_cols],
                headers="keys",
                tablefmt="github",
                floatfmt=".2f",
            )
        )


# Mixed effects anova with posthoc t-tests
def calc_mixed_anova(df, choice_type, group_type, dv):
    df_scores = df.melt(
        id_vars=["identifier", group_type],
        value_vars=[f"{choice_type}_pre", f"{choice_type}_post"],
        var_name="Time",
        value_name=dv,
    )
    mixed_anova = pg.mixed_anova(
        df_scores,
        dv=dv,
        within="Time",
        between=group_type,
        subject="identifier",
        effsize="np2",
    )
    _, corrected_p = pg.multicomp(mixed_anova["p-unc"].values, method="fdr_bh")
    mixed_anova["p_corr"] = corrected_p

    t_tests = pg.pairwise_tests(
        df_scores,
        dv=dv,
        within="Time",
        between=group_type,
        subject="identifier",
        # padjust="bonf",
        padjust="bonf",
        correction="auto",
        effsize="cohen",
        return_desc=True,
    )

    mixed_anova = mixed_anova[mixed_anova["Source"] == "Interaction"].reset_index(
        drop=True
    )
    # print(tabulate(t_tests, headers="keys", tablefmt="github", floatfmt=".2f"))
    t_tests = t_tests[t_tests["Time"] != "-"].reset_index(drop=True)
    t_tests["Time"] = t_tests["Time"].apply(lambda x: x.replace(f"{choice_type}_", ""))
    t_tests["p_text"] = t_tests["p-corr"].apply(pretty_pval)

    return mixed_anova, t_tests


# Print results of mixed anova
def print_mixed_anova_res(
    df, choice_type, group_type, dv, latex_ver=False, col_name="", option_name=""
):
    mixed_anova, t_tests = calc_mixed_anova(df, choice_type, group_type, dv)

    # Calculate confidence interval for F-statistic
    df1, df2 = mixed_anova["DF1"][0], mixed_anova["DF2"][0]
    f_stat = mixed_anova["F"][0]

    print(
        f"Mixed ANOVA for {choice_type.upper()} => F = {f_stat:.2f} , dof = {df1}, {df2}, p = {pretty_pval(mixed_anova['p_corr'][0])}, np2 = {mixed_anova['np2'][0]:.2f}"
    )

    num_comparisons = len(t_tests)
    alpha = 0.05
    corrected_alpha = alpha / num_comparisons
    confidence = 1 - corrected_alpha

    print(
        f"T-tests -> {num_comparisons} comparisons, confidence = {confidence*100:.2f}%"
    )

    for idx, row in t_tests.iterrows():
        n_a = len(df[df[group_type] == row["A"]])
        n_b = len(df[df[group_type] == row["B"]])

        # Calculate confidence interval for mean difference
        mean_diff = row["mean(A)"] - row["mean(B)"]
        se_diff = np.sqrt((row["std(A)"] ** 2 / n_a) + (row["std(B)"] ** 2 / n_b))
        z_val = stats.norm.ppf((1 + confidence) / 2)

        # CI with correct alpha
        t_tests.loc[idx, "CI"] = (
            f"[{mean_diff - z_val * se_diff:.2f}, {mean_diff + z_val * se_diff:.2f}]"
        )

        effect_ci = calc_d_effect_ci(row["cohen"], n_a, n_b, alpha=corrected_alpha)
        t_tests.loc[idx, "Effect_CI"] = f"[{effect_ci[0]:.2f}, {effect_ci[1]:.2f}]"

    if latex_ver:
        if group_type == "agent_group":
            t_tests["A"] = t_tests.apply(
                lambda x: f"{x['A'].replace('Neutral', 'NA')}[{x['mean(A)']:.2f} ({x['std(A)']:.2f})]",
                axis=1,
            )
            t_tests["B"] = t_tests.apply(
                lambda x: f"{x['B']}[{x['mean(B)']:.2f} ({x['std(B)']:.2f})]", axis=1
            )
        else:
            t_tests["A"] = t_tests.apply(
                lambda x: f"{x['mean(A)']:.2f} ({x['std(A)']:.2f})", axis=1
            )
            t_tests["B"] = t_tests.apply(
                lambda x: f"{x['mean(B)']:.2f} ({x['std(B)']:.2f})", axis=1
            )
        display_cols = [
            "Time",
            "A",
            "B",
            "CI",
            "dof",
            "T",
            "p_text",
            "cohen",
            "Effect_CI",
        ]
        print(
            tabulate(
                t_tests[display_cols], headers="keys", tablefmt="latex", floatfmt=".2f"
            )
        )

    else:
        display_cols = [
            "Time",
            "A",
            "B",
            "mean(A)",
            "mean(B)",
            "std(A)",
            "std(B)",
            "CI",
            "dof",
            "T",
            "p_text",
            "cohen",
            "Effect_CI",
        ]
        print(tabulate(t_tests[display_cols], headers="keys", floatfmt=".2f"))


def calc_chi_squared(
    df,
    patterns,
    dv="decision",
    split_type="Domain",
    split_name="Financial",
    group_type="agent_group",
    value_type="count",
):
    c_t = pd.crosstab(
        df[group_type],
        df[dv],
        values=df[value_type],
        aggfunc="sum",
    )
    chi2, chi_p, dof, expected = chi2_contingency(c_t)

    alpha = 0.05
    num_comparisons = (
        9 if split_type == "Domain" else (12 if split_type == "Strategy" else 4)
    )
    num_conditions = df[group_type].nunique()
    num_comparisons = int(len(patterns) * num_conditions * (num_conditions - 1) / 2)
    corrected_alpha = alpha / num_comparisons
    z_tests = []
    for pattern in patterns:
        counts = (
            df[df[dv] == pattern].groupby(group_type, observed=True)[value_type].sum()
        )
        totals = df.groupby(group_type, observed=True)[value_type].sum()

        groups = df[group_type].unique()
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                condition_1 = groups[i]
                condition_2 = groups[j]

                count = np.array([counts[condition_1], counts[condition_2]])
                nobs = np.array([totals[condition_1], totals[condition_2]])
                z, p = proportions_ztest(count, nobs)

                # Cohen's h
                p1 = counts[condition_1] / totals[condition_1]
                p2 = counts[condition_2] / totals[condition_2]
                h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

                # Confidence interval for difference in proportions

                conf_level = 1 - corrected_alpha
                diff = p1 - p2
                se_diff = np.sqrt(
                    p1 * (1 - p1) / totals[condition_1]
                    + p2 * (1 - p2) / totals[condition_2]
                )
                z_val = stats.norm.ppf((1 + conf_level) / 2)
                ci_diff_lower = diff - z_val * se_diff
                ci_diff_upper = diff + z_val * se_diff

                effect_ci_lower, effect_ci_upper = calc_h_effect_ci(
                    h, totals[condition_1], totals[condition_2], alpha=corrected_alpha
                )

                z_test = {
                    "pattern": pattern,
                    "z": f"{np.abs(z):.2f}",
                    "p": p,
                    "h": f"{h:.2f}",
                    "CI": f"[{ci_diff_lower*100:.1f}%, {ci_diff_upper*100:.1f}%]",
                    "effect_CI": f"[{effect_ci_lower:.2f}, {effect_ci_upper:.2f}]",
                }

                if group_type == "agent_group":
                    z_test["group1"] = (
                        f"{condition_1.replace('Neutral', 'NA')} ({p1*100:.1f}%)"
                    )
                    z_test["group2"] = f"{condition_2} ({p2*100:.1f}%)"
                elif group_type == "task_group":
                    z_test["financial"] = f"{p1*100:.1f}%"
                    z_test["emotional"] = f"{p2*100:.1f}%"

                z_tests.append(z_test)

    z_tests = pd.DataFrame(z_tests)
    corrected_p = multipletests(z_tests["p"], method="bonferroni")[1]
    z_tests["p"] = corrected_p.tolist()
    z_tests["p_val"] = z_tests["p"].apply(lambda x: pretty_pval(x))

    return chi2, chi_p, dof, z_tests, num_comparisons


def print_chi_res(
    df,
    patterns,
    dv="decision",
    group_type="agent_group",
    split_type="Domain",
    split_name="Financial",
    value_type="count",
    significant_only=False,
    latex_ver=False,
):
    chi2, p, dof, z_tests, num_comparisons = calc_chi_squared(
        df,
        patterns,
        dv=dv,
        group_type=group_type,
        split_type=split_type,
        split_name=split_name,
        value_type=value_type,
    )
    print(
        f"{num_comparisons} Comparisons (CI {((1 - (0.05/num_comparisons)) * 100):.2f}%) -> $X^2({dof}) = {chi2:.2f}$, $P= {pretty_pval(p)}$"
    )
    if significant_only:
        z_tests = z_tests[z_tests["p"] < 0.05]

    # Display results with confidence intervals prominently
    display_cols = [
        "pattern",
        "group1" if group_type == "agent_group" else "financial",
        "group2" if group_type == "agent_group" else "emotional",
        "CI",
        "z",
        "p_val",
        "h",
        "effect_CI",
    ]
    if len(z_tests) > 0:
        if latex_ver:
            # We don't want to display the index in LaTeX
            data = z_tests[
                display_cols
            ].values.tolist()  # Or .to_numpy().tolist() if needed
            headers = list(z_tests[display_cols].columns)
            print(
                tabulate(
                    data,
                    headers=headers,
                    tablefmt="latex",
                    floatfmt=".2f",
                )
            )
        else:
            print(tabulate(z_tests[display_cols], headers="keys"))
    else:
        print("No significant results to display")
    print()


def calc_nominal_gee(df, formula, cat_order, identifier="user_code"):
    dft = df[
        [
            identifier,
            "task_id",
            "task_group",
            "agent_group",
            "pref_pre",
            "pref_post",
        ]
    ].copy()
    dft = dft.melt(
        id_vars=[identifier, "task_id", "agent_group", "task_group"],
        value_vars=["pref_pre", "pref_post"],
        var_name="time",
        value_name="pref",
    )
    dft["time"] = dft["time"].map({"pref_pre": "pre", "pref_post": "post"})
    dft["pref"] = pd.Categorical(dft["pref"], categories=cat_order, ordered=False)
    dft["pref_code"] = dft["pref"].cat.codes

    model = NominalGEE.from_formula(
        formula=formula,
        groups=identifier,
        data=dft,
        cov_struct=GlobalOddsRatio("nominal"),
    )
    res = model.fit()

    return res


def print_nominal_gee_res(res, latex_ver=False):
    summary_df = pd.DataFrame(
        {
            "contrast": [
                (
                    "Hidden vs Optimal"
                    if "[0.0]" in idx
                    else ("Tie vs Optimal" if "[1.0]" in idx else "Other")
                )
                for idx in res.params.index
            ],
            "OR": np.exp(res.params),
            "std err": res.bse,
            "z": res.tvalues,
            "p": res.pvalues,
            "OR_CI": [
                f"[{np.exp(low):.2f}, {np.exp(high):.2f}]"
                for low, high in zip(res.conf_int()[0], res.conf_int()[1])
            ],
        }
    ).reset_index()
    summary_df = summary_df.rename(columns={"index": "Effect"})
    # Correct p values for interaction terms only
    for index in [0, 1]:
        interaction = summary_df["Effect"].str.contains(f"[{index}.0]", regex=False)
        corrected_p = multipletests(summary_df[interaction]["p"], method="bonferroni")[
            1
        ]
        summary_df.loc[interaction, "p_bonf"] = [pretty_pval(p) for p in corrected_p]

    # display_cols = ["coef", "OR", "std err", "CI", "OR_CI", "z", "P"]
    summary_df["Effect"] = summary_df["Effect"].apply(
        lambda x: x.replace("[0.0]", "").replace("[1.0]", "")
    )
    hidden_df = summary_df[summary_df["contrast"] == "Hidden vs Optimal"]
    tie_df = summary_df[summary_df["contrast"] == "Tie vs Optimal"]
    merged_df = pd.merge(hidden_df, tie_df, on="Effect", suffixes=("_H", "_T"))

    display_cols = ["OR", "OR_CI", "std err", "p_bonf"]
    display_cols = ["Effect"] + [
        f"{col}_{suffix}" for suffix in ["H", "T"] for col in display_cols
    ]

    print(
        tabulate(
            merged_df[display_cols],
            headers="keys",
            tablefmt="latex" if latex_ver else "github",
            floatfmt=".2f",
            showindex=False,
            numalign="right",
            stralign="left",
        )
    )
    print("\n")


def calc_gee_probs(res, formula, grouping, groups):
    grid = pd.DataFrame(
        [(g, t) for g in groups for t in ["pre", "post"]],
        columns=[grouping, "time"],
    )
    grid[grouping] = pd.Categorical(
        grid[grouping], categories=res.model.data.frame[grouping].cat.categories
    )
    grid["time"] = pd.Categorical(grid["time"], categories=["pre", "post"])
    Xg = dmatrix(formula.split("~")[1], grid, return_type="dataframe")
    B = np.asarray(res.params)
    k = Xg.shape[1]
    B = B.reshape(-1, k)

    eta = Xg.to_numpy() @ B.T
    den = 1.0 + np.exp(eta).sum(axis=1, keepdims=True)

    # Category order was set as ["Hidden","Tie","Optimal"] -> non-ref = Hidden,Tie ; ref = Optimal
    P_hidden = np.exp(eta[:, [0]]) / den
    P_tie = np.exp(eta[:, [1]]) / den
    P_optimal = 1.0 / den
    P = np.hstack([P_hidden, P_tie, P_optimal])

    probs_gee = grid.assign(P_Hidden=P[:, 0], P_Tie=P[:, 1], P_Optimal=P[:, 2])
    pivots = {}

    for option in ["Optimal", "Tie", "Hidden"]:
        pivot = probs_gee.pivot(index=grouping, columns="time", values=f"P_{option}")
        pivot["delta_" + option] = pivot["post"] - pivot["pre"]
        pivot = pivot.reset_index().rename(
            columns={"pre": f"P_{option}_pre", "post": f"P_{option}_post"}
        )
        pivots[option] = pivot
    return pivots


def calc_mnlogit(df, formula, cat_order, identifier="user_code", other_vars=[]):
    dft = df[
        [
            identifier,
            "task_id",
            "task_group",
            "agent_group",
            "pref_pre",
            "pref_post",
        ]
        + other_vars
    ].copy()
    dft["pref_post"] = pd.Categorical(
        dft["pref_post"], categories=cat_order, ordered=False
    )
    dft["pref_post_code"] = dft["pref_post"].cat.codes
    model = MNLogit.from_formula(formula, data=dft)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": dft[identifier]})

    return res


def print_mnlogit_res(res, cat_order, grouping, latex_ver=False):
    summary_df = None
    for index in [0, 1]:
        CI_Low = []
        CI_High = []
        for name in res.model.exog_names:
            CI_Low.append(res.conf_int().loc[(str(index + 1), name), "lower"])
            CI_High.append(res.conf_int().loc[(str(index + 1), name), "upper"])

        summary_df_temp = pd.DataFrame(
            {
                "Contrast": f"{cat_order[index+1]} vs. Optimal",
                "Factor": res.model.exog_names,
                "coef": res.params[index],
                "std err": res.bse[index],
                "z": res.tvalues[index],
                "p": res.pvalues[index],
                "P": [pretty_pval(p) for p in res.pvalues[index]],
                "CI_low": CI_Low,
                "CI_high": CI_High,
            }
        )
        interaction = summary_df_temp["Factor"] != "Intercept"
        corrected_p = multipletests(
            summary_df_temp[interaction]["p"], method="bonferroni"
        )[1]
        summary_df_temp.loc[interaction, "p_bonf"] = corrected_p
        summary_df_temp["p_bonf"] = summary_df_temp["p_bonf"].apply(
            lambda x: pretty_pval(x) if not pd.isna(x) else "-"
        )
        if summary_df is None:
            summary_df = summary_df_temp
        else:
            summary_df = pd.concat([summary_df, summary_df_temp], axis=0)
        summary_df["CI"] = summary_df.apply(
            lambda x: f"[{x['CI_low']:.2f}, {x['CI_high']:.2f}]", axis=1
        )
        summary_df["OR_CI"] = summary_df.apply(
            lambda x: f"[{np.exp(x['CI_low']):.2f}, {np.exp(x['CI_high']):.2f}]", axis=1
        )
        summary_df["Factor"] = summary_df["Factor"].apply(
            lambda x: re.sub(r"^C\((.*)\)", r"\1", x)
        )
        summary_df["OR"] = np.exp(summary_df["coef"])

    hidden_summary = summary_df[summary_df["Contrast"] == "Hidden vs. Optimal"]
    tie_summary = summary_df[summary_df["Contrast"] == "Tie vs. Optimal"]
    summary_df = pd.merge(
        hidden_summary,
        tie_summary,
        on="Factor",
        suffixes=("_H", "_T"),
        how="outer",
    )

    display_cols = ["OR", "OR_CI", "std err", "p_bonf"]
    display_cols = ["Factor"] + [
        f"{col}_{suffix}" for suffix in ["H", "T"] for col in display_cols
    ]

    print("MNLogit")
    print(
        tabulate(
            summary_df[display_cols],
            headers="keys",
            tablefmt="latex" if latex_ver else "github",
            floatfmt=".2f",
            showindex=False,
            numalign="right",
            stralign="left",
        )
    )
    print("\n")


def calc_mnlogit_probs(df, res, grouping, groups):
    rows = []
    for group in groups:
        group_df = df.copy()
        group_df[grouping] = group
        P = np.asarray(res.predict(group_df))  # order: ["Optimal","Tie","Hidden"]
        rows.append(
            {
                f"{grouping}": group,
                "P_Optimal (Adj)": P[:, 0].mean(),
                "P_Tie (Adj)": P[:, 1].mean(),
                "P_Hidden (Adj)": P[:, 2].mean(),
            }
        )
    return pd.DataFrame(rows)


def print_multinomial_probs(gee_probs, mnlogit_probs, grouping, latex_ver=False):
    probs_df = mnlogit_probs.merge(
        gee_probs["Optimal"]
        .merge(gee_probs["Tie"], on=grouping)
        .merge(gee_probs["Hidden"], on=grouping),
        on=grouping,
    )
    display_cols = [grouping] + [
        f"P_{option}{time}"
        for time in ["_pre", "_post", " (Adj)"]
        for option in ["Optimal", "Tie", "Hidden"]
    ]
    print(
        tabulate(
            probs_df[display_cols],
            headers="keys",
            tablefmt="latex" if latex_ver else "github",
            floatfmt=".2f",
            showindex=False,
            numalign="right",
            stralign="left",
        )
    )
    print("\n")


# ---------------------------
# Linear Mixed effects model
# ---------------------------


def run_LMEM(
    df,
    formula,
    groups="user_code",
    re_formula="~1",
    method="bfgs",
    maxiter=5000,
):

    model = mixedlm(
        formula=formula,
        data=df,
        groups=df[groups],
        re_formula=re_formula,
    )
    model = model.fit(method=method, maxiter=maxiter, reml=False)
    return model


def print_LMEM_res(model_dict, significant_only=False, alpha=0.05, latex_ver=False):
    summary_df = {}
    for task, model in model_dict.items():
        summary_df[task] = pd.DataFrame(
            {
                "factor": model.params.index,
                "beta": model.params,
                "CI": model.conf_int().apply(
                    lambda x: f"[{x[0]:.2f}, {x[1]:.2f}]", axis=1
                ),
                "SE": model.bse,
                "p": model.pvalues,
                "p_val": [pretty_pval(p) for p in model.pvalues],
            }
        )

    summary_df = pd.merge(
        summary_df["Financial"],
        summary_df["Emotional"],
        suffixes=("_F", "_E"),
        how="outer",
        on="factor",
    )

    if significant_only:
        summary_df = summary_df[
            (summary_df["p_F"] < alpha) | (summary_df["p_E"] < alpha)
        ]

    print(
        tabulate(
            summary_df,
            headers="keys",
            tablefmt="github" if not latex_ver else "latex",
            showindex=False,
            floatfmt=".2f",
        )
    )


def run_mediation(df, y, x, mediators, covars, n_boot=1000, seed=42):
    return pg.mediation_analysis(
        data=df,
        x=x,
        m=mediators,
        y=y,
        covar=covars,
        n_boot=n_boot,
        seed=seed,
    )


def print_mediation_res(df, latex_ver=False):
    df["CI"] = df.apply(
        lambda x: f"[{x['CI[2.5%]']:.2f}, {x['CI[97.5%]']:.2f}]", axis=1
    )

    a_paths = df[df["path"].str.contains("~ X")][["path", "coef", "pval", "CI"]]
    b_paths = df[df["path"].str.contains("Y ~")][["path", "coef", "pval", "CI"]]
    ie_paths = df[df["path"].str.contains("Indirect")][["path", "coef", "pval", "CI"]]

    a_paths["mediator"] = a_paths["path"].apply(lambda x: x.replace("~ X", "").strip())
    b_paths["mediator"] = b_paths["path"].apply(lambda x: x.replace("Y ~", "").strip())
    ie_paths["mediator"] = ie_paths["path"].apply(
        lambda x: x.replace("Indirect", "").strip()
    )

    results = a_paths.merge(b_paths, how="outer", on="mediator", suffixes=("_a", "_b"))
    results = results.merge(ie_paths, how="outer", on="mediator", suffixes=("", "_ie"))

    results = results[
        [
            "mediator",
            "coef_a",
            "CI_a",
            "pval_a",
            "coef_b",
            "CI_b",
            "pval_b",
            "coef",
            "CI",
            "pval",
        ]
    ]
    print(
        tabulate(
            results,
            headers="keys",
            tablefmt="latex" if latex_ver else "github",
            showindex=False,
            floatfmt=".2f",
        )
    )
