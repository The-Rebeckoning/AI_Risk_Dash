import pandas as pd

from .dataframes import (
    industry_counts_monthly_display_df,
    industry_counts_monthly_long_df,
    industry_counts_yearly_display_df,
    reported_case_totals_monthly_df,
    stakeholder_counts_yearly_df,
)


EXPLORE_YEAR_MIN = 2020
EXPLORE_YEAR_MAX = 2025


def build_story_metrics() -> dict[str, str]:
    """Compute headline figures for the opening section."""
    story_stakeholder_df = stakeholder_counts_yearly_df[
        stakeholder_counts_yearly_df["Year"].between(2020, 2025)
    ].copy()
    in_depth_stakeholder_df = story_stakeholder_df[
        ~story_stakeholder_df["Stakeholder"].isin(["General public", "Other"])
    ].copy()
    story_industry_df = industry_counts_yearly_display_df[
        industry_counts_yearly_display_df["Year"].between(2020, 2025)
    ].copy()
    story_incidents_df = reported_case_totals_monthly_df[
        reported_case_totals_monthly_df["Date"].dt.year.between(2020, 2025)
    ].copy()
    story_incidents_df["Year"] = story_incidents_df["Date"].dt.year

    total_incidents = int(story_incidents_df["Total Incidents & Hazards"].sum())

    stakeholder_totals = (
        in_depth_stakeholder_df.groupby("Stakeholder", as_index=False)["Count"].sum()
        .sort_values("Count", ascending=False)
    )
    top_stakeholder = stakeholder_totals.iloc[0]

    industry_totals = (
        story_industry_df.groupby("Industry", as_index=False)["Count"].sum()
        .sort_values("Count", ascending=False)
    )
    top_industry = industry_totals.iloc[0]

    yearly_totals = (
        story_incidents_df.groupby("Year", as_index=False)["Total Incidents & Hazards"]
        .sum()
        .sort_values("Year")
    )
    first_year = yearly_totals.iloc[0]
    last_year = yearly_totals.iloc[-1]
    growth_pct = (
        (last_year["Total Incidents & Hazards"] - first_year["Total Incidents & Hazards"])
        / first_year["Total Incidents & Hazards"]
    ) * 100

    return {
        "total_incidents": f"{total_incidents:,}",
        "top_stakeholder": top_stakeholder["Stakeholder"],
        "top_stakeholder_count": f"{int(top_stakeholder['Count']):,}",
        "top_industry": top_industry["Industry"],
        "growth": f"{growth_pct:.0f}%",
        "first_year": str(int(first_year["Year"])),
        "last_year": str(int(last_year["Year"])),
    }


def build_story_stakeholder_note() -> str:
    """Return the overview note for included and excluded stakeholder groups."""
    return (
        "Other stakeholder groups included in this dataset are Women, Civil society, and Trade unions."
    )


def build_selection_summary(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
    selected_years: tuple[int, int],
) -> tuple[str, str]:
    """Generate one concise narrative based on the active filters."""
    year_range_label = build_year_window_label(selected_years)

    if filtered_stakeholder_df.empty:
        return (
            f"No reported cases are available for {selected_stakeholder.lower()} in {year_range_label}.",
            "Try widening the time period or switching to all stakeholders to compare broader patterns.",
        )

    ranked = (
        filtered_stakeholder_df.groupby("Stakeholder", as_index=False)["Count"].sum()
        .sort_values("Count", ascending=False)
    )
    peak_row = filtered_stakeholder_df.sort_values("Count", ascending=False).iloc[0]

    if selected_stakeholder == "All stakeholders":
        lead = ranked.iloc[0]
        return (
            f"{lead['Stakeholder']} appears most often in the selected time window, with {int(lead['Count']):,} coded cases.",
            f"The sharpest point in the trend is {int(peak_row['Year'])}, when {peak_row['Stakeholder']} reached {int(peak_row['Count']):,} reported cases.",
        )

    total_cases = int(filtered_stakeholder_df["Count"].sum())
    return (
        f"{selected_stakeholder} appears in {total_cases:,} coded cases across {year_range_label.lower()}.",
        f"The highest annual level in this view is {int(peak_row['Year'])}, with {int(peak_row['Count']):,} reported cases affecting {selected_stakeholder.lower()}.",
    )


def build_year_window_label(selected_years: tuple[int, int]) -> str:
    """Return one short label for the active time window."""
    start_year, end_year = selected_years
    return f"{start_year}-{end_year}"


def get_partial_2026_label() -> str:
    """Return the latest available 2026 month label for Explore copy."""
    latest_2026_date = industry_counts_monthly_display_df.loc[
        industry_counts_monthly_display_df["Year"] == 2026, "Date"
    ].max()
    if pd.isna(latest_2026_date):
        return "2026"
    return latest_2026_date.strftime("%B %Y")


def get_partial_2026_until_label() -> str:
    """Return the short label for the available 2026 partial data."""
    latest_2026_date = industry_counts_monthly_display_df.loc[
        industry_counts_monthly_display_df["Year"] == 2026, "Date"
    ].max()
    if pd.isna(latest_2026_date):
        return "2026"
    return latest_2026_date.strftime("%b %Y")


def build_effective_year_window_label(
    selected_years: tuple[int, int],
    include_2026: bool,
) -> str:
    """Return the active time-window label, including the 2026 partial slice when enabled."""
    base_label = build_year_window_label(selected_years)
    if include_2026 and selected_years[1] == EXPLORE_YEAR_MAX:
        return f"{base_label} + partial 2026 ({get_partial_2026_until_label()})"
    return base_label


def build_fastest_growing_industry_summary(
    industry_trend_df: pd.DataFrame,
    monthly_industry_df: pd.DataFrame,
    include_2026: bool,
) -> str:
    """Return the industry with the strongest growth across the window."""
    if industry_trend_df.empty:
        return "No industry trend data is available for the selected window."

    base_trend_df = industry_trend_df[industry_trend_df["Year"] < 2026].copy()
    if base_trend_df.empty:
        base_trend_df = industry_trend_df.copy()

    year_bounds = sorted(base_trend_df["Year"].unique())
    start_year = year_bounds[0]
    end_year = year_bounds[-1]

    start_df = (
        base_trend_df[base_trend_df["Year"] == start_year][["Industry", "Count"]]
        .rename(columns={"Count": "Start Count"})
    )
    end_df = (
        base_trend_df[base_trend_df["Year"] == end_year][["Industry", "Count"]]
        .rename(columns={"Count": "End Count"})
    )
    growth_df = start_df.merge(end_df, on="Industry", how="outer").fillna(0)
    growth_df["Growth"] = growth_df["End Count"] - growth_df["Start Count"]
    growth_row = growth_df.iloc[growth_df["Growth"].argmax()]

    summary = f"{growth_row['Industry']} shows the largest increase in reported cases over this period."

    if include_2026 and not monthly_industry_df.empty and (monthly_industry_df["Year"] == 2026).any():
        latest_2026_date = monthly_industry_df.loc[
            monthly_industry_df["Year"] == 2026, "Date"
        ].max()
        comparison_month = int(latest_2026_date.month)
        ytd_df = monthly_industry_df[
            (
                (monthly_industry_df["Year"] == 2025)
                & (monthly_industry_df["Date"].dt.month <= comparison_month)
            )
            | (
                (monthly_industry_df["Year"] == 2026)
                & (monthly_industry_df["Date"].dt.month <= comparison_month)
            )
        ].copy()

        ytd_growth_df = (
            ytd_df.groupby(["Year", "Industry"], as_index=False)["Count"].sum()
            .pivot(index="Industry", columns="Year", values="Count")
            .fillna(0)
            .reset_index()
        )
        if {2025, 2026}.issubset(ytd_growth_df.columns):
            ytd_growth_df["Growth"] = ytd_growth_df[2026] - ytd_growth_df[2025]
            ytd_row = ytd_growth_df.iloc[ytd_growth_df["Growth"].argmax()]
            month_label = latest_2026_date.strftime("%B")
            summary += (
                "<br><br>"
                f"January–{month_label} cases for {ytd_row['Industry']} rose from "
                f"{int(ytd_row[2025]):,} in 2025 to {int(ytd_row[2026]):,} in 2026."
            )
            if ytd_row["Industry"] != growth_row["Industry"]:
                summary += (
                    f" {growth_row['Industry']} is no longer the fastest-growing industry "
                    "in the year-to-date comparison."
                )
            else:
                summary += (
                    "\n\nIt remains the fastest-growing industry in the year-to-date comparison."
                )

    return summary


def build_stakeholder_peak_summary(
    filtered_stakeholder_df: pd.DataFrame,
    include_2026: bool,
) -> str:
    """Return the peak year summary for incidents and hazards."""
    if filtered_stakeholder_df.empty:
        return "No incidents and hazards data is available for the selected view."

    year_min = int(filtered_stakeholder_df["Year"].min())
    year_max = int(filtered_stakeholder_df["Year"].max())
    yearly_totals = (
        reported_case_totals_monthly_df[
            reported_case_totals_monthly_df["Date"].dt.year.between(year_min, year_max)
        ]
        .assign(Year=lambda df: df["Date"].dt.year)
        .groupby("Year", as_index=False)["Total Incidents & Hazards"]
        .sum()
        .rename(columns={"Total Incidents & Hazards": "Count"})
    )
    yearly_totals = yearly_totals.sort_values("Year").reset_index(drop=True)
    peak_row = yearly_totals.iloc[yearly_totals["Count"].idxmax()]
    summary = (
        f"{int(peak_row['Year'])} had the highest annual total of incidents and hazards "
        f"({int(peak_row['Count']):,})."
    )

    if include_2026 and (yearly_totals["Year"] == 2026).any():
        current_2026 = float(yearly_totals.loc[yearly_totals["Year"] == 2026, "Count"].iloc[0])
        observed_months = reported_case_totals_monthly_df.loc[
            reported_case_totals_monthly_df["Date"].dt.year == 2026,
            "Date",
        ].dt.month.nunique()
        if observed_months == 0:
            return summary
        projected_2026 = current_2026 / observed_months * 12

        prior_year_totals = yearly_totals[yearly_totals["Year"] < 2026]
        if not prior_year_totals.empty:
            prior_peak_row = prior_year_totals.iloc[prior_year_totals["Count"].idxmax()]
            if projected_2026 > float(prior_peak_row["Count"]):
                summary += (
                    "<br><br>"
                    f"If the remaining months in 2026 continue at the same pace, "
                    f"2026 will become the peak year at about {projected_2026:,.0f} cases."
                )
            else:
                summary += (
                    "<br><br>"
                    f"If the remaining months in 2026 continue at the same pace, "
                    f"2026 would still remain below the current peak of {int(prior_peak_row['Year'])}."
                )

    return summary


def build_stakeholder_peak_metric(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
) -> tuple[str, str]:
    """Return a compact metric-card version of the stakeholder peak summary."""
    if filtered_stakeholder_df.empty:
        return ("No data", "No stakeholder data is available for the selected view.")

    yearly_totals = (
        filtered_stakeholder_df.groupby("Year", as_index=False)["Count"].sum()
        if selected_stakeholder == "All stakeholders"
        else filtered_stakeholder_df[["Year", "Count"]].copy()
    )
    yearly_totals = yearly_totals.sort_values("Year").reset_index(drop=True)
    peak_row = yearly_totals.iloc[yearly_totals["Count"].idxmax()]

    value = f"{int(peak_row['Count']):,} total cases in {int(peak_row['Year'])}"
    return (value, "")


def build_stakeholder_total_metric(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
) -> tuple[str, str]:
    """Return a compact metric summarizing total coded cases in the active view."""
    if filtered_stakeholder_df.empty:
        return ("No data", "No stakeholder data is available for the selected view.")

    total_cases = int(filtered_stakeholder_df["Count"].sum())
    label = "All groups" if selected_stakeholder == "All stakeholders" else selected_stakeholder
    return (f"{total_cases:,}", f"{label} across the active time window.")


def build_stakeholder_change_metric(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
) -> tuple[str, str]:
    """Return a compact metric for change across the active stakeholder window."""
    if filtered_stakeholder_df.empty:
        return ("No data", "No stakeholder trend data is available for the selected view.")

    base_df = filtered_stakeholder_df[filtered_stakeholder_df["Year"] < 2026].copy()
    if base_df.empty:
        base_df = filtered_stakeholder_df.copy()

    yearly_totals = (
        base_df.groupby("Year", as_index=False)["Count"].sum()
        if selected_stakeholder == "All stakeholders"
        else base_df[["Year", "Count"]].copy()
    )
    yearly_totals = yearly_totals.sort_values("Year").reset_index(drop=True)

    if len(yearly_totals) < 2:
        only_row = yearly_totals.iloc[0]
        return (f"{int(only_row['Count']):,}", f"Only {int(only_row['Year'])} is available in this view.")

    first_row = yearly_totals.iloc[0]
    last_row = yearly_totals.iloc[-1]
    change = int(last_row["Count"] - first_row["Count"])
    direction = "up" if change >= 0 else "down"
    value = f"{abs(change):,} {direction}"
    copy = f"From {int(first_row['Year'])} to {int(last_row['Year'])}."
    return (value, copy)


def build_stakeholder_story_bridge(
    filtered_stakeholder_df: pd.DataFrame,
    selected_stakeholder: str,
) -> tuple[str, str]:
    """Return a screenshot-friendly title and takeaway for the active stakeholder view."""
    if filtered_stakeholder_df.empty:
        return (
            "No stakeholder pattern is available in this view.",
            "Try widening the time period or selecting all stakeholders to compare a broader pattern.",
        )

    base_df = filtered_stakeholder_df[filtered_stakeholder_df["Year"] < 2026].copy()
    if base_df.empty:
        base_df = filtered_stakeholder_df.copy()

    yearly_totals = (
        base_df.groupby("Year", as_index=False)["Count"].sum()
        if selected_stakeholder == "All stakeholders"
        else base_df[["Year", "Count"]].copy()
    ).sort_values("Year").reset_index(drop=True)

    peak_row = yearly_totals.iloc[yearly_totals["Count"].idxmax()]
    total_cases = int(base_df["Count"].sum())

    if selected_stakeholder == "All stakeholders":
        title = "How reported AI harm shifts across stakeholder groups"
        lead_row = (
            base_df.groupby("Stakeholder", as_index=False)["Count"].sum()
            .sort_values("Count", ascending=False)
            .iloc[0]
        )
        copy = (
            f"{lead_row['Stakeholder']} appears most often in this window, with {int(lead_row['Count']):,} coded cases. "
            f"The highest annual level across the selected groups is {int(peak_row['Year'])}, at {int(peak_row['Count']):,} cases."
        )
        return (title, copy)

    start_year = int(yearly_totals.iloc[0]["Year"])
    end_year = int(yearly_totals.iloc[-1]["Year"])
    title = f"AI Cases, {start_year} - {end_year}: {selected_stakeholder}"

    if len(yearly_totals) < 2:
        copy = (
            f"{selected_stakeholder} appears in {total_cases:,} coded cases in this view. "
            f"The highest annual level shown is {int(peak_row['Year'])}, with {int(peak_row['Count']):,} cases."
        )
        return (title, copy)

    first_row = yearly_totals.iloc[0]
    last_row = yearly_totals.iloc[-1]
    change = int(last_row["Count"] - first_row["Count"])
    direction_phrase = (
        "rose over the selected period"
        if change > 0
        else "declined over the selected period"
        if change < 0
        else "stayed level over the selected period"
    )
    copy = (
        f"{selected_stakeholder} appears in {total_cases:,} coded cases and {direction_phrase}. "
        f"The highest annual level in this view is {int(peak_row['Year'])}, with {int(peak_row['Count']):,} cases."
    )
    return (title, copy)


def build_fastest_growing_industry_metric(
    industry_trend_df: pd.DataFrame,
) -> tuple[str, str]:
    """Return a compact metric-card version of the fastest-growing industry summary."""
    if industry_trend_df.empty:
        return ("No data", "No industry trend data is available for the selected window.")

    base_trend_df = industry_trend_df[industry_trend_df["Year"] < 2026].copy()
    if base_trend_df.empty:
        base_trend_df = industry_trend_df.copy()

    year_bounds = sorted(base_trend_df["Year"].unique())
    start_year = year_bounds[0]
    end_year = year_bounds[-1]
    start_df = (
        base_trend_df[base_trend_df["Year"] == start_year][["Industry", "Count"]]
        .rename(columns={"Count": "Start Count"})
    )
    end_df = (
        base_trend_df[base_trend_df["Year"] == end_year][["Industry", "Count"]]
        .rename(columns={"Count": "End Count"})
    )
    growth_df = start_df.merge(end_df, on="Industry", how="outer").fillna(0)
    growth_df["Growth"] = growth_df["End Count"] - growth_df["Start Count"]
    growth_row = growth_df.iloc[growth_df["Growth"].argmax()]
    value = str(growth_row["Industry"])
    return (value, "")


def prepare_explore_data(
    selected_years: tuple[int, int],
    selected_stakeholder: str,
) -> dict[str, pd.DataFrame]:
    """Prepare filtered stakeholder and industry data for the Explore tab."""
    explore_stakeholder_df = stakeholder_counts_yearly_df[
        ~stakeholder_counts_yearly_df["Stakeholder"].isin(["General public", "Other"])
    ].copy()

    filtered_stakeholder_df = explore_stakeholder_df[
        explore_stakeholder_df["Year"].between(selected_years[0], selected_years[1])
    ]
    if selected_stakeholder != "All stakeholders":
        filtered_stakeholder_df = filtered_stakeholder_df[
            filtered_stakeholder_df["Stakeholder"] == selected_stakeholder
        ]

    stakeholder_high_level_df = stakeholder_counts_yearly_df[
        stakeholder_counts_yearly_df["Year"].between(selected_years[0], selected_years[1])
    ].copy()
    filtered_industry_df = industry_counts_yearly_display_df[
        industry_counts_yearly_display_df["Year"].between(selected_years[0], selected_years[1])
    ]
    filtered_industry_monthly_df = industry_counts_monthly_long_df[
        industry_counts_monthly_long_df["Year"].between(selected_years[0], selected_years[1])
    ].copy()
    industry_trend_df = industry_counts_yearly_display_df[
        industry_counts_yearly_display_df["Year"].between(selected_years[0], selected_years[1])
    ].copy()

    return {
        "explore_stakeholder_df": explore_stakeholder_df,
        "filtered_stakeholder_df": filtered_stakeholder_df,
        "stakeholder_high_level_df": stakeholder_high_level_df,
        "filtered_industry_df": filtered_industry_df,
        "filtered_industry_monthly_df": filtered_industry_monthly_df,
        "industry_trend_df": industry_trend_df,
    }


def normalize_explore_years(
    selected_years: tuple[int, int] | list[int] | None,
) -> tuple[int, int]:
    """Clamp and normalize the year-range control value."""
    if not selected_years or len(selected_years) != 2:
        return (EXPLORE_YEAR_MIN, EXPLORE_YEAR_MAX)

    start_year = int(selected_years[0])
    end_year = int(selected_years[1])
    start_year = max(EXPLORE_YEAR_MIN, min(start_year, EXPLORE_YEAR_MAX))
    end_year = max(EXPLORE_YEAR_MIN, min(end_year, EXPLORE_YEAR_MAX))
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    return (start_year, end_year)
