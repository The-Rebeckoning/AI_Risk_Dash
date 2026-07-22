from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INDUSTRY_COLUMNS = [
    "Government, security, and defence",
    "Mobility and autonomous vehicles",
    "Digital security",
    "Media, social platforms, and marketing",
    "Robots, sensors, and IT hardware",
    "Consumer services",
    "Financial and insurance services",
    "Healthcare, drugs, and biotechnology",
    "IT infrastructure and hosting",
    "Education and training",
]

INDUSTRY_SOURCE_LABELS = {
    "Government & security & defence": "Government, security, and defence",
    "Government / security / defence": "Government, security, and defence",
    "Media & social platforms & marketing": "Media, social platforms, and marketing",
    "Media / social platforms / marketing": "Media, social platforms, and marketing",
    "Robots & sensors & IT hardware": "Robots, sensors, and IT hardware",
    "Robots / sensors / IT hardware": "Robots, sensors, and IT hardware",
    "Healthcare & drugs & biotechnology": "Healthcare, drugs, and biotechnology",
    "Healthcare / drugs / biotechnology": "Healthcare, drugs, and biotechnology",
    "Arts & entertainment &": "Arts, entertainment, and recreation",
    "Arts & entertainment & recreation": "Arts, entertainment, and recreation",
    "Arts / entertainment / recreation": "Arts, entertainment, and recreation",
}

INDUSTRY_DISPLAY_LABELS = {
    "Government, security, and defence": "Government, security, and defense",
}

STAKEHOLDER_COLUMNS = [
    "General public",
    "Consumers",
    "Workers",
    "Government",
    "Business",
    "Children",
    "Civil society",
    "Other",
    "Women",
    "Trade unions",
]

STAKEHOLDER_TEXT = {
    "General public": "This view shows incidents affecting the general public over time.",
    "Consumers": "This view focuses on harms or incidents affecting consumers.",
    "Workers": "This view highlights stakeholder impact on workers.",
    "Government": "This view summarizes issues affecting government stakeholders.",
    "Business": "This view tracks incidents affecting businesses.",
    "Children": "This view focuses on cases affecting children.",
    "Civil society": "This view highlights impacts on civil society groups.",
    "Other": "This view captures stakeholders outside the main categories.",
    "Women": "This view focuses on incidents affecting women.",
    "Trade unions": "This view highlights impacts on trade unions.",
}


def _read_csv(
    filename: str,
    column_names: list[str] | None = None,
    skiprows: int | None = None,
    usecols: list[int] | None = None,
) -> pd.DataFrame:
    """Load one CSV file from the project's data folder."""
    read_kwargs = {"comment": "#"}
    if column_names is not None:
        read_kwargs.update({"header": None, "names": column_names})
    if skiprows is not None:
        read_kwargs["skiprows"] = skiprows
    if usecols is not None:
        read_kwargs["usecols"] = usecols

    return pd.read_csv(DATA_DIR / filename, **read_kwargs)


def _prepare_date_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a parsed `Date` column and a derived `Year` column."""
    prepared_df = df.copy()
    if "Date" in prepared_df.columns:
        prepared_df["Date"] = pd.to_datetime(prepared_df["Date"])
        prepared_df["Year"] = prepared_df["Date"].dt.year
    return prepared_df


def _prepare_trimmed_columns_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with extra whitespace removed from column names."""
    prepared_df = df.copy()
    prepared_df.columns = prepared_df.columns.str.strip()
    return prepared_df


def _prepare_industries_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned industries dataframe with parsed dates and fixed labels."""
    prepared_df = _prepare_trimmed_columns_dataframe(df)
    prepared_df = prepared_df.rename(columns=INDUSTRY_SOURCE_LABELS)
    prepared_df["Date"] = pd.to_datetime(
        prepared_df["Date"],
        format="%Y-%m",
        errors="coerce",
    )
    prepared_df = prepared_df.dropna(subset=["Date"]).copy()
    prepared_df["Year"] = prepared_df["Date"].dt.year
    return prepared_df



# CSV-backed base tables
stakeholder_counts_monthly_df = _prepare_date_dataframe(
    _read_csv("aim-affected_stakeholders-05-2026.csv")
)
industry_counts_monthly_df = _prepare_industries_dataframe(
    _read_csv(
        "aim-industries-05-2026.csv",
        column_names=["Date", *INDUSTRY_COLUMNS],
        skiprows=1,
        usecols=list(range(11)),
    )
)
severity_split_monthly_df = _prepare_date_dataframe(
    _prepare_trimmed_columns_dataframe(_read_csv("aim-severity-05-2026.csv"))
)
reported_case_totals_monthly_df = _prepare_date_dataframe(_read_csv("aim-incidents-05-2026.csv"))


def to_long_format(
    df: pd.DataFrame,
    value_columns: list[str],
    var_name: str = "Category",
    value_name: str = "value",
) -> pd.DataFrame:
    """Return a dataframe reshaped from wide format into long format.

    This turns multiple category columns into one label column and one value
    column. Use it when you want a dataframe that is easier to filter, group,
    and chart.
    """
    id_columns = [column for column in df.columns if column not in value_columns]
    return df.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name=var_name,
        value_name=value_name,
    )


def get_filter_options(
    df: pd.DataFrame,
    column_name: str,
    sort_values: bool = True,
) -> list:
    """Return the unique values for one dataframe column.

    Use this to populate a dropdown or selectbox for any dataframe column, such
    as stakeholder, country, severity, or industry.
    """
    options = df[column_name].dropna().unique().tolist()
    if sort_values:
        options = sorted(options)
    return options


def get_stakeholder_description(stakeholder: str) -> str:
    """Return the descriptive text for the selected stakeholder."""
    return STAKEHOLDER_TEXT.get(
        stakeholder,
        "This view shows incidents affecting the selected stakeholder over time.",
    )


def filter_dataframe(
    df: pd.DataFrame,
    column_name: str,
    selected_value,
) -> pd.DataFrame:
    """Return only the rows where one column matches the selected value.

    Use this when a user picks one option in the UI and you want the matching
    subset of rows for a table or chart.
    """
    return df[df[column_name] == selected_value].copy()


def build_time_series(
    df: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    description_text: str | None = None,
    description_column: str = "tooltip_text",
) -> pd.DataFrame:
    """Return grouped totals ready for a trend chart.

    This groups the dataframe by the columns you choose, sums the value column,
    and can add tooltip text. Use it for line charts that show change over time.
    """
    grouped_df = df.groupby(group_columns, as_index=False)[value_column].sum()
    if description_text is not None:
        grouped_df[description_column] = description_text
    return grouped_df


def group_by_year(
    df: pd.DataFrame,
    value_column: str,
    category_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return totals grouped by year and any optional category columns."""
    group_columns = ["Year", *(category_columns or [])]
    return build_time_series(df, group_columns, value_column)


def get_category_totals(
    df: pd.DataFrame,
    label_name: str,
    value_name: str = "Total Count",
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return total counts by category, ready for a summary chart.

    This adds up category columns across the dataframe and returns a simple
    two-column table. Use it for bar charts that compare totals by category.
    """
    excluded = set(exclude_columns or ["Date", "Year"])
    value_columns = [column for column in df.columns if column not in excluded]
    totals = df[value_columns].sum().sort_values(ascending=False)
    totals_df = totals.reset_index()
    totals_df.columns = [label_name, value_name]
    return totals_df


def rename_industry_labels_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with industry labels normalized for display text."""
    return df.rename(columns=INDUSTRY_DISPLAY_LABELS).replace({"Industry": INDUSTRY_DISPLAY_LABELS})


def build_monthly_category_counts(
    df: pd.DataFrame,
    category_columns: list[str],
    category_name: str,
) -> pd.DataFrame:
    """Return a long-form monthly category-count table."""
    return to_long_format(
        df,
        category_columns,
        var_name=category_name,
        value_name="Count",
    )


stakeholder_counts_long_df = to_long_format(
    stakeholder_counts_monthly_df,
    STAKEHOLDER_COLUMNS,
    var_name="Stakeholder",
    value_name="Count",
)

stakeholder_counts_yearly_df = group_by_year(
    stakeholder_counts_long_df,
    "Count",
    ["Stakeholder"],
)

stakeholder_counts_yearly_totals_df = group_by_year(stakeholder_counts_long_df, "Count")

industry_counts_yearly_df = group_by_year(
    to_long_format(
        industry_counts_monthly_df,
        INDUSTRY_COLUMNS,
        var_name="Industry",
        value_name="Count",
    ),
    "Count",
    ["Industry"],
)

industry_count_totals_df = get_category_totals(
    industry_counts_monthly_df,
    "Industry",
    "Total Count",
    exclude_columns=["Date", "Year"],
)

industry_counts_yearly_display_df = rename_industry_labels_for_display(industry_counts_yearly_df)
industry_counts_monthly_display_df = rename_industry_labels_for_display(industry_counts_monthly_df)
industry_counts_monthly_long_df = build_monthly_category_counts(
    industry_counts_monthly_display_df,
    [INDUSTRY_DISPLAY_LABELS.get(column, column) for column in INDUSTRY_COLUMNS],
    "Industry",
)
