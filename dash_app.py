"""Dash edition of the Reported AI Cases Dashboard.

Run with: python dash_app.py
The original Streamlit application remains in app.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_mantine_components as dmc
from dash import ALL, Dash, Input, MATCH, Output, State, callback, ctx, dash_table, dcc, html

from ai_lib import (
    build_fastest_growing_industry_summary,
    build_story_metrics,
    build_stakeholder_peak_summary,
    get_partial_2026_until_label,
    industry_counts_yearly_display_df,
    prepare_explore_data,
    reported_case_totals_monthly_df,
    severity_split_monthly_df,
    stakeholder_counts_yearly_df,
)
from ai_lib.openai_api import get_article_component_data


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
COLORS = ["#222222", "#4f4f4f", "#6f6f6f", "#8e3b46", "#b85c6b", "#c78d96"]
NAV_ITEMS = (
    ("overview", "Overview"),
    ("explore", "Explore the data"),
    ("methodology", "Data & methodology"),
    ("downloads", "Download the data"),
)
DATASETS = {
    "stakeholders": ("Stakeholders", "aim-affected_stakeholders-05-2026.csv"),
    "industries": ("Industry", "aim-industries-05-2026.csv"),
    "totals": ("Monthly Totals", "aim-incidents-05-2026.csv"),
    "severity": ("Incident vs Hazard Split", "aim-severity-05-2026.csv"),
}


def style_chart(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.42)",
        margin=dict(l=28, r=20, t=62, b=28),
        font=dict(color="#262626", size=14),
        title_font=dict(size=20, color="#111"),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,.08)", zeroline=False)
    return fig


def place_legend_below(fig: go.Figure, height: int, bottom_margin: int) -> go.Figure:
    """Place a centered, wrapping legend beneath a chart."""
    fig.update_layout(
        height=height,
        margin=dict(l=28, r=20, t=62, b=bottom_margin),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            title_text="",
        ),
    )
    return fig


def card(*children, class_name: str = "section-card") -> dmc.Paper:
    return dmc.Paper(list(children), className=class_name, radius="lg", withBorder=True)


def intro(kicker: str, title: str, copy: str = "") -> html.Div:
    return html.Div([
        html.Div(kicker, className="section-kicker"),
        html.Div(title, className="section-title"),
        html.Div(copy, className="section-copy") if copy else None,
    ])


def insight(label: str, text: str, collapsible: bool = True):
    display_text = text.replace("<br><br>", "\n\n")
    if not collapsible:
        return html.Div(
            [
                html.Div(label, className="insight-label"),
                html.Div(display_text, className="insight-text"),
            ],
            className="insight-card",
        )
    return html.Details(
        [
            html.Summary(label, className="insight-label"),
            html.Div(display_text, className="insight-text"),
        ],
        className="insight-card",
    )


def chart_insight(label: str, text: str, footer: list | None = None) -> html.Div:
    """Render a compact chart takeaway with a visually separate supporting note."""
    sections = text.replace("<br><br>", "\n\n").split("\n\n", 1)
    return html.Div([
        html.Div(label, className="chart-insight-label"),
        html.Div(sections[0], className="chart-insight-lead"),
        html.Div(sections[1], className="chart-insight-note") if len(sections) > 1 else None,
        html.Div(
            [html.P(paragraph) for paragraph in footer],
            className="chart-insight-footer",
        ) if footer else None,
    ], className="chart-insight")


def metric(label: str, value: str, copy: str) -> html.Div:
    value_class = "metric-value metric-value-long" if len(value) > 20 else "metric-value"
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className=value_class),
        html.Div(copy, className="metric-copy"),
    ], className="metric-card")


def graph(fig: go.Figure) -> dcc.Graph:
    return dcc.Graph(figure=fig, config={"displayModeBar": False}, className="chart-shell")


def overview_figures() -> tuple[go.Figure, go.Figure, go.Figure]:
    totals = (reported_case_totals_monthly_df
              .assign(Year=lambda df: df["Date"].dt.year)
              .query("2020 <= Year <= 2025")
              .groupby("Year", as_index=False)["Total Incidents & Hazards"].sum())
    latest_total = int(totals.loc[totals["Year"] == 2025, "Total Incidents & Hazards"].iloc[0])
    previous_total = int(totals.loc[totals["Year"] == 2024, "Total Incidents & Hazards"].iloc[0])
    latest_growth = (latest_total / previous_total - 1) * 100
    bar_labels = [f"{int(total):,}" for total in totals["Total Incidents & Hazards"]]
    totals_fig = px.bar(totals, x="Year", y="Total Incidents & Hazards")
    totals_fig.update_traces(
        marker_color=["#c9c9c9" if year < 2025 else "#8e3b46" for year in totals["Year"]],
        marker_line_width=0,
        marker_cornerradius=5,
        text=bar_labels,
        texttemplate="%{text}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="Year %{x}<br>Reported cases %{y:,}<extra></extra>",
    )
    style_chart(totals_fig, 350)
    totals_fig.update_layout(
        title=None,
        bargap=0.32,
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=35, b=8),
        xaxis_title=None,
        yaxis_title="Reported cases",
        showlegend=False,
    )
    totals_fig.add_annotation(
        x=2025,
        y=latest_total,
        text=f"<b>+{latest_growth:.0f}%</b> vs 2024",
        showarrow=False,
        yshift=52,
        bgcolor="#f5e7e9",
        bordercolor="#8e3b46",
        borderwidth=1,
        borderpad=7,
        font=dict(color="#71303a", size=13),
        opacity=1,
    )
    totals_fig.update_xaxes(tickmode="linear", dtick=1)
    totals_fig.update_yaxes(
        tickformat=",",
        gridcolor="rgba(0,0,0,.07)",
        range=[0, totals["Total Incidents & Hazards"].max() * 1.25],
    )

    stakeholder = stakeholder_counts_yearly_df.query("2020 <= Year <= 2025")
    stakeholder = stakeholder[~stakeholder["Stakeholder"].isin(["General public", "Other"])]
    top = stakeholder.groupby("Stakeholder")["Count"].sum().nlargest(5).index
    stakeholder_fig = px.line(
        stakeholder[stakeholder["Stakeholder"].isin(top)], x="Year", y="Count",
        color="Stakeholder", markers=True, title="Affected stakeholder patterns",
        color_discrete_sequence=COLORS,
    )

    industries = industry_counts_yearly_display_df.query("2020 <= Year <= 2025")
    industry_totals = industries.groupby("Industry", as_index=False)["Count"].sum().nlargest(10, "Count")
    industry_fig = px.bar(
        industry_totals.sort_values("Count"), x="Count", y="Industry", orientation="h",
        title="Industries with the most reported cases", color="Count",
        color_continuous_scale=["#d8d8d8", "#9a9a9a", "#2a2a2a"],
    )
    industry_fig.update_layout(coloraxis_showscale=False)
    return totals_fig, style_chart(stakeholder_fig), style_chart(industry_fig, 470)


def severity_figure() -> go.Figure:
    fig = go.Figure([
        go.Scatter(x=severity_split_monthly_df["Date"], y=severity_split_monthly_df["AI incident"], stackgroup="one", name="AI incident", line=dict(color="#1f1f1f")),
        go.Scatter(x=severity_split_monthly_df["Date"], y=severity_split_monthly_df["AI hazard"], stackgroup="one", name="AI hazard", line=dict(color="#b85c6b")),
    ])
    fig.update_layout(title="Incident and hazard split over time", hovermode="x unified")
    return style_chart(fig, 390)


METRICS = build_story_metrics()
TOTALS_FIG, STAKEHOLDER_FIG, INDUSTRY_FIG = overview_figures()


def overview_layout() -> html.Div:
    return html.Div([
        html.Div([
            html.Div("Based on OECD AI Incidents and Hazards Monitor data", className="hero-eyebrow"),
            html.H1("Reported AI cases and what they show", className="hero-title"),
            html.P("An overview of reported AI cases, showing who is affected, which industries are involved, and how incidents appear over time.", className="hero-copy"),
            html.P("Counts reflect reported and coded cases in the OECD monitor, not every real-world AI harm.", className="hero-note"),
        ], className="hero-shell"),
        html.Div([
            metric("Incidents and hazards", METRICS["total_incidents"], "Reported cases between 2020–2025."),
            metric("Most affected group", METRICS["top_stakeholder"], f'{METRICS["top_stakeholder_count"]} stakeholder-coded cases.'),
            metric("Most exposed industry", METRICS["top_industry"], "Largest concentration of reported cases."),
            metric("Change over time", METRICS["growth"], f'{METRICS["first_year"]} to {METRICS["last_year"]}.'),
        ], className="metric-grid"),
        card(intro("Reported cases by year", "How reported AI cases change over time"), graph(TOTALS_FIG)),
        html.Div([
            card(graph(STAKEHOLDER_FIG)),
            card(intro("Stakeholder patterns", "Affected stakeholders", "A case can be assigned to more than one stakeholder."),
                 insight("Missing groups", "Some groups are not captured as categories and may be underrepresented."),
                 insight("Coverage versus occurrence", "Counts reflect coverage and coding, not unique events.")),
        ], className="two-column-grid overview-grid graph-first-grid"),
        html.Div([
            card(intro("Industry patterns", "Where reported cases surface"),
                 insight("Visibility shapes the pattern", "Public-facing failures and regulated industries are more likely to be documented."),
                 insight("Risk is interpreted by people", "AI risk is technical, organizational, and social.")),
            card(graph(INDUSTRY_FIG)),
        ], className="two-column-grid overview-grid graph-last-grid"),
    ])


def explore_layout() -> html.Div:
    options = sorted(stakeholder_counts_yearly_df["Stakeholder"].unique())
    return html.Div([
        card(intro("Interactive analysis", "Explore patterns in reported AI cases", "Adjust the year window and stakeholder filter."),
             html.Div([
                 html.Div([
                     html.Div([
                         html.Div("Years", className="filter-title"),
                         html.Div("Choose the reporting window", className="filter-description"),
                     ], className="filter-heading"),
                     html.Div(
                         dmc.RangeSlider(
                             id="year-range",
                             min=2020,
                             max=2025,
                             step=1,
                             value=[2020, 2025],
                             marks=[{"value": year, "label": str(year)} for year in range(2020, 2026)],
                             minRange=1,
                             color="wine",
                             size="md",
                         ),
                         className="year-slider-shell",
                     ),
                     dmc.Checkbox(id="include-2026", checked=True, className="partial-year-check", color="wine"),
                     html.Div(id="partial-2026-note", className="partial-year-note"),
                 ], className="filter-block year-filter-block"),
                 html.Div([
                     html.Div([
                         html.Div("Filter by group", className="filter-eyebrow"),
                         html.Div("Affected stakeholder", className="filter-title stakeholder-filter-title"),
                         html.Div(
                             "Focus the charts and related case study on one stakeholder group.",
                             className="filter-description stakeholder-filter-description",
                         ),
                     ], className="filter-heading stakeholder-filter-heading"),
                     dmc.Select(
                         data=["All stakeholders", *options],
                         value="All stakeholders",
                         id="stakeholder-filter",
                         clearable=False,
                         searchable=True,
                         radius="md",
                         size="md",
                         className="stakeholder-select",
                     )
                     ,
                     html.Div(
                         "Your selection updates every chart and the related case below.",
                         className="stakeholder-filter-note",
                     ),
                 ], className="filter-block stakeholder-filter-block"),
             ], className="filter-grid"), class_name="section-card explore-intro-card"),
        dcc.Loading(
            html.Div(id="industry-chart-content"),
            type="circle",
            delay_show=350,
            overlay_style={"visibility": "visible", "filter": "blur(1px)"},
        ),
        html.Div([
            dcc.Loading(
                html.Div(id="stakeholder-chart-content"),
                type="circle",
                delay_show=350,
                overlay_style={"visibility": "visible", "filter": "blur(1px)"},
            ),
            dcc.Loading(html.Div(id="case-content"), type="circle"),
        ], className="explore-stakeholder-stack"),
        dcc.Store(id="case-index", data=0),
    ])


def methodology_layout() -> html.Div:
    return html.Div([
        card(intro("Method and source notes", "Data & methodology", "How to interpret the source, measures, and limitations."), class_name="section-card explore-intro-card"),
        card(
            intro("Key terms", "Definitions", "How the OECD monitor distinguishes reported incidents from potential hazards."),
            html.P(["Source: ", html.A("OECD AI Incidents and Hazards Monitor", href="https://oecd.ai/en/incidents", target="_blank"), "."]),
            html.Div([
                insight("Incident", "An event linked to harm reported to have occurred.", collapsible=False),
                insight("Hazard", "An event linked to plausible potential harm.", collapsible=False),
            ], className="methodology-definition-grid"),
            class_name="section-card methodology-definition-card",
        ),
        html.Div([
            card(
                intro("Incidents and hazards", "How the source data is split", "Monthly reported cases, separated by incidents and hazards."),
                graph(severity_figure()),
                class_name="section-card methodology-chart-card",
            ),
            card(
                intro("Interpretation", "How to read the counts", "Use these notes when comparing totals across views."),
                insight(
                    "What the counts mean",
                    "Overall totals count cases. Stakeholder and industry views count category assignments, so their totals can be higher.",
                    collapsible=False,
                ),
                insight("Coverage bias", "Media attention affects what appears.", collapsible=False),
                insight("Larger counts", "More reports do not necessarily mean more real-world occurrence.", collapsible=False),
            ),
        ], className="two-column-grid methodology-analysis-grid"),
    ])


def read_preview(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename, comment="#", nrows=10)


def downloads_layout() -> html.Div:
    sections = []
    for key, (title, filename) in DATASETS.items():
        preview = read_preview(filename)
        sections.append(card(
            intro("Dataset", title, f"Preview and download the {title.lower()} source data."),
            dash_table.DataTable(preview.to_dict("records"), [{"name": c, "id": c} for c in preview.columns], page_size=10,
                                 style_table={"overflowX": "auto"}, style_cell={"textAlign": "left", "padding": "8px", "fontFamily": "inherit"}),
            dmc.Button(f"Download {title} CSV", id={"type": "download-button", "name": key}, className="dash-button", color="wine", radius="xl"),
            dcc.Download(id={"type": "download", "name": key}),
        ))
    return html.Div([card(intro("Dataset access", "Data downloads", "Preview and download each source CSV."), class_name="section-card explore-intro-card"), *sections])


app = Dash(__name__, title="Reported AI Cases Dashboard", suppress_callback_exceptions=True, assets_folder="dash_assets")
server = app.server
app.layout = dmc.MantineProvider(
    dmc.AppShell(
        [
            dcc.Location(id="url", refresh=False),
            dmc.AppShellNavbar(
                [
                    html.Div([
                        html.Div("REPORTED AI CASES", className="toc-brand"),
                        html.Div(["Exploring AI", html.Br(), "Harm & Risk"], className="toc-title"),
                    ], className="toc-heading"),
                    dmc.Stack([
                        dmc.NavLink(
                            label=label,
                            id={"type": "toc-link", "page": page_key},
                            href=f"#{page_key}",
                            refresh=False,
                            active=page_key == "overview",
                            color="wine",
                            variant="filled",
                            className="toc-link",
                        )
                        for page_key, label in NAV_ITEMS
                    ], gap="xs"),
                ],
                className="toc-navbar",
            ),
            dmc.AppShellMain(html.Div(id="page", className="app-shell")),
        ],
        navbar={"width": 300, "breakpoint": "sm"},
        padding=0,
        className="mantine-shell",
    ),
    theme={
        "primaryColor": "wine",
        "colors": {
            "wine": [
                "#fbf2f3", "#f5e3e6", "#e9c5cb", "#d9a2ac", "#c57b88",
                "#ad5666", "#9d4453", "#8e3b46", "#79323c", "#642a32",
            ]
        },
        "fontFamily": "Inter, ui-sans-serif, system-ui, sans-serif",
        "headings": {"fontFamily": "Inter, ui-sans-serif, system-ui, sans-serif"},
        "defaultRadius": "md",
    },
)


@callback(
    Output("page", "children"),
    Output({"type": "toc-link", "page": ALL}, "active"),
    Input("url", "hash"),
)
def render_tab(url_hash: str):
    requested_page = (url_hash or "#overview").removeprefix("#")
    layouts = {
        "overview": overview_layout,
        "explore": explore_layout,
        "methodology": methodology_layout,
        "downloads": downloads_layout,
    }
    active_page = requested_page if requested_page in layouts else "overview"
    return layouts[active_page](), [page_key == active_page for page_key, _ in NAV_ITEMS]


@callback(
    Output("industry-chart-content", "children"),
    Output("stakeholder-chart-content", "children"),
    Input("year-range", "value"),
    Input("stakeholder-filter", "value"),
    Input("include-2026", "checked"),
)
def update_explore_charts(years, stakeholder, include_2026_checked):
    includes_2026 = years[1] == 2025 and include_2026_checked
    effective_years = (years[0], 2026 if includes_2026 else min(years[1], 2025))
    data = prepare_explore_data(effective_years, stakeholder)
    industry = data["industry_trend_df"]
    affected = data["filtered_stakeholder_df"]
    industry_fig = style_chart(px.area(industry, x="Year", y="Count", color="Industry", title="Reported cases by industry", color_discrete_sequence=COLORS))
    if stakeholder == "All stakeholders":
        affected_fig = px.line(affected, x="Year", y="Count", color="Stakeholder", title="Affected stakeholders", color_discrete_sequence=COLORS)
    else:
        affected_fig = px.line(affected, x="Year", y="Count", markers=True, title=f"Cases affecting {stakeholder}")
        affected_fig.update_traces(line_color="#8e3b46")
    affected_fig = style_chart(affected_fig)
    industry_fig.update_yaxes(title="Reported cases")
    affected_fig.update_yaxes(title="Reported cases")
    if includes_2026:
        for fig in (industry_fig, affected_fig):
            fig.add_vrect(
                x0=2025.5,
                x1=2026.5,
                fillcolor="rgba(142,59,70,.06)",
                line_width=0,
                layer="below",
            )
            fig.add_annotation(
                x=2026,
                y=1,
                yref="paper",
                text="Partial 2026 · through May",
                showarrow=False,
                yshift=22,
                font=dict(size=11, color="#8e3b46"),
            )
    industry_fig = place_legend_below(industry_fig, height=470, bottom_margin=150)
    if stakeholder == "All stakeholders":
        affected_fig = place_legend_below(affected_fig, height=470, bottom_margin=115)
    industry_summary = build_fastest_growing_industry_summary(industry, data["filtered_industry_monthly_df"], includes_2026)
    stakeholder_summary = build_stakeholder_peak_summary(affected, includes_2026)
    industry_card = html.Div([
        card(
            chart_insight(
                "Largest increase",
                industry_summary,
                footer=[
                    "Rapidly rising reports may reshape how industry professionals perceive—and respond to—AI risk."
                ],
            ),
            class_name="section-card chart-summary-card",
        ),
        card(graph(industry_fig), class_name="section-card explore-chart-card"),
    ], className="two-column-grid explore-industry-grid")
    stakeholder_card = html.Div([
        card(graph(affected_fig), class_name="section-card stakeholder-graph-card explore-chart-card"),
        card(
            chart_insight(
                "Peak year",
                stakeholder_summary,
                footer=[
                    "The annual pattern provides context for understanding who is affected and how reported harms take shape.",
                    [
                        "The rising totals are only part of the story.",
                        html.Br(),
                        html.Br(),
                        html.Strong(
                            "Their significance becomes clearer through the stories and experiences behind the data."
                        ),
                    ],
                ],
            ),
            class_name="section-card chart-summary-card stakeholder-pattern-card",
        ),
    ], className="two-column-grid stakeholder-analysis-grid")
    return industry_card, stakeholder_card


@callback(
    Output("include-2026", "label"),
    Output("include-2026", "disabled"),
    Input("year-range", "value"),
)
def update_partial_2026_options(years):
    through_label = get_partial_2026_until_label()
    reaches_latest_complete_year = years[1] == 2025
    label = f"Include partial 2026 data through {through_label}"
    return label, not reaches_latest_complete_year


@callback(
    Output("partial-2026-note", "children"),
    Input("year-range", "value"),
    Input("include-2026", "checked"),
)
def update_partial_2026_note(years, include_2026_checked):
    through_label = get_partial_2026_until_label()
    reaches_latest_complete_year = years[1] == 2025
    checked = include_2026_checked
    if not reaches_latest_complete_year:
        note = "Move the end year to 2025 to make partial 2026 data available."
    elif checked:
        note = f"Including year-to-date 2026 data through {through_label}; comparisons use matching months."
    else:
        note = "Partial 2026 data is excluded; the charts and summaries use complete years only."
    return note


@callback(Output("case-content", "children"), Input("stakeholder-filter", "value"), Input("case-index", "data"))
def update_case_card(stakeholder, case_index):
    if stakeholder == "All stakeholders":
        return card(intro("Related case", "Choose a stakeholder", "Select a group to see a relevant case study."))

    try:
        case = get_article_component_data(stakeholder, case_index=case_index or 0, api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        case = {
            "title": "Case study temporarily unavailable",
            "summary": "We couldn't load a related case right now. Please try again shortly.",
            "relevance": "",
        }
    case_children = [
        intro("Related case", case.get("title", "AI harm case")),
        html.P(case.get("summary", ""), className="case-summary"),
    ]
    if case.get("relevance"):
        case_children.append(insight("Why this case matters", case["relevance"], collapsible=False))
    case_children.append(html.Div([
        html.A(
            "View OECD source",
            href=case.get("article_url") or case.get("source_url") or "https://oecd.ai/en/incidents",
            target="_blank",
        ),
        dmc.Button("Load another", id="load-another", className="dash-button", color="wine", radius="xl"),
    ], className="case-actions"))
    return card(*case_children, class_name="section-card related-case-card")


@callback(Output("case-index", "data"), Input("load-another", "n_clicks"), State("case-index", "data"), prevent_initial_call=True)
def next_case(_clicks, index):
    return (index or 0) + 1


@callback(Output({"type": "download", "name": MATCH}, "data"), Input({"type": "download-button", "name": MATCH}, "n_clicks"), prevent_initial_call=True)
def download_dataset(_clicks):
    key = ctx.triggered_id["name"]
    return dcc.send_file(DATA_DIR / DATASETS[key][1])


if __name__ == "__main__":
    debug_mode = os.getenv("DASH_DEBUG", "true").lower() == "true"
    app.run(
        debug=debug_mode,
        dev_tools_hot_reload=debug_mode,
        dev_tools_hot_reload_interval=1,
        dev_tools_hot_reload_watch_interval=0.5,
        dev_tools_hot_reload_max_retry=20,
        use_reloader=debug_mode,
        extra_files=[
            str(ROOT / "dash_app.py"),
            str(ROOT / "dash_assets" / "dash.css"),
        ],
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8050")),
    )
