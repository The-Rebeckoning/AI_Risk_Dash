import json
import os
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlencode, urljoin

import requests
from bs4 import BeautifulSoup
from openai import OpenAI


DEFAULT_FROM_DATE = "2020-04-02"
DEFAULT_TO_DATE = date.today().isoformat()
DEFAULT_MODEL = "gpt-5-nano"
PRELOADED_CASE_STUDIES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "preloaded_case_studies.json"
)
CASE_STUDY_STAKEHOLDERS = (
    "Consumers",
    "Workers",
    "Government",
    "Business",
    "Children",
    "Civil society",
    "Women",
    "Trade unions",
)


def get_openai_client(api_key: str | None = None) -> OpenAI:
    """Return an OpenAI client using an explicit key or env var."""
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("Set OPENAI_API_KEY or pass api_key explicitly.")
    return OpenAI(api_key=resolved_api_key)


def load_preloaded_case_studies() -> dict[str, dict[str, str]]:
    """Load the local preloaded case studies used for first article loads."""
    if not PRELOADED_CASE_STUDIES_PATH.exists():
        return {}
    return json.loads(PRELOADED_CASE_STUDIES_PATH.read_text())


def build_oecd_url(
    selected_value: str,
    from_date: str = DEFAULT_FROM_DATE,
    to_date: str = DEFAULT_TO_DATE,
    num_results: int = 20,
) -> str:
    """Build the OECD AI incidents URL for one selected stakeholder."""
    properties_config = {
        "principles": [],
        "industries": [],
        "harm_types": [],
        "harm_levels": [],
        "harmed_entities": [selected_value],
        "business_functions": [],
        "ai_tasks": [],
        "autonomy_levels": [],
        "languages": [],
    }

    params = {
        "search_terms": "[]",
        "and_condition": "false",
        "from_date": from_date,
        "to_date": to_date,
        "properties_config": json.dumps(properties_config, separators=(",", ":")),
        "order_by": "date",
        "num_results": num_results,
    }
    return "https://oecd.ai/en/incidents?" + urlencode(params)


def fetch_page(url: str, timeout: int = 20) -> str:
    """Fetch one page of HTML."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def extract_article_urls(html: str) -> list[str]:
    """Extract OECD incident/article URLs from the results page when possible."""
    soup = BeautifulSoup(html, "html.parser")
    candidate_hrefs: list[str] = []

    for link in soup.find_all("a", href=True):
        href = link["href"].strip()
        if not href:
            continue
        if "/en/incidents/" in href and "/en/incidents?" not in href:
            candidate_hrefs.append(urljoin("https://oecd.ai", href))

    seen: set[str] = set()
    article_urls: list[str] = []
    for href in candidate_hrefs:
        if href not in seen:
            seen.add(href)
            article_urls.append(href)
    return article_urls


def extract_article_text(html: str, max_chars: int = 12000) -> str:
    """Extract plain text from HTML for summarization."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    content_root = soup.find("article") or soup.find("main") or soup.body or soup
    text = content_root.get_text(" ", strip=True)
    return text[:max_chars]


def build_article_prompt(selected_value: str, article_text: str) -> str:
    """Build the summarization prompt for the selected stakeholder."""
    stakeholder_list = ", ".join(CASE_STUDY_STAKEHOLDERS)
    return f"""
The user selected this stakeholder: {selected_value}.

Using the OECD AI Incidents Monitor content below, identify one relevant incident/article and return JSON only.

Use this exact schema:
{{
  "title": "Article Title - OECD.AI",
  "summary": "A plain-prose summary of the case.",
  "relevance": "A short plain-prose explanation of why this case is relevant to {selected_value}.",
  "also_affects": "One additional affected stakeholder from this list if it is clearly supported by the article, otherwise an empty string: {stakeholder_list}"
}}

Return valid JSON only. No markdown fences.
If the source title is unclear, write a concise title based on the case.
Do not repeat {selected_value} in also_affects.

OECD content:
{article_text}
""".strip()


def parse_article_response(response_text: str, fallback_title: str = "AI Harm Case") -> dict[str, str]:
    """Normalize one model response into structured article fields."""
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        parsed = {}

    title = str(parsed.get("title", "")).strip()
    summary = str(parsed.get("summary", "")).strip()
    relevance = str(parsed.get("relevance", "")).strip()
    also_affects = str(parsed.get("also_affects", "")).strip()

    if not any([title, summary, relevance]):
        blocks = [block.strip() for block in response_text.split("\n\n") if block.strip()]
        if blocks:
            title = blocks[0]
        if len(blocks) >= 2:
            summary = blocks[1]
        if len(blocks) >= 3:
            relevance = "\n\n".join(blocks[2:])

    def clean_title(text: str) -> str:
        text = re.sub(r"^\s*[-*•]\s*", "", text).strip()
        text = re.sub(r"^(incident|article|incident/article)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = text.replace("(OECD AI Incidents Monitor)", "").strip()
        text = re.sub(r"(?:\s*-\s*)?OECD\.AI$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+", " ", text).strip(" -:")
        text = text.replace(" - ", " ")
        return text

    def clean_body(text: str) -> str:
        text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"summary\s*\(.*?\)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"why it matches.*?:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\baffected stakeholder\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bstakeholder\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*-\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s-\s", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    title = clean_title(title)
    summary = clean_body(summary)
    relevance = clean_body(relevance)
    also_affects = re.sub(r"\s{2,}", " ", also_affects).strip(" -:")
    title = title or fallback_title
    return {
        "title": title,
        "summary": summary,
        "relevance": relevance,
        "also_affects": also_affects,
    }


def normalize_additional_stakeholder(selected_value: str, candidate: str) -> str:
    """Keep only valid case-study stakeholder labels for the secondary chip."""
    normalized_candidate = re.sub(r"\s+", " ", (candidate or "")).strip()
    if not normalized_candidate:
        return ""

    lookup = {stakeholder.lower(): stakeholder for stakeholder in CASE_STUDY_STAKEHOLDERS}
    resolved = lookup.get(normalized_candidate.lower(), "")
    if not resolved or resolved == selected_value:
        return ""
    return resolved


def fetch_article_text_for_stakeholder(
    selected_value: str,
    from_date: str = DEFAULT_FROM_DATE,
    to_date: str = DEFAULT_TO_DATE,
    num_results: int = 20,
) -> str:
    """Fetch OECD incident page text for one stakeholder."""
    url = build_oecd_url(
        selected_value,
        from_date=from_date,
        to_date=to_date,
        num_results=num_results,
    )
    html = fetch_page(url)
    return extract_article_text(html)


def fetch_article_bundle_for_stakeholder(
    selected_value: str,
    case_index: int = 0,
    from_date: str = DEFAULT_FROM_DATE,
    to_date: str = DEFAULT_TO_DATE,
    num_results: int = 20,
) -> dict[str, str]:
    """Fetch one OECD result-page article bundle for the selected stakeholder."""
    source_url = build_oecd_url(
        selected_value,
        from_date=from_date,
        to_date=to_date,
        num_results=num_results,
    )
    results_html = fetch_page(source_url)
    article_urls = extract_article_urls(results_html)
    article_url = ""
    article_text = extract_article_text(results_html)
    if article_urls:
        article_url = article_urls[case_index % len(article_urls)]
        try:
            article_text = extract_article_text(fetch_page(article_url))
        except requests.RequestException:
            article_text = extract_article_text(results_html)
    return {
        "source_url": source_url,
        "article_url": article_url,
        "article_text": article_text,
    }


def summarize_stakeholder_article(
    selected_value: str,
    case_index: int = 0,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    from_date: str = DEFAULT_FROM_DATE,
    to_date: str = DEFAULT_TO_DATE,
    num_results: int = 20,
) -> str:
    """Return an AI summary of one OECD incident relevant to the stakeholder."""
    article_bundle = fetch_article_bundle_for_stakeholder(
        selected_value,
        case_index=case_index,
        from_date=from_date,
        to_date=to_date,
        num_results=num_results,
    )
    prompt = build_article_prompt(selected_value, article_bundle["article_text"])
    client = get_openai_client(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def get_live_article_component_data(
    selected_value: str,
    case_index: int = 0,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    from_date: str = DEFAULT_FROM_DATE,
    to_date: str = DEFAULT_TO_DATE,
    num_results: int = 20,
) -> dict[str, str]:
    """Return one live article component payload from OECD and OpenAI."""
    client = get_openai_client(api_key=api_key)
    article_bundle = fetch_article_bundle_for_stakeholder(
        selected_value,
        case_index=case_index,
        from_date=from_date,
        to_date=to_date,
        num_results=num_results,
    )

    prompt = build_article_prompt(selected_value, article_bundle["article_text"])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    response_text = response.choices[0].message.content or "{}"
    article_content = parse_article_response(response_text)
    preferred_source_url = article_bundle.get("article_url") or article_bundle.get("source_url", "")
    source_list_url = article_bundle.get("source_url", "")
    return {
        "selected_value": selected_value,
        "source_url": preferred_source_url,
        "source_list_url": source_list_url,
        "title": article_content["title"],
        "summary": article_content["summary"],
        "relevance": article_content["relevance"],
        "also_affects": normalize_additional_stakeholder(
            selected_value,
            article_content.get("also_affects", ""),
        ),
        "is_preloaded": False,
    }


def get_article_component_data(
    selected_value: str,
    case_index: int = 0,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    from_date: str = DEFAULT_FROM_DATE,
    to_date: str = DEFAULT_TO_DATE,
    num_results: int = 20,
) -> dict[str, str]:
    """Return reusable article component data for the frontend."""
    preloaded_case = load_preloaded_case_studies().get(selected_value)
    has_api_key = bool(api_key or os.environ.get("OPENAI_API_KEY"))
    if preloaded_case and (case_index == 0 or not has_api_key):
        return {
            **preloaded_case,
            "selected_value": selected_value,
            "also_affects": normalize_additional_stakeholder(
                selected_value,
                preloaded_case.get("also_affects", ""),
            ),
            "is_preloaded": True,
        }

    return get_live_article_component_data(
        selected_value=selected_value,
        case_index=case_index,
        api_key=api_key,
        model=model,
        from_date=from_date,
        to_date=to_date,
        num_results=num_results,
    )
