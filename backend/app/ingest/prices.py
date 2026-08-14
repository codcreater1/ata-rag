"""Structured tuition data for akademiata.pl.

Tuition is the single most-asked question ("what is tuition for Computer
Science?") and it is **not** present in the page HTML: the site renders a
JavaScript "tuition calculator" that pulls its numbers from a Google Apps
Script endpoint. A crawl-and-embed pipeline alone therefore cannot answer it.

We fetch that endpoint directly and turn each programme into a small, factual
Markdown card. Those cards are embedded alongside crawled pages, so the normal
retrieval path returns exact figures with a link to the programme page.

Payload shape (as published by the site):

    RAW -> {lang: {city: {mode: [ {k, deg, r10, r12, rekr, wps, ps, ak}, ... ]}}}

    k     programme name          r10   monthly fee, 10 instalments (PLN)
    deg   1 = bachelor, 2 = master r12   monthly fee, 12 instalments (PLN)
    rekr  recruitment fee (PLN)    wps   enrolment fee (PLN)
    ps    programme page URL       ak    internal key

The endpoint URL is published in the page source as ``akademiataPrices``. It is
overridable via ``PRICES_API_URL`` in case the site rotates the deployment.
"""

from __future__ import annotations

import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

DEFAULT_PRICES_URL = (
    "https://script.google.com/macros/s/"
    "AKfycby89Mt7UgeY6jKnq2YQNwumt_CBp46UVd1mbKvxqEkg_46vjGAeN-8lcL_OokQVFnAW/exec"
)

CALCULATOR_PAGE = "https://akademiata.pl/kalkulator-czesnego/"

CITY_NAMES = {"wwa": "Warszawa", "wro": "Wrocław"}
MODE_NAMES = {
    "s": ("stacjonarne", "full-time"),
    "n": ("niestacjonarne", "part-time"),
}
DEGREE_NAMES = {
    1: ("studia I stopnia", "bachelor's studies"),
    2: ("studia II stopnia", "master's studies"),
}


def prices_url() -> str:
    return os.getenv("PRICES_API_URL", DEFAULT_PRICES_URL)


def discover_prices_url(timeout: float = 30) -> str | None:
    """Re-read the endpoint from the live calculator page.

    Used as a self-healing fallback: if the hard-coded deployment id stops
    working, the site itself still tells us the current one.
    """
    try:
        r = httpx.get(CALCULATOR_PAGE, timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "ATA-RAG-Bot/1.0"})
        r.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("could not load calculator page: %s", exc)
        return None

    m = re.search(r'akademiataPrices\s*=\s*\{[^}]*"googleApiUrl"\s*:\s*"([^"]+)"', r.text)
    return m.group(1) if m else None


def fetch_raw(timeout: float = 60) -> dict:
    """Fetch the pricing payload, re-discovering the endpoint if needed."""
    for url in filter(None, [prices_url(), discover_prices_url()]):
        try:
            r = httpx.get(url, timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": "ATA-RAG-Bot/1.0"})
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "RAW" in data:
                return data
            logger.warning("unexpected pricing payload from %s", url)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("pricing fetch failed for %s: %s", url, exc)
    raise RuntimeError("Could not fetch tuition data from any known endpoint")


def _card(entry: dict, *, lang: str, city: str, mode: str) -> tuple[str, str, dict] | None:
    """Render one programme's fees as (title, markdown, metadata)."""
    name = (entry.get("k") or "").strip()
    r10, r12 = entry.get("r10"), entry.get("r12")
    if not name or (r10 is None and r12 is None):
        return None

    city_label = CITY_NAMES.get(city, city)
    mode_pl, mode_en = MODE_NAMES.get(mode, (mode, mode))
    deg_pl, deg_en = DEGREE_NAMES.get(entry.get("deg"), ("", ""))
    url = (entry.get("ps") or CALCULATOR_PAGE).strip()

    if lang == "en":
        title = f"Tuition — {name} ({city_label}, {mode_en})"
        lines = [
            f"# Tuition fees — {name}",
            "",
            f"- Programme: **{name}** ({deg_en})",
            f"- Campus: **{city_label}**",
            f"- Study mode: **{mode_en}**",
        ]
        if r10 is not None:
            lines.append(f"- Tuition paid in 10 instalments: **{r10} PLN per month**")
        if r12 is not None:
            lines.append(f"- Tuition paid in 12 instalments: **{r12} PLN per month**")
        if entry.get("rekr") is not None:
            lines.append(f"- Recruitment fee: **{entry['rekr']} PLN**")
        if entry.get("wps") is not None:
            lines.append(f"- Enrolment fee (wpisowe): **{entry['wps']} PLN**")
        lines += ["", f"Source: {url}"]
    else:
        title = f"Czesne — {name} ({city_label}, {mode_pl})"
        lines = [
            f"# Czesne — {name}",
            "",
            f"- Kierunek: **{name}** ({deg_pl})",
            f"- Miasto: **{city_label}**",
            f"- Tryb studiów: **{mode_pl}**",
        ]
        if r10 is not None:
            lines.append(f"- Czesne płatne w 10 ratach: **{r10} zł miesięcznie**")
        if r12 is not None:
            lines.append(f"- Czesne płatne w 12 ratach: **{r12} zł miesięcznie**")
        if entry.get("rekr") is not None:
            lines.append(f"- Opłata rekrutacyjna: **{entry['rekr']} zł**")
        if entry.get("wps") is not None:
            lines.append(f"- Wpisowe: **{entry['wps']} zł**")
        lines += ["", f"Źródło: {url}"]

    meta = {
        "programme": name,
        "city": city_label,
        "mode": mode_en,
        "degree": entry.get("deg"),
        "monthly_10": r10,
        "monthly_12": r12,
        "recruitment_fee": entry.get("rekr"),
        "enrolment_fee": entry.get("wps"),
        "kind": "tuition",
    }
    return title, "\n".join(lines), meta


def _international_card(entry: dict, *, city: str) -> tuple[str, str, dict] | None:
    """International (English) fees have their own shape and currency.

    Domestic pricing is monthly PLN; the international table is **EUR**, split by
    EU vs non-EU citizenship, quoted per year (``r``) and per semester (``s``).
    Verified against the rendered calculator, which prints e.g. "3600 EUR".
    """
    name = (entry.get("k") or "").strip()
    eu, ne = entry.get("eu") or {}, entry.get("ne") or {}
    if not name or not (eu or ne):
        return None

    city_label = CITY_NAMES.get(city, city)
    spec = (entry.get("s") or "").strip()
    _deg_pl, deg_en = DEGREE_NAMES.get(entry.get("deg"), ("", ""))
    label = f"{name} – {spec}" if spec else name

    title = f"Tuition — {label} ({city_label}, international)"
    lines = [
        f"# Tuition fees — {label}",
        "",
        f"- Programme: **{label}** ({deg_en})",
        f"- Campus: **{city_label}**",
        "- Applies to: **international (English-taught) studies**",
    ]
    if eu.get("r") is not None:
        lines.append(f"- EU citizens: **{eu['r']} EUR per year**"
                     + (f" (**{eu['s']} EUR per semester**)" if eu.get("s") is not None else ""))
    if ne.get("r") is not None:
        lines.append(f"- Non-EU citizens: **{ne['r']} EUR per year**"
                     + (f" (**{ne['s']} EUR per semester**)" if ne.get("s") is not None else ""))
    if entry.get("rekr") is not None:
        lines.append(f"- Recruitment fee: **{entry['rekr']} EUR**")
    if entry.get("wps"):
        lines.append(f"- Enrolment fee: **{entry['wps']} EUR**")

    url = (entry.get("ps") or f"{CALCULATOR_PAGE}").strip()
    lines += ["", f"Source: {url}"]

    meta = {
        "programme": name,
        "specialisation": spec or None,
        "city": city_label,
        "degree": entry.get("deg"),
        "currency": "EUR",
        "eu_per_year": eu.get("r"), "eu_per_semester": eu.get("s"),
        "non_eu_per_year": ne.get("r"), "non_eu_per_semester": ne.get("s"),
        "recruitment_fee": entry.get("rekr"),
        "kind": "tuition_international",
    }
    return title, "\n".join(lines), meta


def _slugify(name: str) -> str:
    trans = str.maketrans("ąćęłńóśźż", "acelnoszz")
    return re.sub(r"[^a-z0-9]+", "-", name.lower().translate(trans)).strip("-")


def _url_score(url: str, programme: str) -> int:
    """Prefer the URL that actually belongs to this programme.

    The payload repeats one fee across every specialisation of a programme, each
    with its own page URL, so picking the first would cite (say) the
    cybersecurity page as the source of the generic Informatyka fee.
    """
    slug = _slugify(programme)
    tail = url.rstrip("/").rsplit("/", 1)[-1] if url else ""
    if not tail:
        return 0
    if tail == slug:
        return 3
    if tail.endswith(f"-{slug}"):     # e.g. wroclaw-informatyka
        return 2
    if slug in tail:
        return 1
    return 0


def build_price_pages() -> list:
    """Return ``crawler.Page`` objects — one per programme/city/mode/language."""
    from app.ingest.crawler import Page

    data = fetch_raw()
    best: dict[tuple, tuple] = {}

    pages: list[Page] = []
    recruitment_fees: set[int] = set()
    enrolment_fees: set[int] = set()

    for lang, cities in (data.get("RAW") or {}).items():
        if not isinstance(cities, dict):
            continue

        for city, block in cities.items():
            # International (English) tuition: a flat list per city, priced in
            # EUR by citizenship. Each specialisation is its own offer, so these
            # are emitted directly rather than de-duplicated by fee.
            if isinstance(block, list):
                for entry in block:
                    if not isinstance(entry, dict):
                        continue
                    built = _international_card(entry, city=city)
                    if not built:
                        continue
                    title, md, meta = built
                    if entry.get("rekr") is not None:
                        recruitment_fees.add(entry["rekr"])
                    if entry.get("wps"):
                        enrolment_fees.add(entry["wps"])
                    pages.append(Page(
                        url=entry.get("ps") or CALCULATOR_PAGE, title=title,
                        markdown=md, language=lang, source_type="tuition",
                        metadata=meta,
                    ))
                continue

            # Domestic (Polish) tuition: {mode: [entries]}, monthly PLN.
            if not isinstance(block, dict):
                continue
            for mode, entries in block.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    built = _card(entry, lang=lang, city=city, mode=mode)
                    if not built:
                        continue
                    _title, _md, meta = built

                    key = (lang, city, mode, meta["programme"],
                           meta["monthly_10"], meta["monthly_12"])
                    score = _url_score(entry.get("ps") or "", meta["programme"])
                    if key not in best or score > best[key][0]:
                        best[key] = (score, entry, lang, city, mode)

    for score, entry, lang, city, mode in best.values():
        title, md, meta = _card(entry, lang=lang, city=city, mode=mode)
        # No specialisation URL matched the programme: cite the calculator page
        # rather than an unrelated specialisation.
        url = entry.get("ps") if score > 0 else CALCULATOR_PAGE
        if score == 0:
            md = re.sub(r"(Source|Źródło): .*$", rf"\1: {CALCULATOR_PAGE}", md)
        pages.append(Page(
            url=url or CALCULATOR_PAGE, title=title, markdown=md,
            language=lang, source_type="tuition", metadata=meta,
        ))

    # A standalone fees card. The recruitment (application) fee is buried in
    # every per-programme tuition card, so a generic "is there a recruitment
    # fee, and how much?" matched the programme pages that merely list it as a
    # required document — never the amount. This card carries the figure on its
    # own so the question retrieves it directly.
    fees_page = _fees_card(recruitment_fees, enrolment_fees)
    if fees_page:
        pages.append(fees_page)

    logger.info("built %d tuition cards", len(pages))
    return pages


def _fee_phrase(values: set[int], unit: str) -> str:
    """'200 EUR' for one value, '200–300 EUR' for a range."""
    vs = sorted(values)
    return f"{vs[0]} {unit}" if len(vs) == 1 else f"{vs[0]}–{vs[-1]} {unit}"


def _fees_card(recruitment: set[int], enrolment: set[int]):
    """A dedicated, directly-retrievable summary of the one-off fees."""
    from app.ingest.crawler import Page

    if not recruitment and not enrolment:
        return None

    lines = [
        "# Recruitment and enrolment fees",
        "",
        ("These are one-off fees paid when you apply, separate from tuition. "
         "Also called the application fee or admission fee."),
        "",
    ]
    if recruitment:
        lines.append(f"- Recruitment fee (application fee): "
                     f"**{_fee_phrase(recruitment, 'EUR')}** for international "
                     f"(English-taught) programmes.")
    if enrolment:
        lines.append(f"- Enrolment fee: **{_fee_phrase(enrolment, 'EUR')}**.")
    lines += ["", f"Source: {CALCULATOR_PAGE}"]

    return Page(
        url=f"{CALCULATOR_PAGE.rstrip('/')}/#fees",
        title="Recruitment and enrolment fees",
        markdown="\n".join(lines), language="en", source_type="fees",
        metadata={"recruitment_fee_eur": sorted(recruitment) or None,
                  "enrolment_fee_eur": sorted(enrolment) or None, "kind": "fees"},
    )
