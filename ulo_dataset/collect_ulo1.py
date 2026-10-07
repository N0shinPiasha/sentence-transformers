

import re
import time
from datetime import date
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://unitguides.mq.edu.au"
YEAR = 2026
UNITS_PER_DEPT = 7          # change this to collect more or fewer
DELAY_SECONDS = 1.0         # be polite to the university server
HEADERS = {"User-Agent": "Mozilla/5.0 (COMP8240 student research project)"}

# department name on the website -> broad discipline label for the dataset
DEPARTMENTS = {
    "School of Computing": "Computing",
    "School of Mathematical and Physical Sciences": "Mathematics & Statistics",
    "Department of Marketing": "Business",
    "Department of Accounting and Corporate Governance": "Business",
    "Macquarie Medical School": "Medicine & Health",
    "School of Health Sciences and Nursing": "Medicine & Health",
}

# the 4 guides you found yourself (always included)
SEED_URLS = {
    f"{BASE}/unit_offerings/175413/unit_guide": "Computing",                 # COMP6210
    f"{BASE}/unit_offerings/174470/unit_guide": "Computing",                 # COMP8240
    f"{BASE}/unit_offerings/174962/unit_guide": "Computing",                 # COMP6350
    f"{BASE}/unit_offerings/176625/unit_guide": "Mathematics & Statistics",  # STAT6191
}

CODE_RE = re.compile(r"\b([A-Z]{4}\d{4})\b")
ULO_RE = re.compile(r"^\s*(ULO\s*\d+)\s*[:.\-]?\s*(.+)$", re.S)


def get(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(DELAY_SECONDS)
    return BeautifulSoup(r.text, "html.parser")


def unit_links_for(department):
    """Return {unit_code: guide_url} for the first units listed on page 1."""
    url = f"{BASE}/units/show_year/{YEAR}/{quote(department)}"
    soup = get(url)
    links = {}
    for a in soup.find_all("a", href=True):
        if "/unit_offerings/" in a["href"] and a["href"].endswith("/unit_guide"):
            m = CODE_RE.search(a.get_text())
            if m and m.group(1) not in links:      # one offering per unit
                href = a["href"]
                links[m.group(1)] = href if href.startswith("http") else BASE + href
        if len(links) >= UNITS_PER_DEPT:
            break
    return links


def scrape_guide(url):
    """Return (unit_code, unit_name, [(ulo_number, ulo_text), ...])."""
    soup = get(url)
    title = ""
    for h1 in soup.find_all("h1"):
        if CODE_RE.search(h1.get_text()):
            title = h1.get_text(" ", strip=True)
            break
    m = CODE_RE.search(title)
    code = m.group(1) if m else ""
    name = re.sub(r"^[A-Z]{4}\d{4}\s*[–-]\s*", "", title).strip()

    ulos = []
    for li in soup.find_all("li"):
        hit = ULO_RE.match(li.get_text(" ", strip=True))
        if hit:
            num = hit.group(1).replace(" ", "")
            if num not in [u[0] for u in ulos]:
                ulos.append((num, " ".join(hit.group(2).split())))
    return code, name, ulos


def main():
    targets = dict(SEED_URLS)                       # url -> discipline
    for dept, discipline in DEPARTMENTS.items():
        try:
            for code, url in unit_links_for(dept).items():
                targets.setdefault(url, discipline)
            print(f"Listed units from: {dept}")
        except Exception as e:
            print(f"Could not list {dept}: {e}")

    rows, seen_codes, today = [], set(), date.today().isoformat()
    for url, discipline in targets.items():
        try:
            code, name, ulos = scrape_guide(url)
        except Exception as e:
            print(f"  FAILED {url}: {e}")
            continue
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)
        print(f"  {code:<9} {len(ulos)} ULOs  {name}")
        for num, text in ulos:
            rows.append({
                "unit_code": code, "unit_name": name, "discipline": discipline,
                "ulo_number": num, "ulo_text": text,
                "source_url": url, "date_collected": today,
            })

    df = pd.DataFrame(rows)
    df.insert(0, "ulo_id", range(1, len(df) + 1))
    df.to_csv("ulos.csv", index=False)

    print("\n" + "=" * 50)
    print(f"Units scraped : {df['unit_code'].nunique()}")
    print(f"ULOs collected: {len(df)}")
    print(df.groupby("discipline")["unit_code"].nunique().rename("units"))
    print("Saved to ulos.csv")


if __name__ == "__main__":
    main()
