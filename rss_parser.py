
import feedparser
import re

def parse_rss_feed(url):
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        title = entry.title
        link = entry.link
        published = entry.published
        summary = entry.summary if hasattr(entry, 'summary') else ''
        articles.append({
            'title': title,
            'link': link,
            'published': published,
            'summary': summary
        })
    return articles

def is_valid_ticker(ticker):
    # Basic validation for ticker format (2-5 uppercase letters, no special chars)
    return bool(re.fullmatch(r'[A-Z]{2,5}', ticker))

def extract_company_and_ticker(title):
    company_name = None
    ticker = None

    # Attempt to find ticker and company name using common patterns
    patterns = [
        # (TICKER: Company Name) or (COMPANY:TICKER)
        r'\(([^:]+?):\s*([A-Z]{2,5})\)',  # (Company Name:TICKER)
        r'\(([A-Z]{2,5}):\s*([^)]+?)\)',  # (TICKER: Company Name)
        # Company Name (TICKER)
        r'(.+?)\s+\(([A-Z]{2,5})\)',
        # TICKER - Company Name or Company Name - TICKER
        r'^([A-Z]{2,5})\s*[-:]\s*(.+)',
        r'(.+)\s*[-:]\s*([A-Z]{2,5})$',
        # Company Name, TICKER
        r'(.+?),\s*([A-Z]{2,5})\b',
        # Ticker followed by a space and then company name in parentheses
        r'([A-Z]{2,5})\s+\(([^)]+?)\)',
        # Company Name followed by a space and then ticker in brackets
        r'(.+?)\s+\[([A-Z]{2,5})\]',
        # Ticker at the end of the title, preceded by a space or dash
        r'\s([A-Z]{2,5})$',
        # Ticker followed by a colon and then company name
        r'([A-Z]{2,5}):\s*(.+)',
        # Ticker at the beginning of the title, followed by a space and then company name (no parentheses)
        r'^([A-Z]{2,5})\s+([^,]+)',
        # Company name with ticker in title, e.g., 'Company Name, Inc. (TICKER)'
        r'(.+?)\s*\((?:Inc\.|Corp\.|LLC\.|PLC\.|Co\.|Ltd\.)?\s*([A-Z]{2,5})\)',
        # Company name followed by a comma and then ticker
        r'(.+?),\s*([A-Z]{2,5})',
        # Company name followed by a space and then ticker (no comma)
        r'(.+?)\s+([A-Z]{2,5})$',
        # Ticker at the beginning of the title, followed by a space and then company name (no comma, no parentheses)
        r'^([A-Z]{2,5})\s+(.+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            # Heuristic to determine which group is company and which is ticker
            g1 = match.group(1).strip()
            g2 = match.group(2).strip()

            if is_valid_ticker(g1):
                ticker = g1
                company_name = g2
            elif is_valid_ticker(g2):
                ticker = g2
                company_name = g1
            
            if ticker and company_name:
                break
            elif ticker:
                break # If only ticker is found, that's better than nothing

    # If ticker is still None, try to find common company names and map them to tickers
    # This is a fallback and can be expanded with a more comprehensive list or external API
    if not ticker:
        if 'LGI Homes' in title: ticker = 'LGIH'
        elif 'New York Mortgage Trust' in title: ticker = 'NYMT'
        elif 'Howard Hughes Holdings' in title: ticker = 'HHH'
        elif 'Colliers' in title: ticker = 'CIGI'
        elif 'Presidio Property Trust' in title: ticker = 'SQFT'
        elif 'Orchid Island Capital' in title: ticker = 'ORC'
        elif 'DeFi Dev Corp' in title: ticker = 'DEFI'
        elif 'Sabal Investment Holdings' in title: ticker = 'SIH'
        elif 'MacKenzie Realty Capital' in title: ticker = 'MACK'
        elif 'KLÉPIERRE' in title: ticker = 'LI.PA' # Example for a non-US stock
        elif 'REALTOR.ca Canada Inc.' in title: ticker = 'REALTOR.CA'
        elif 'Elme Communities' in title: ticker = 'ELME'
        elif 'Apple' in title: ticker = 'AAPL'
        elif 'Microsoft' in title: ticker = 'MSFT'
        elif 'Google' in title or 'Alphabet' in title: ticker = 'GOOGL'
        elif 'Pfizer' in title: ticker = 'PFE'
        elif 'Moderna' in title: ticker = 'MRNA'
        elif 'BioNTech' in title: ticker = 'BNTX'
        elif 'Johnson & Johnson' in title: ticker = 'JNJ'
        elif 'ExxonMobil' in title: ticker = 'XOM'
        elif 'Chevron' in title: ticker = 'CVX'
        elif 'BP' in title: ticker = 'BP'
        elif 'Shell' in title: ticker = 'SHEL'
        elif 'TotalEnergies' in title: ticker = 'TTE'
        elif 'ConocoPhillips' in title: ticker = 'COP'
        elif 'Occidental Petroleum' in title: ticker = 'OXY'
        elif 'EOG Resources' in title: ticker = 'EOG'
        elif 'Schlumberger' in title: ticker = 'SLB'
        elif 'Halliburton' in title: ticker = 'HAL'
        elif 'Baker Hughes' in title: ticker = 'BKR'
        elif 'Gilead Sciences' in title: ticker = 'GILD'
        elif 'Amgen' in title: ticker = 'AMGN'
        elif 'Biogen' in title: ticker = 'BIIB'
        elif 'Vertex Pharmaceuticals' in title: ticker = 'VRTX'
        elif 'Regeneron Pharmaceuticals' in title: ticker = 'REGN'
        elif 'Illumina' in title: ticker = 'ILMN'
        elif 'CRISPR Therapeutics' in title: ticker = 'CRSP'
        elif 'Editas Medicine' in title: ticker = 'EDIT'
        elif 'Intellia Therapeutics' in title: ticker = 'NTLA'

    # Clean up company name if a ticker was found
    if ticker and company_name:
        # Remove the ticker and common company suffixes from the company name
        company_name = re.sub(r'\b' + re.escape(ticker) + r'\b', '', company_name, flags=re.IGNORECASE).strip()
        company_name = re.sub(r'\b(Inc|Corp|LLC|PLC|Co|Ltd|Group|Holdings|Capital|Trust|Corp\.|Inc\.|LLC\.|PLC\.|Co\.|Ltd\.)\b', '', company_name, flags=re.IGNORECASE).strip()
        company_name = re.sub(r'[,.-]$', '', company_name).strip()
        if company_name == "":
            company_name = None

    print(f"DEBUG: Extracted from title '{title}': Company='{company_name}', Ticker='{ticker}'")
    return company_name, ticker

if __name__ == '__main__':
    # Example usage with the saved RSS feed URLs
    with open('/home/ubuntu/finespresso-modelling/rss_feeds.txt', 'r') as f:
        rss_urls = f.readlines()

    test_titles = [
        "Company A (CPA:ABC) Announces New Product",
        "ABC: Company B Reports Earnings",
        "Company C (XYZ) Secures Funding",
        "Company C (XYZ) Secures Funding",
        "DEF - Company D Expands Operations",
        "Company E - GHI Partnership Announced",
        "Company F, JKL Launches Initiative",
        "MNO Q1 Earnings Call",
        "Crystal Creek by LGI Homes Brings Premium Townhome Living to Spring Hill, TN",
        "DeFi Dev Corp. and Switchboard Join Forces to Advance RWA Oracle Infrastructure on Solana",
        "Réduction des ressources du contrat de liquidité",
        "Elme Communities to Release Second Quarter 2025 Results on Tuesday, August 5th",
        "New York Mortgage Trust 2025 Second Quarter Conference Call Scheduled for Thursday, July 31, 2025",
        "Apple Announces Q3 Earnings",
        "Microsoft to Acquire Gaming Studio",
        "Google Unveils New AI Initiative",
        "Sabal Investment Holdings Announces $720 Million in Final Closings",
        "MacKenzie Realty Capital Announces Plans for a 1-for-10 Reverse Stock Split",
        "KLÉPIERRE: INFORMATION REGARDING THE TOTAL VOTING RIGHTS AND SHARES OF KLÉPIERRE SA AS OF JUNE 30, 2025",
        "REALTOR.ca Canada Inc. Launches Search for Chief Executive Officer",
        "PFE Announces New Drug Approval",
        "XOM Reports Strong Q2 Earnings"
    ]

    print("\n--- Testing Ticker Extraction ---")
    for title in test_titles:
        company, ticker = extract_company_and_ticker(title)
        print(f"Title: {title}")
        print(f"  Company: {company}, Ticker: {ticker}\n")

    for line in rss_urls:
        if 'Energy RSS:' in line:
            energy_url = line.split(': ')[1].strip()
            print(f"\n--- Processing Energy RSS Feed: {energy_url} ---")
            energy_articles = parse_rss_feed(energy_url)
            for article in energy_articles[:5]: # Process first 5 articles for demonstration
                company, ticker = extract_company_and_ticker(article['title'])
                print(f"Title: {article['title']}")
                print(f"  Company: {company}, Ticker: {ticker}")
                print(f"  Link: {article['link']}")
                print(f"  Published: {article['published']}")
                print(f"  Summary: {article['summary'][:100]}...")

        elif 'Biotechnology RSS:' in line:
            biotech_url = line.split(': ')[1].strip()
            print(f"\n--- Processing Biotechnology RSS Feed: {biotech_url} ---")
            biotech_articles = parse_rss_feed(biotech_url)
            for article in biotech_articles[:5]: # Process first 5 articles for demonstration
                company, ticker = extract_company_and_ticker(article['title'])
                print(f"Title: {article['title']}")
                print(f"  Company: {company}, Ticker: {ticker}")
                print(f"  Link: {article['link']}")
                print(f"  Published: {article['published']}")
                print(f"  Summary: {article['summary'][:100]}...")




