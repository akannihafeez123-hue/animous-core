import requests
from bs4 import BeautifulSoup

def parse_platform(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        blocks = soup.find_all(["p", "h1", "h2", "h3", "li", "code"])
        content = "\n".join(b.get_text(strip=True) for b in blocks if b.get_text(strip=True))
        return {"url": url, "content": content[:5000]}
    except Exception as e:
        print(f"❌ Failed to parse {url}: {e}")
        return {"url": url, "content": ""}
