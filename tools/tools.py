from langchain_community.tools.tavily_search import TavilySearchResults
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def get_profile_url_tavily(name: str):
    """Searches for Linkedin profile page."""
    search = TavilySearchResults()
    res = search.run(f"{name} LinkedIn site:linkedin.com/in/")

    return res[0]["url"]

def scrape_profile_mocked(linkedin_profile_url: str):
    """If the found LinkedIn profile url is storywell(김우정), 
    this provides cached result from gist"""
    return scrape_linkedin_profile(linkedin_profile_url, mock=True)

def scrape_linkedin_profile(linkedin_profile_url: str, mock: bool = False):
    """Scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profiles"""

    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/sinclairjang/c5f0008e14e8a7e89475234bbe21978a/raw/62332a83f7d51b903dd83e8d42315024ef1046d9/scrapin_api_storyswell.json"
        response = requests.get(
            linkedin_profile_url,
            timeout=10,
        )
    else:
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "apikey": os.environ["SCRAPIN_API_KEY"],
            "linkedInUrl": linkedin_profile_url
        }
        response = requests.get(
            api_endpoint,
            params=params,
            timeout=10,
        )

    data = response.json().get("person")
    data = {
        k: v 
        for k, v in data.items()
        if v not in ([], "", None) and k not in ["certifications"]
    }

    return data