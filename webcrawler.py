from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

def get_linkedin_profile_tavily(txt: str):
    load_dotenv()
    # Search for Linkedin or Twitter profile page
    search = TavilySearchResults()
    res = search.run(f"{txt}")
    return res