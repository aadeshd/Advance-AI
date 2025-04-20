import requests

def get_linkedin_info(url: str, mock:bool = True):

    profile_url = url

    if mock:

        response = requests.get(
            profile_url,
            timeout=10
        )

        data = response.json()

        return data
    
    else:
        pass
        # Add logic to scrape linkedin data
