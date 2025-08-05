import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
GOOGLE_FACT_CHECK_API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

def check_facts(text_content):
    """
    Makes an actual API call to the Google Fact Check Tools API to verify a
    given text claim.

    Args:
        text_content (str): The text claim to be fact-checked.

    Returns:
        dict: A dictionary containing the fact-check result, including a
              label, a source URL, and a brief summary, or None if no result
              is found or an error occurs.
    """
    if not GOOGLE_FACT_CHECK_API_KEY:
        print("Error: Google Fact Check API key not found. Please set it in your .env file.")
        return None

    api_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        'query': text_content,
        'key': GOOGLE_FACT_CHECK_API_KEY,
    }

    try:
        print("  -> Calling Google Fact Check API for external verification...")
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'claims' in data and len(data['claims']) > 0:
            result = data['claims'][0]
            claim_review = result['claimReview'][0]

            fact_check_result = {
                "claim": result['text'],
                "label": claim_review['textualRating'],
                "publisher": claim_review['publisher']['name'],
                "summary": claim_review.get('title', 'No summary available.'),
                "url": claim_review['url']
            }
            return fact_check_result
        else:
            print("  -> No fact-check results found for this claim.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the API request: {e}")
        return None
    except KeyError:
        print("Error parsing Google API response. Unexpected JSON structure.")
        return None


