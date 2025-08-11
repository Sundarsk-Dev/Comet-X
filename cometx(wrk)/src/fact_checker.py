# FILE: src/fact_checker.py
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json

# Load environment variables from the.env file in the project root
load_dotenv()

# --- Main Function ---
def check_facts(query: str):
    """
    Searches the Google Fact Check Tools API for fact-checks related to a query.
    
    Args:
        query (str): The text claim or topic to search for.
    
    Returns:
        list: A list of dictionaries, where each dictionary contains the parsed
              details of a fact-check. Returns an empty list if no results are
              found or if an error occurs.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please ensure you have a .env file with the key in the project root.")
        return []

    try:
        service = build("factchecktools", "v1alpha1", developerKey=api_key)
        request = service.claims().search(query=query)
        response = request.execute()
        
        return _parse_response(response)

    except HttpError as e:
        print(f"An HTTP error occurred: {e.content.decode('utf-8')}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

# --- Helper Function for Parsing ---
def _parse_response(response: dict) -> list:
    """
    Parses the raw JSON response from the Fact Check API into a structured list.
    
    Args:
        response (dict): The raw JSON response from the API.
    
    Returns:
        list: A list of parsed fact-check dictionaries.
    """
    parsed_results = []
    claims = response.get("claims", [])

    if not claims:
        return []

    for claim in claims:
        for review in claim.get("claimReview", []):
            parsed_result = {
                "claim_text": claim.get("text"),
                "publisher": review.get("publisher", {}).get("name"),
                "publisher_site": review.get("publisher", {}).get("site"),
                "review_url": review.get("url"),
                "review_title": review.get("title"),
                "rating": review.get("textualRating"),
            }
            parsed_results.append(parsed_result)
    
    return parsed_results

# --- Main execution block for direct script running ---
if __name__ == "__main__":
    print("--- Fact-Check Module Direct Run Test ---")
    test_query = "The Earth is flat"
    print(f"\nSearching for fact-checks on: '{test_query}'")
    results = check_facts(test_query)
    if results:
        print(f"\nFound {len(results)} fact-check(s):")
        print(json.dumps(results, indent=2))
    else:
        print("\nNo fact-checks found for this query.")
    print("\n--- Test Complete ---")
