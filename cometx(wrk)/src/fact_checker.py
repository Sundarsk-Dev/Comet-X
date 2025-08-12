# src/factcheck_module.py

import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json

# Load environment variables from the.env file in the project root
load_dotenv()

# --- Main Function ---
def search_fact_checks(query: str):
    """
    Searches the Google Fact Check Tools API for fact-checks related to a query.

    Args:
        query (str): The text claim or topic to search for.

    Returns:
        list: A list of dictionaries, where each dictionary contains the parsed
              details of a fact-check. Returns an empty list if no results are
              found or if an error occurs.
    """
    # Retrieve the API key from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please ensure you have a.env file with the key in the project root.")
        return []

    # When using the google-api-python-client, you might notice that code editors
    # like VS Code do not provide autocomplete for methods like.claims() or.search().
    # This is expected behavior. The library is built dynamically from a "discovery document"
    # provided by Google at runtime, so the editor cannot know the available API methods
    # ahead of time. You must refer to the official API documentation to know which
    # services and methods are available.
    try:
        # Build the service object for the Fact Check Tools API
        service = build("factchecktools", "v1alpha1", developerKey=api_key)

        # Construct the request to the claims:search endpoint
        request = service.claims().search(query=query)

        # Execute the request and get the response
        response = request.execute()

        # Parse the response and return the results
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
        # Each claim can have multiple reviews from different publishers
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
    print("--- Fact-Check Module Interactive Test ---")
    
    # Prompt the user for a claim to test
    user_query = input("\nEnter the claim you want to fact-check: ")
    
    if not user_query:
        print("No claim entered. Exiting.")
    else:
        print(f"\nSearching for fact-checks on: '{user_query}'")
        
        results = search_fact_checks(user_query)
        
        if results:
            print(f"\nFound {len(results)} fact-check(s):")
            # Pretty-print the results as a JSON object for readability
            print(json.dumps(results, indent=2))
        else:
            print("\nNo fact-checks found for this query.")
    
    print("\n--- Test Complete ---")
