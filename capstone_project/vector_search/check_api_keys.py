import requests
import os
import time
import sys # Import the sys module

# --- Add project root to sys.path ---
# Calculate the path to the 'capstone_project' directory (two levels up from this file)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add it to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of addition ---

# Now the absolute import should work
from vector_search.config import GEMINI_API_KEYS

# --- Configuration ---
# Check if the imported list is empty
if not GEMINI_API_KEYS:
    print("Error: GEMINI_API_KEYS list imported from config.py is empty.")
    print("Please ensure your config.py file defines the GEMINI_API_KEYS list correctly.")
    exit()

# API endpoint for listing models (a lightweight check)
API_ENDPOINT_LIST_MODELS = "https://generativelanguage.googleapis.com/v1beta/models"

# --- Function to Test a Single Key ---
def test_api_key(api_key):
    """Tests a single Gemini API key by trying to list models."""
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    key_suffix = api_key[-4:] # For logging, don't expose full key

    try:
        response = requests.get(API_ENDPOINT_LIST_MODELS, headers=headers, timeout=15)

        if response.status_code == 200:
            return "OK"
        elif response.status_code == 429:
            return "Rate Limited (429)"
        elif response.status_code == 403:
            try:
                error_detail = response.json().get('error', {}).get('message', 'Forbidden')
                return f"Forbidden/Invalid (403): {error_detail}"
            except requests.exceptions.JSONDecodeError:
                return f"Forbidden/Invalid (403): {response.text}"
        elif response.status_code == 400:
             try:
                error_detail = response.json().get('error', {}).get('message', 'Bad Request')
                # Check specifically for API key validation errors if possible
                if "API key not valid" in error_detail:
                     return f"Invalid API Key (400): {error_detail}"
                else:
                     return f"Bad Request (400): {error_detail}"
             except requests.exceptions.JSONDecodeError:
                return f"Bad Request (400): {response.text}"
        else:
            return f"HTTP Error {response.status_code}: {response.text[:100]}..." # Truncate long errors

    except requests.exceptions.Timeout:
        return "Network Error: Timeout"
    except requests.exceptions.RequestException as e:
        return f"Network Error: {type(e).__name__}"
    except Exception as e:
        return f"Unexpected Error: {e}"

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Testing {len(GEMINI_API_KEYS)} Gemini API Keys from config.py...")
    print("-" * 30)

    results = {}
    for i, key in enumerate(GEMINI_API_KEYS):
        key_suffix = key[-4:]
        print(f"Testing Key {i+1} (...{key_suffix})... ", end="", flush=True)
        status = test_api_key(key)
        results[f"Key ...{key_suffix}"] = status
        print(status)
        time.sleep(0.5) # Small delay between checks

    print("-" * 30)
    print("Summary:")
    for key_info, status in results.items():
        print(f"- {key_info}: {status}")

    print("\nNote: 'Rate Limited (429)' means the key is valid but temporarily blocked due to usage.")
    print("      'Forbidden/Invalid' or other errors might indicate a problem with the key itself or project settings.")