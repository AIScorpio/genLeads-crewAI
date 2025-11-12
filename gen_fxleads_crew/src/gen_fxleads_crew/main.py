#!/usr/bin/env python
from gen_fxleads_crew.crew import GenFxleadsCrew
import time

leads_details = {
    'expected_num': 3,
    'industry': ["Electronics","Healthcare"],
    'material': "silver",
    'region': "China",
    "offering": "FX options",  # NDF : FX Forward
}

class RateLimitError(Exception):
    pass
    
def call_model(leads_details):
    try:
        # Your actual code to call the model
        response = GenFxleadsCrew().crew().kickoff(inputs=leads_details)
        
        # Check if the response indicates a rate limit error and raise RateLimitError if so
        if 'error' in response and response['error']['code'] == 'rate_limit_exceeded':
            raise RateLimitError(response)
        
        # Process the response if no rate limit error
        print("Model call successful\n")
        return response
    
    except RateLimitError as e:
        # Re-raise the rate limit error to be handled by the retry function
        raise e
    except Exception as e:
        # Handle other potential exceptions from the model call
        print(f"An unexpected error occurred: {e}")
        raise e

def call_crew_with_retry(leads_details):
    try:
        return call_model(leads_details)
    except RateLimitError as e:
        # Extract the wait time from the error message
        wait_time = float(e.args[0]['error']['message'].split('try again in ')[1].split('s.')[0])
        print(f"Rate limit exceeded. Retrying in {wait_time} seconds.")
        time.sleep(70)
        return call_crew_with_retry(leads_details)
    except Exception as e:
         # Handle other potential exceptions from the model call
        time.sleep(70)
        return call_crew_with_retry(leads_details)

# if __name__ == "__main__":
#     call_crew_with_retry(leads_details)

def run():
    call_crew_with_retry(leads_details)