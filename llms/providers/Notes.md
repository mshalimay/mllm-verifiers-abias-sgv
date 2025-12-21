
Due to multiple differences between LLM providers that makes little of the code re-usable between them, it has been more convenient to have a file of functions and globals than an object-oriented solution.

Nonetheless, the general structure is similar among providers.

In the `provider_utils.py` files:
1) Convert prompt messages and generation config from uniform to provider-specific format
2) Call the API 'num_generations' times
3) Convert the API response back to list of messages in uniform format
4) Provider-specific logic for error handling and retry with exponential backoff


In the `client_manager.py` files:
1) Manage API keys and clients for the provider
2) Load API keys from environment variables or files
3) Rotate API keys

In the `error_utils.py` files:
1) Parse error messages from the provider's API errors
2) Extract information such as quota limits and retry delays


In the `prompter.py` files:
1) Convert prompts from uniform format to provider-specific format
2) Convert responses from provider-specific format back to uniform format

In the `file_manager.py` files:
1) Manage file uploads and downloads for the provider, if such functionality exists (e.g., upload large prompts to cloud and reuse them)
2) Handle file format conversions if necessary