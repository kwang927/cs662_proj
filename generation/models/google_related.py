import os
import sys

# Suppress gRPC/ALTS warnings before importing Google libraries
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# Suppress absl logging warnings
import logging
logging.getLogger("absl").setLevel(logging.ERROR)

# Redirect stderr temporarily to suppress the C++ warning
import io
_stderr = sys.stderr
sys.stderr = io.StringIO()

import google.generativeai as genai

# Restore stderr
sys.stderr = _stderr
from google.api_core import exceptions
import time
import threading
from collections import deque

GOOGLE_API_KEY = "replace_with_your_google_gemini_api_key"

# Rate limiting configuration
REQUESTS_PER_MINUTE = 25
TOKENS_PER_MINUTE = 1_000_000

# Thread-safe rate limiter
class RateLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_timestamps = deque()
        self.token_usage = deque()  # (timestamp, token_count)
        self.lock = threading.Lock()

    def _clean_old_entries(self, current_time: float):
        """Remove entries older than 60 seconds."""
        cutoff = current_time - 60.0

        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()

        while self.token_usage and self.token_usage[0][0] < cutoff:
            self.token_usage.popleft()

    def _get_current_token_usage(self) -> int:
        """Get total tokens used in the last minute."""
        return sum(tokens for _, tokens in self.token_usage)

    def wait_if_needed(self, estimated_tokens: int = 0):
        """Wait if we're about to exceed rate limits."""
        with self.lock:
            current_time = time.time()
            self._clean_old_entries(current_time)

            # Check request limit
            while len(self.request_timestamps) >= self.requests_per_minute:
                oldest = self.request_timestamps[0]
                wait_time = 60.0 - (current_time - oldest) + 0.1
                if wait_time > 0:
                    print(f"[RateLimiter] Request limit reached. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                current_time = time.time()
                self._clean_old_entries(current_time)

            # Check token limit
            current_tokens = self._get_current_token_usage()
            while current_tokens + estimated_tokens > self.tokens_per_minute:
                if self.token_usage:
                    oldest_time = self.token_usage[0][0]
                    wait_time = 60.0 - (current_time - oldest_time) + 0.1
                    if wait_time > 0:
                        print(f"[RateLimiter] Token limit reached ({current_tokens:,} tokens). Waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                else:
                    break
                current_time = time.time()
                self._clean_old_entries(current_time)
                current_tokens = self._get_current_token_usage()

    def record_request(self, tokens_used: int = 0):
        """Record a completed request."""
        with self.lock:
            current_time = time.time()
            self.request_timestamps.append(current_time)
            if tokens_used > 0:
                self.token_usage.append((current_time, tokens_used))


# Global rate limiter instance
rate_limiter = RateLimiter(REQUESTS_PER_MINUTE, TOKENS_PER_MINUTE)


def estimate_tokens(text: str) -> int:
    """Rough estimate of tokens (approximately 4 characters per token)."""
    return len(text) // 4 + 1


def call_gemini(user_prompt: str, system_prompt: str = "", model_name: str = "gemini-2.0-flash-lite"):
    """
    Calls the Gemini model using the google.generativeai package.

    Args:
        user_prompt (str): The main input prompt.
        system_prompt (str, optional): System instructions (persona/constraints).
        model_name (str, optional): Model name (e.g., "gemini-2.0-flash-lite", "gemini-2.5-pro-preview", "gemini-3-pro-preview").

    Returns:
        response: The response from the model.
    """
    # Estimate input tokens for rate limiting
    estimated_input = estimate_tokens(user_prompt) + estimate_tokens(system_prompt)

    # Wait if we're approaching rate limits
    rate_limiter.wait_if_needed(estimated_input)

    genai.configure(api_key=GOOGLE_API_KEY)

    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt if system_prompt else None
        )

        response = model.generate_content(user_prompt)

        # Record the request and estimate total tokens used
        total_tokens = estimated_input
        if hasattr(response, 'text'):
            total_tokens += estimate_tokens(response.text)
        rate_limiter.record_request(total_tokens)

        return response

    except exceptions.InvalidArgument as e:
        rate_limiter.record_request(estimated_input)
        return f"Invalid Argument Error: {e}"
    except Exception as e:
        rate_limiter.record_request(estimated_input)
        return f"An error occurred: {e}"


def generate_gemini_response(prompt: str, model: str = "gemini-2.0-flash-lite", temperature: float = 0, max_tokens: int = 1024):
    """
    Pipeline-compatible wrapper for call_gemini.
    Used by generate_model_outputs.py.

    Args:
        prompt (str): The input prompt.
        model (str): Model name.
        temperature (float): Temperature for generation (not used by current API).
        max_tokens (int): Max tokens (not used by current API).

    Returns:
        tuple: (response_text, (0, 0)) - text and token counts placeholder.
    """
    response = call_gemini(user_prompt=prompt, model_name=model)

    if isinstance(response, str):
        # Error message was returned
        return response, (0, 0)

    try:
        return response.text, (0, 0)
    except Exception as e:
        return f"An error occurred: {e}", (0, 0)
