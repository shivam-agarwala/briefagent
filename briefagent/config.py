"""
Central configuration. Everything tunable lives here so no magic numbers are
scattered through the code.

All values can be overridden with environment variables, which is what lets the
eval harness run the same code with different settings.
"""

import os

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# The model name is read from an env var so the provider can be swapped without
# touching code. Run `python -m briefagent.agent --list-models` to see what your
# API key actually has access to, since model names change over time.
# Defaults are a starting guess, not a guarantee. Model names rot: Google
# retired gemini-2.5-flash for new keys in 2026 and returns a 404 naming its
# replacement. Always confirm with --list-models, and override with
# BRIEFAGENT_MODEL rather than editing this.
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "gemini": "gemini-3.8-flash",
    # Everything below speaks the OpenAI chat completions format and is served
    # by briefagent/openai_compat.py. See PROVIDERS there for base URLs and
    # the env var each one reads its key from.
    "groq": "openai/gpt-oss-120b",
    "cerebras": "llama-3.3-70b",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "github": "openai/gpt-4o-mini",
    "ollama": "llama3.1",
    "openai_compat": "",
}

# Providers served by the OpenAI-compatible adapter.
OPENAI_COMPAT_PROVIDERS = {
    "groq",
    "cerebras",
    "openrouter",
    "github",
    "ollama",
    "openai_compat",
}


def get_provider() -> str:
    """Which API to call: "anthropic" or "gemini".

    Gemini is supported because it has a free tier with no credit card, and
    this project should be runnable by someone who cannot pay for credits. The
    agent loop is identical either way. Only llm.py and gemini.py know.
    """
    return os.environ.get("BRIEFAGENT_PROVIDER", "anthropic").strip().lower()


def get_model() -> str:
    env = os.environ.get("BRIEFAGENT_MODEL")
    if env:
        return env
    return DEFAULT_MODELS.get(get_provider(), DEFAULT_MODELS["anthropic"])


def llm_min_interval() -> float:
    """Minimum seconds between model calls.

    Free tiers cap requests per minute and the agent loop fires calls back to
    back, so without pacing the second call of every run gets a 429. Gemini's
    free tier is roughly 10 per minute, so 6.5 seconds is a safe default there.
    """
    env = os.environ.get("BRIEFAGENT_LLM_MIN_INTERVAL")
    if env:
        return float(env)

    provider = get_provider()
    if provider == "gemini":
        return 6.5
    if provider in OPENAI_COMPAT_PROVIDERS:
        from .openai_compat import PROVIDERS

        return float(PROVIDERS[provider]["interval"])
    return 0.0


# Read at import for cheap access. Call llm_min_interval() to re-read the env.
LLM_MIN_INTERVAL_SECONDS = llm_min_interval()


# Max tokens the model may generate in a single turn.
MAX_OUTPUT_TOKENS = int(os.environ.get("BRIEFAGENT_MAX_OUTPUT_TOKENS", "2000"))


# ---------------------------------------------------------------------------
# Loop limits
# ---------------------------------------------------------------------------

# Hard cap on fetch_url calls per company. Prevents a confused model from
# crawling forever.
MAX_TOOL_CALLS = int(os.environ.get("BRIEFAGENT_MAX_TOOL_CALLS", "6"))

# Hard cap on model turns. Always a bit above MAX_TOOL_CALLS so the model still
# gets a turn to write the brief after its last fetch.
MAX_STEPS = int(os.environ.get("BRIEFAGENT_MAX_STEPS", "10"))

# Characters of cleaned page text handed back to the model per fetch.
FETCH_CHAR_CAP = int(os.environ.get("BRIEFAGENT_FETCH_CHAR_CAP", "6000"))

# How many claims the model is allowed per section.
MAX_CLAIMS_PER_SECTION = 3


# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

HTTP_TIMEOUT_SECONDS = float(os.environ.get("BRIEFAGENT_HTTP_TIMEOUT", "15"))

# Timeout for a single model call. Without this the Gemini SDK waits forever,
# and a rate limited call looks like a frozen program rather than an error.
LLM_TIMEOUT_SECONDS = float(os.environ.get("BRIEFAGENT_LLM_TIMEOUT", "90"))

# Thinking token budget for models that support it. 0 disables thinking.
#
# Default off, for two reasons. Thinking tokens count against max_output_tokens
# on Gemini 3.x, so a thinking model can spend the whole budget reasoning and
# return no tool call at all. And the experiment is about page selection, not
# about how long the model deliberates, so leaving it off keeps the systems
# comparable. Set -1 for automatic.
THINKING_BUDGET = int(os.environ.get("BRIEFAGENT_THINKING_BUDGET", "0"))

USER_AGENT = os.environ.get(
    "BRIEFAGENT_USER_AGENT",
    "briefagent/0.1 (research prototype; +https://github.com/)",
)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# Prices are NOT hardcoded. Model pricing changes, and a README full of invented
# dollar figures is worse than no figures at all. Set these two env vars to the
# current published price per million tokens for whatever model you are running.
# If they are unset, cost is reported as 0.0 and marked unpriced in the trace.
PRICE_IN_PER_MTOK = float(os.environ.get("BRIEFAGENT_PRICE_IN_PER_MTOK", "0"))
PRICE_OUT_PER_MTOK = float(os.environ.get("BRIEFAGENT_PRICE_OUT_PER_MTOK", "0"))


def prices_configured() -> bool:
    return PRICE_IN_PER_MTOK > 0 or PRICE_OUT_PER_MTOK > 0


def estimate_cost(tokens_in: int, tokens_out: int) -> float:
    """Dollar estimate for one model call."""
    return (
        tokens_in / 1_000_000 * PRICE_IN_PER_MTOK
        + tokens_out / 1_000_000 * PRICE_OUT_PER_MTOK
    )


# ---------------------------------------------------------------------------
# Budget caps
#
# A run that costs more than you expected is a bug, not a surprise. These abort
# the run cleanly and still emit whatever brief the model can produce.
# ---------------------------------------------------------------------------

BUDGET_MAX_TOKENS = int(os.environ.get("BRIEFAGENT_BUDGET_MAX_TOKENS", "60000"))

# Only enforced when prices are configured, since otherwise cost is always 0.
BUDGET_MAX_USD = float(os.environ.get("BRIEFAGENT_BUDGET_MAX_USD", "0.50"))


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------

# 6 rather than 4. Most retries here are not waiting for a server to recover,
# they are resampling a model that broke the tool calling protocol. That is a
# dice roll, so extra attempts convert directly into completed runs.
RETRY_ATTEMPTS = int(os.environ.get("BRIEFAGENT_RETRY_ATTEMPTS", "6"))
RETRY_BASE_DELAY = float(os.environ.get("BRIEFAGENT_RETRY_BASE_DELAY", "1.0"))
RETRY_MAX_DELAY = float(os.environ.get("BRIEFAGENT_RETRY_MAX_DELAY", "20.0"))


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

# Seconds between outbound requests. One request per second, politely.
RATE_LIMIT_SECONDS = float(os.environ.get("BRIEFAGENT_RATE_LIMIT_SECONDS", "1.0"))

RESPECT_ROBOTS = os.environ.get("BRIEFAGENT_RESPECT_ROBOTS", "1") != "0"

MAX_REDIRECTS = int(os.environ.get("BRIEFAGENT_MAX_REDIRECTS", "5"))
