# This file is modified from the biomni package
# https://github.com/openbmb/Biomni
# https://github.com/openbmb/Biomni/blob/main/biomni/llm.py
# This is used to get the llm instance based on the model name and source.

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

if TYPE_CHECKING:
    from BaseAgent.config import AgentConfig

SourceType = Literal["OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom"]
ALLOWED_SOURCES: set[str] = set(SourceType.__args__)

# Configure the rate limiter
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,  # Allows one request every 1 second
    check_every_n_seconds=1,  # Checks every second if a request is allowed
    max_bucket_size=1,  # Controls the maximum burst size of requests
)


@dataclass
class UsageMetrics:
    """Unified usage metrics returned by language model clients."""

    provider: SourceType
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    currency: str | None = "USD"
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Convert the metrics to a serialisable dictionary."""

        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "currency": self.currency if self.cost is not None else None,
            "details": self.details,
        }


def _ensure_mapping(value: Any) -> Mapping[str, Any] | None:
    if hasattr(value, "items"):
        try:
            return dict(value.items())
        except Exception:  # noqa: BLE001
            try:
                return dict(value)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                return None
    return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_response_metadata(response: Any) -> Mapping[str, Any] | None:
    # BaseMessage object with response_metadata or metadata attr: LangChain objects
    if isinstance(response, BaseMessage):
        meta = getattr(response, "response_metadata", None)
        mapping = _ensure_mapping(meta)
        if mapping is not None:
            return mapping
        meta = getattr(response, "metadata", None)
        mapping = _ensure_mapping(meta)
        if mapping is not None:
            return mapping

    mapping = None
    # Plain dict response: Raw client JSON or normalized dict
    if isinstance(response, Mapping):
        mapping = _ensure_mapping(response.get("response_metadata"))
        if mapping is not None:
            return mapping
        return _ensure_mapping(response)

    # Custom object with response_metadata attr: Custom or SDK objects
    if hasattr(response, "response_metadata"):
        mapping = _ensure_mapping(getattr(response, "response_metadata"))
        if mapping is not None:
            return mapping

    return None


def _extract_usage_metrics_unified(provider: SourceType, model: str | None, metadata: Mapping[str, Any]) -> UsageMetrics:
    """Unified usage metrics extraction that handles all provider-specific field variations."""
    raw_metadata = dict(metadata)
    
    # Extract model name with provider-specific fallbacks
    model_name = (
        raw_metadata.get("model")
        or raw_metadata.get("model_name")
        or raw_metadata.get("modelId")  # Bedrock
        or model
    )

    # Get usage dict with fallbacks
    token_usage = _ensure_mapping(raw_metadata.get("token_usage"))
    if token_usage is None:
        token_usage = _ensure_mapping(raw_metadata.get("usage"))

    def lookup(*keys: str) -> Any:
        current: Any = raw_metadata
        for key in keys:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
        return current

    if token_usage is None:
        token_usage = _ensure_mapping(lookup("response", "usage"))

    usage_dict = dict(token_usage) if token_usage is not None else {}

    # Extract input tokens - check all provider variations
    input_tokens = _coerce_int(
        raw_metadata.get("prompt_eval_count")  # Ollama
        or usage_dict.get("prompt_tokens")  # OpenAI
        or usage_dict.get("input_tokens")  # Anthropic/others
        or usage_dict.get("inputTokens")  # Bedrock camelCase
        or raw_metadata.get("inputTokens")  # Bedrock top-level
        or raw_metadata.get("input_tokens")
        or lookup("usage", "prompt_tokens")
        or lookup("usage", "input_tokens")
    )
    
    # Extract output tokens - check all provider variations
    output_tokens = _coerce_int(
        raw_metadata.get("eval_count")  # Ollama
        or usage_dict.get("completion_tokens")  # OpenAI
        or usage_dict.get("output_tokens")  # Anthropic/others
        or usage_dict.get("outputTokens")  # Bedrock camelCase
        or raw_metadata.get("outputTokens")  # Bedrock top-level
        or raw_metadata.get("output_tokens")
        or lookup("usage", "completion_tokens")
        or lookup("usage", "output_tokens")
    )
    
    # Extract total tokens - check all provider variations
    total_tokens = _coerce_int(
        usage_dict.get("total_tokens")
        or usage_dict.get("totalTokens")  # Bedrock camelCase
        or raw_metadata.get("totalTokens")  # Bedrock top-level
        or raw_metadata.get("total_tokens")
        or lookup("usage", "total_tokens")
    )

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    # Extract cost - check all provider variations
    cost = _coerce_float(
        usage_dict.get("total_cost")
        or usage_dict.get("cost")
        or usage_dict.get("totalCost")  # Bedrock camelCase
        or usage_dict.get("cache_creation_input_tokens_cost")  # Anthropic caching
        or raw_metadata.get("total_cost")
        or raw_metadata.get("cost")
        or raw_metadata.get("totalCost")  # Bedrock top-level
        or lookup("usage", "total_cost")
        or lookup("usage", "cost")
        or lookup("usage", "estimated_cost")
    )

    currency = raw_metadata.get("currency") or usage_dict.get("currency")

    return UsageMetrics(
        provider=provider,
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost=cost,
        currency=currency,
        details={"response_metadata": raw_metadata},
    )


def extract_usage_metrics(
    provider: SourceType,
    response: Any,
    *,
    model: str | None = None,
) -> UsageMetrics | None:
    """Extract normalised usage metrics from a language model response."""

    metadata = _get_response_metadata(response)
    if metadata is None:
        return None

    metrics: UsageMetrics = _extract_usage_metrics_unified(provider, model, metadata)
    return metrics


def _detect_source(model: str, base_url: str | None) -> SourceType:
    """
    Detect the source of the model based on the model name and base URL. This function is not catching all the cases.
    Args:
        model (str): The model name to detect the source of.
        base_url (str): The base URL of the model.
    Returns:
        SourceType: The source of the model.
    """
    lower_model = model.lower()

    prefix_rules: list[tuple[str | tuple[str, ...], SourceType]] = [
        ("claude-", "Anthropic"),
        ("gpt-oss", "Ollama"),
        ("gpt-", "OpenAI"),
        ("azure-claude-", "AnthropicFoundry"),
        ("azure-gpt-", "AzureOpenAI"),
        ("gemini-", "Gemini"),
    ]

    for prefix, source in prefix_rules:
        if model.startswith(prefix):
            return source

    if base_url is not None:
        return "Custom"

    ollama_markers = {
        "llama",
        "mistral",
        "qwen",
        "gemma",
        "phi",
        "dolphin",
        "orca",
        "vicuna",
        "deepseek",
    }
    if "/" in model or any(marker in lower_model for marker in ollama_markers):
        return "Ollama"

    raise ValueError("Unable to determine model source. Please specify 'source' parameter.")


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    source: SourceType | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    config: Optional["AgentConfig"] = None,
) -> tuple[SourceType, BaseChatModel]:
    """
    Get a language model instance based on the specified model name and source.
    This function supports models from OpenAI, Azure OpenAI, Anthropic, Ollama, Gemini, Bedrock, and custom model serving.
    Args:
        model (str): The model name to use
        temperature (float): Temperature setting for generation
        stop_sequences (list): Sequences that will stop generation
        source (str): Source provider: "OpenAI", "AzureOpenAI", "AnthropicFoundry", "Anthropic", "Ollama", "Gemini", "Bedrock", or "Custom"
                      If None, will attempt to auto-detect from model name
        base_url (str): The base URL for custom model serving (e.g., "http://localhost:8000/v1"), default is None
        api_key (str): The API key for the custom llm
        config (BiomniConfig): Optional configuration object. If provided, unspecified parameters will use config values
    """
    # Use config values for any unspecified parameters
    if config is not None:
        model = model or config.llm_model
        temperature = temperature if temperature is not None else config.temperature
        source = source or config.source
        base_url = base_url or config.base_url
        api_key = api_key or config.api_key

    # Use defaults if still not specified
    if model is None:
        model = "claude-sonnet-4-5-20250929"
    if temperature is None:
        temperature = 0.7
    # Auto-detect source from model name if not specified
    if source is None:
        env_source = os.getenv("LLM_SOURCE")
        if env_source in ALLOWED_SOURCES:
            source = env_source
        else:
            source = _detect_source(model, base_url)

    # Todo: move chatmodel configuration up here, like temperature, stop_sequences, etc.
    # kwargs = {
    #         "model": model,
    #         "temperature": temperature,
    #         "stop_sequences": stop_sequences,
    #         "rate_limiter": rate_limiter,
    #     }
    # if api_key is not None:
    #     kwargs["api_key"] = api_key
    # if base_url is not None:
    #     kwargs["base_url"] = base_url

    # Create appropriate model based on source
    if source == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for OpenAI models. Install with: pip install langchain-openai"
            )

        # Tune parameters for gpt-5
        if model.startswith("gpt-5"):
            print(f"Tuning parameters for gpt-5: temperature=1.0, stop_sequences=None")
            temperature = 1.0
            stop_sequences = None

        # Build kwargs dict, only including api_key if it's not None
        # This prevents ChatOpenAI from using an async callable when api_key is None
        kwargs = {
            "model": model,
            "temperature": temperature,
            "stop_sequences": stop_sequences,
            "rate_limiter": rate_limiter,
        }
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        

        print(f"kwargs: {kwargs}")
        return source, ChatOpenAI(**kwargs)
    elif source == "AzureOpenAI":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Azure OpenAI models. Install with: pip install langchain-openai"
            )
        model = model.replace("azure-", "")
        return source, AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=model,
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            temperature=temperature,
            rate_limiter=rate_limiter,
        )
    elif source == "AnthropicFoundry":
        from langchain_anthropic import ChatAnthropic
        from anthropic import AnthropicFoundry, AsyncAnthropicFoundry
        
        azure_endpoint = base_url or os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
        azure_api_key = api_key or os.getenv("ANTHROPIC_FOUNDRY_API_KEY")
        
        # Create ChatAnthropic instance with dummy values
        # These will be overridden by our custom client
        chat = ChatAnthropic(
            model=model,
            api_key=azure_api_key,  # Required by ChatAnthropic, but AnthropicFoundry will handle auth
            base_url=azure_endpoint,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            rate_limiter=rate_limiter,
        )
        
        # Override both sync and async clients with AnthropicFoundry
        # This is necessary because ChatAnthropic expects standard Anthropic auth,
        # but Azure Foundry uses different auth headers (api-key instead of x-api-key)
        
        # Create a simple caching wrapper to mimic cached_property behavior
        _sync_client_cache = {}
        _async_client_cache = {}
        
        def _get_sync_client(self):
            if 'client' not in _sync_client_cache:
                _sync_client_cache['client'] = AnthropicFoundry(
                    api_key=azure_api_key,
                    base_url=azure_endpoint,
                    max_retries=self.max_retries,
                    default_headers=self.default_headers,
                )
            return _sync_client_cache['client']
        
        def _get_async_client(self):
            if 'client' not in _async_client_cache:
                _async_client_cache['client'] = AsyncAnthropicFoundry(
                    api_key=azure_api_key,
                    base_url=azure_endpoint,
                    max_retries=self.max_retries,
                    default_headers=self.default_headers,
                )
            return _async_client_cache['client']
        
        # Create property objects that properly handle the descriptor protocol
        chat.__class__._client = property(_get_sync_client)
        chat.__class__._async_client = property(_get_async_client)

        # todo: remove
        print("Creating AnthropicFoundry model: {model}")
        return source, chat
    elif source == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-anthropic package is required for Anthropic models. Install with: pip install langchain-anthropic"
            )

        # enable prompt caching for Claude 3+ models
        # https://docs.anthropic.com/en/docs/prompt-caching
        supported_prompt_caching_models = ["claude-3-haiku-20240307", "claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-opus-4-1-20250805", "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]
        enable_prompt_caching = model in supported_prompt_caching_models
        extra_kwargs = {}
        if enable_prompt_caching:
            extra_kwargs["default_headers"] = {
                "anthropic-beta": "prompt-caching-2024-07-31"
            }

        return source, ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            rate_limiter=rate_limiter,
            **extra_kwargs,
        )

    elif source == "Gemini":
        # If you want to use ChatGoogleGenerativeAI, you need to pass the stop sequences upon invoking the model.
        # return ChatGoogleGenerativeAI(
        #     model=model,
        #     temperature=temperature,
        #     google_api_key=api_key,
        # )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Gemini models. Install with: pip install langchain-openai"
            )
        return source, ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            stop_sequences=stop_sequences,
            rate_limiter=rate_limiter,
        )

    elif source == "Groq":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for Groq models. Install with: pip install langchain-openai"
            )
        return source, ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            stop_sequences=stop_sequences,
            rate_limiter=rate_limiter,
        )

    elif source == "Ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-ollama package is required for Ollama models. Install with: pip install langchain-ollama"
            )
        
        # Create the base LLM
        base_llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_ctx=8192,  # Context window size
            num_predict=-1,  # No prediction limit (helps avoid truncation)
            rate_limiter=rate_limiter,
        )
        
        # Only apply wrapper for gpt-oss models with tool calling issues
        model_lower = model.lower()
        if "gpt-oss" in model_lower:
            print(f"⚠️  Warning: {model} has tool calling behavior in Ollama.")
            print("   BaseAgent will extract code from tool call errors and wrap it in <execute> tags.")
            print("   For better experience, consider using: 'llama3.2:3b' or 'qwen2.5:7b'")
            
            # Create a wrapper class that intercepts invoke calls
            class OllamaWithToolCallExtraction:
                """Wrapper that extracts code from Ollama tool call parsing errors for gpt-oss models."""
                
                def __init__(self, base_llm):
                    self._base_llm = base_llm
                    # Copy essential attributes
                    self.model_name = getattr(base_llm, 'model', None) or getattr(base_llm, 'model_name', None)
                
                def invoke(self, input, config=None, **kwargs):
                    """Intercept tool call errors and extract the raw code."""
                    try:
                        return self._base_llm.invoke(input, config=config, **kwargs)
                    except Exception as e:
                        error_msg = str(e)
                        # Check if this is a tool call parsing error with raw content
                        if "error parsing tool call" in error_msg and "raw=" in error_msg:
                            import re
                            from langchain_core.messages import AIMessage
                            
                            # Extract the raw field: raw='CONTENT'
                            match = re.search(r"raw='(.*?)'(?:,| \()", error_msg, re.DOTALL)
                            if match:
                                raw_content = match.group(1)
                                # Unescape newlines and quotes
                                raw_content = raw_content.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
                                
                                # Wrap in <execute> tags and return
                                wrapped_content = f"<execute>\n{raw_content}\n</execute>"
                                return AIMessage(content=wrapped_content)
                        
                        # If we can't extract, re-raise the original error
                        raise
                
                def __getattr__(self, name):
                    """Forward all other attribute access to the base LLM."""
                    return getattr(self._base_llm, name)
            
            # Wrap the LLM for gpt-oss models only
            return source, OllamaWithToolCallExtraction(base_llm)
        else:
            # For other Ollama models, return unwrapped
            return source, base_llm

    elif source == "Bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-aws package is required for Bedrock models. Install with: pip install langchain-aws"
            )
        return source, ChatBedrock(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            rate_limiter=rate_limiter,
        )

    elif source == "Custom":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-openai package is required for custom models. Install with: pip install langchain-openai"
            )
        # Custom LLM serving such as SGLang. Must expose an openai compatible API.
        assert base_url is not None, "base_url must be provided for customly served LLMs"
        if model.startswith("gpt-5"):
            print(f"Tuning parameters for gpt-5.1: temperature=1.0, stop_sequences=None")
            temperature = 1.0
            stop_sequences = None

        return source, ChatOpenAI(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            base_url=base_url,
            api_key=api_key,
            rate_limiter=rate_limiter,
        )

    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', 'Gemini', 'Groq', 'Bedrock', or 'Ollama'"
        )