# This file is modified from the biomni package
# https://github.com/openbmb/Biomni
# https://github.com/openbmb/Biomni/blob/main/biomni/llm.py
# This is used to get the llm instance based on the model name and source.

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional

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
    if isinstance(value, Mapping):
        return value
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
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
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


def _base_metrics(provider: SourceType, model: str | None, metadata: Mapping[str, Any]) -> UsageMetrics:
    raw_metadata = dict(metadata)
    model_name = raw_metadata.get("model") or raw_metadata.get("model_name") or model

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

    input_tokens = _coerce_int(
        usage_dict.get("prompt_tokens")
        or usage_dict.get("input_tokens")
        or raw_metadata.get("input_tokens")
        or lookup("usage", "prompt_tokens")
        or lookup("usage", "input_tokens")
    )
    output_tokens = _coerce_int(
        usage_dict.get("completion_tokens")
        or usage_dict.get("output_tokens")
        or raw_metadata.get("output_tokens")
        or lookup("usage", "completion_tokens")
        or lookup("usage", "output_tokens")
    )
    total_tokens = _coerce_int(
        usage_dict.get("total_tokens")
        or raw_metadata.get("total_tokens")
        or lookup("usage", "total_tokens")
    )

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    cost = _coerce_float(
        usage_dict.get("total_cost")
        or usage_dict.get("cost")
        or raw_metadata.get("total_cost")
        or raw_metadata.get("cost")
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


def _extract_openai_family_usage(provider: SourceType, model: str | None, metadata: Mapping[str, Any]) -> UsageMetrics:
    metrics = _base_metrics(provider, model, metadata)
    return metrics


def _extract_anthropic_usage(provider: SourceType, model: str | None, metadata: Mapping[str, Any]) -> UsageMetrics:
    raw_metadata = dict(metadata)
    model_name = raw_metadata.get("model") or raw_metadata.get("model_name") or model

    usage = _ensure_mapping(raw_metadata.get("usage")) or raw_metadata

    input_tokens = _coerce_int(usage.get("input_tokens") or raw_metadata.get("input_tokens"))
    output_tokens = _coerce_int(usage.get("output_tokens") or raw_metadata.get("output_tokens"))
    total_tokens = _coerce_int(usage.get("total_tokens") or raw_metadata.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    cost = _coerce_float(
        usage.get("total_cost")
        or raw_metadata.get("total_cost")
        or usage.get("cache_creation_input_tokens_cost")
    )

    return UsageMetrics(
        provider=provider,
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost=cost,
        currency=raw_metadata.get("currency"),
        details={"response_metadata": raw_metadata},
    )


def _extract_bedrock_usage(provider: SourceType, model: str | None, metadata: Mapping[str, Any]) -> UsageMetrics:
    raw_metadata = dict(metadata)
    usage = _ensure_mapping(raw_metadata.get("usage")) or {}

    input_tokens = _coerce_int(
        usage.get("inputTokens")
        or usage.get("input_tokens")
        or raw_metadata.get("inputTokens")
        or raw_metadata.get("input_tokens")
    )
    output_tokens = _coerce_int(
        usage.get("outputTokens")
        or usage.get("output_tokens")
        or raw_metadata.get("outputTokens")
        or raw_metadata.get("output_tokens")
    )
    total_tokens = _coerce_int(
        usage.get("totalTokens")
        or usage.get("total_tokens")
        or raw_metadata.get("totalTokens")
        or raw_metadata.get("total_tokens")
    )
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    cost = _coerce_float(
        usage.get("cost")
        or raw_metadata.get("cost")
        or usage.get("totalCost")
        or raw_metadata.get("totalCost")
    )

    model_name = raw_metadata.get("modelId") or raw_metadata.get("model") or model

    return UsageMetrics(
        provider=provider,
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost=cost,
        currency=raw_metadata.get("currency"),
        details={"response_metadata": raw_metadata},
    )


def _extract_ollama_usage(provider: SourceType, model: str | None, metadata: Mapping[str, Any]) -> UsageMetrics:
    raw_metadata = dict(metadata)
    input_tokens = _coerce_int(raw_metadata.get("prompt_eval_count"))
    output_tokens = _coerce_int(raw_metadata.get("eval_count"))
    total_tokens = _coerce_int(raw_metadata.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return UsageMetrics(
        provider=provider,
        model=raw_metadata.get("model") or model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost=_coerce_float(raw_metadata.get("cost")),
        currency=raw_metadata.get("currency"),
        details={"response_metadata": raw_metadata},
    )


def _extract_generic_usage(provider: SourceType, model: str | None, metadata: Mapping[str, Any]) -> UsageMetrics:
    return _base_metrics(provider, model, metadata)


_USAGE_EXTRACTORS: dict[SourceType, Any] = {
    "OpenAI": _extract_openai_family_usage,
    "AzureOpenAI": _extract_openai_family_usage,
    "Gemini": _extract_openai_family_usage,
    "Groq": _extract_openai_family_usage,
    "Custom": _extract_openai_family_usage,
    "Anthropic": _extract_anthropic_usage,
    "Bedrock": _extract_bedrock_usage,
    "Ollama": _extract_ollama_usage,
}


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

    extractor = _USAGE_EXTRACTORS.get(provider, _extract_generic_usage)
    metrics: UsageMetrics = extractor(provider, model, metadata)
    return metrics


def _detect_source(model: str, base_url: str | None) -> SourceType:
    lower_model = model.lower()

    prefix_rules: list[tuple[str | tuple[str, ...], SourceType]] = [
        ("claude-", "Anthropic"),
        ("gpt-oss", "Ollama"),
        ("gpt-", "OpenAI"),
        ("azure-", "AzureOpenAI"),
        ("gemini-", "Gemini"),
    ]

    for prefix, source in prefix_rules:
        if model.startswith(prefix):
            return source

    if "groq" in lower_model:
        return "Groq"

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

    if model.startswith(
        ("anthropic.claude-", "amazon.titan-", "meta.llama-", "mistral.", "cohere.", "ai21.", "us.")
    ):
        return "Bedrock"

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
        source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", or "Custom"
                      If None, will attempt to auto-detect from model name
        base_url (str): The base URL for custom model serving (e.g., "http://localhost:8000/v1"), default is None
        api_key (str): The API key for the custom llm
        config (BiomniConfig): Optional configuration object. If provided, unspecified parameters will use config values
    """
    # Use config values for any unspecified parameters
    if config is not None:
        model = config.llm_model if model is None else model
        temperature = config.temperature if temperature is None else temperature
        source = config.source if source is None else source
        base_url = config.base_url if base_url is None else base_url
        if api_key is None:
            api_key = config.api_key or "EMPTY"

    # Use defaults if still not specified
    if model is None:
        model = "claude-sonnet-4-5-20250929"
    if temperature is None:
        temperature = 0.7
    if api_key is None:
        api_key = "EMPTY"
    # Auto-detect source from model name if not specified
    if source is None:
        env_source = os.getenv("LLM_SOURCE")
        if env_source in ALLOWED_SOURCES:
            source = env_source
        else:
            source = _detect_source(model, base_url)

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
            return source, ChatOpenAI(model=model, temperature=1.0, stop_sequences=None)
        return source, ChatOpenAI(model=model, temperature=temperature, stop_sequences=stop_sequences)

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

    elif source == "Anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(  # noqa: B904
                "langchain-anthropic package is required for Anthropic models. Install with: pip install langchain-anthropic"
            )

        # enable prompt caching for Claude 3+ models
        enable_prompt_caching = model.startswith("claude-3") or model.startswith("claude-4")
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
        return source, ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=8192,
            stop_sequences=stop_sequences,
            base_url=base_url,
            api_key=api_key,
            rate_limiter=rate_limiter,
        )

    else:
        raise ValueError(
            f"Invalid source: {source}. Valid options are 'OpenAI', 'AzureOpenAI', 'Anthropic', 'Gemini', 'Groq', 'Bedrock', or 'Ollama'"
        )