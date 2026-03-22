"""LLM client abstraction with multiple provider support.

Supports:
- OpenAI: GPT-4o-mini, GPT-4o
- Anthropic: Claude Haiku 4.5
- Google: Gemini 2.5 Flash
- Groq: Llama 3.1 8B (free tier, OpenAI-compatible)
- Hugging Face: Mistral 7B (Router OpenAI-compatible API)
- DeepSeek: deepseek-chat, deepseek-reasoner
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Supported models
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MODEL_4O = "gpt-4o"
# Anthropic: claude-3-haiku is deprecated; use current Haiku (4.5)
ANTHROPIC_MODEL = "claude-haiku-4-5"
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.1-8b-instant"
# Model for HF Inference Router (OpenAI-compatible endpoint). Pass in request body, not URL path.
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"

def _timeout_seconds() -> float:
    try:
        return float(os.getenv("LLM_TIMEOUT_SECONDS", "120").strip())
    except ValueError:
        return 120.0


def _gemini_request_delay() -> float:
    """Seconds to wait between Gemini API calls to avoid 429. Set GEMINI_REQUEST_DELAY_SECONDS (e.g. 1 or 2)."""
    try:
        return max(0.0, float((os.getenv("GEMINI_REQUEST_DELAY_SECONDS") or "0").strip()))
    except ValueError:
        return 0.0


_LOCAL_MODEL_LOCK = threading.Lock()
_LOCAL_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}
_LOCAL_GGUF_CACHE: Dict[str, Any] = {}

# Gemini rate limiting: only one in-flight request and optional delay to avoid 429
_GEMINI_LOCK = threading.Lock()
_GEMINI_LAST_CALL_TIME: float = 0

# OpenAI rate limiting for low-TPM tiers (e.g. GPT-4o free/Tier 1 = 30k TPM).
# Set OPENAI_REQUEST_DELAY_SECONDS in .env (e.g. 8) to pace requests.
_OPENAI_LOCK = threading.Lock()
_OPENAI_LAST_CALL_TIME: float = 0


def _openai_request_delay() -> float:
    """Seconds to wait between OpenAI calls. Prevents 429 on low-TPM tiers."""
    try:
        return max(0.0, float((os.getenv("OPENAI_REQUEST_DELAY_SECONDS") or "0").strip()))
    except ValueError:
        return 0.0


def _local_max_new_tokens() -> int:
    raw = (os.getenv("LOCAL_MAX_NEW_TOKENS") or "").strip() or "256"
    try:
        return max(16, int(raw))
    except ValueError:
        return 256


class LLMClient:
    def __init__(self, provider: str, model: Optional[str] = None) -> None:
        if not provider:
            raise ValueError(
                "LLM provider must be specified. Supported: 'openai', 'anthropic', 'google', 'groq', 'huggingface', 'deepseek', 'local'"
            )
        self.provider = provider.strip().lower()
        self.model = (model or "").strip() or None

    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        return self._provider_response(prompt, max_tokens=max_tokens)

    def _provider_response(self, prompt: str, *, max_tokens: int | None = None) -> str:
        provider_lower = self.provider.lower()
        if provider_lower in ("openai", "openai-4o", "gpt-4o-mini", "gpt4omini", "gpt-4o", "gpt4o"):
            return self._openai_response(prompt, max_tokens=max_tokens)
        elif provider_lower in ("anthropic", "claude", "claude-3-haiku", "claude3haiku"):
            return self._anthropic_response(prompt, max_tokens=max_tokens)
        elif provider_lower in ("google", "gemini", "gemini-1.5-flash", "gemini15flash"):
            return self._gemini_response(prompt, max_tokens=max_tokens)
        elif provider_lower in ("groq", "llama", "llama-3"):
            return self._groq_response(prompt, max_tokens=max_tokens)
        elif provider_lower in ("huggingface", "hf", "zephyr"):
            return self._huggingface_response(prompt, max_tokens=max_tokens)
        elif provider_lower in ("deepseek",):
            return self._deepseek_response(prompt, max_tokens=max_tokens)
        elif provider_lower in ("local", "local_transformers", "local-hf"):
            model_path = (self.model or "").strip()
            if model_path and Path(model_path).suffix.lower() == ".gguf":
                return self._local_gguf_response(prompt)
            return self._local_transformers_response(prompt)
        raise RuntimeError(
            f"LLM provider '{self.provider}' is not configured. "
            "Supported: 'openai', 'anthropic', 'google', 'groq', 'huggingface', 'deepseek', 'local'."
        )

    def _openai_response(self, prompt: str, *, max_tokens: int | None = None) -> str:
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            # Load .env from project root (parent of src/), not cwd
            _root = Path(__file__).resolve().parent.parent.parent
            load_dotenv(_root / ".env")
        except ImportError:
            pass
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it to use GPT-4o-mini (e.g. in PowerShell: $env:OPENAI_API_KEY='your_key')"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "The openai package is not installed in this Python environment. "
                "If you use a virtual environment (venv), activate it, then run: pip install openai"
            )

        client = OpenAI(api_key=api_key, timeout=_timeout_seconds(), max_retries=2)
        from openai import RateLimitError

        global _OPENAI_LAST_CALL_TIME
        delay = _openai_request_delay()
        if delay > 0:
            with _OPENAI_LOCK:
                elapsed = time.time() - _OPENAI_LAST_CALL_TIME
                if elapsed < delay:
                    time.sleep(delay - elapsed)

        last_err: Exception | None = None
        for attempt in range(4):
            try:
                with _OPENAI_LOCK:
                    _OPENAI_LAST_CALL_TIME = time.time()
                response = client.chat.completions.create(
                    model=self.model or OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts ontology concepts (classes) and relations from domain text. Respond with valid JSON only: {\"classes\": [{\"label\", \"definition\", \"synonyms\"}], \"relations\": [{\"source\", \"target\", \"label\"}]}."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens or 4096,
                    temperature=0.3,
                )
                return (response.choices[0].message.content or "").strip()
            except RateLimitError as e:
                last_err = e
                wait = 2 ** attempt * 15
                import logging
                logging.getLogger(__name__).warning(
                    "OpenAI 429 rate limit (attempt %d/4) — waiting %ds: %s",
                    attempt + 1, wait, e,
                )
                time.sleep(wait)
        raise last_err  # type: ignore[misc]

    def _anthropic_response(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Generate response using Anthropic Claude Haiku (4.5)."""
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            _root = Path(__file__).resolve().parent.parent.parent
            load_dotenv(_root / ".env")
        except ImportError:
            pass
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it to use Claude Haiku 4.5 (e.g. in PowerShell: $env:ANTHROPIC_API_KEY='your_key')"
            )
        try:
            from anthropic import Anthropic
        except ImportError:
            raise RuntimeError(
                "The anthropic package is not installed. "
                "Install it with: pip install anthropic"
            )

        # Map deprecated/short ids to current model so saved configs keep working
        model = self.model or ANTHROPIC_MODEL
        if model in ("claude-3-haiku", "claude-3-5-haiku", "claude-3-haiku-20240307", "claude-3-5-haiku-20241022"):
            model = ANTHROPIC_MODEL
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens or 4096,
            temperature=0.3,
            system="You are a helpful assistant that extracts ontology concepts (classes) and relations from domain text. Respond with valid JSON only: {\"classes\": [{\"label\", \"definition\", \"synonyms\"}], \"relations\": [{\"source\", \"target\", \"label\"}]}.",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return (response.content[0].text if response.content else "").strip()

    def _gemini_response(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Generate response using Google Gemini 2.5 Flash. Rate-limited to avoid 429."""
        global _GEMINI_LAST_CALL_TIME
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            _root = Path(__file__).resolve().parent.parent.parent
            load_dotenv(_root / ".env")
        except ImportError:
            pass
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Set it to use Gemini 2.5 Flash (e.g. in PowerShell: $env:GOOGLE_API_KEY='your_key')"
            )
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError(
                "The google-generativeai package is not installed. "
                "Install it with: pip install google-generativeai"
            )

        delay = _gemini_request_delay()
        with _GEMINI_LOCK:
            if delay > 0 and _GEMINI_LAST_CALL_TIME > 0:
                elapsed = time.monotonic() - _GEMINI_LAST_CALL_TIME
                if elapsed < delay:
                    time.sleep(delay - elapsed)
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model or GEMINI_MODEL)
            full_prompt = (
                "You are a helpful assistant that extracts ontology concepts (classes) and relations from domain text. "
                "Respond with valid JSON only: {\"classes\": [{\"label\", \"definition\", \"synonyms\"}], \"relations\": [{\"source\", \"target\", \"label\"}]}.\n\n"
                + prompt
            )
            response = model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": max_tokens or 4096,
                }
            )
            _GEMINI_LAST_CALL_TIME = time.monotonic()
        return (response.text or "").strip()

    def _groq_response(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Generate response using Groq (free tier, OpenAI-compatible API). Model: Llama 3.1 8B."""
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            _root = Path(__file__).resolve().parent.parent.parent
            load_dotenv(_root / ".env")
        except ImportError:
            pass
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com and set it in .env"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "The openai package is required for Groq (OpenAI-compatible). pip install openai"
            )
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=_timeout_seconds(), max_retries=2)
        response = client.chat.completions.create(
            model=self.model or GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts ontology concepts (classes) and relations from domain text. Respond with valid JSON only: {\"classes\": [{\"label\", \"definition\", \"synonyms\"}], \"relations\": [{\"source\", \"target\", \"label\"}]}."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens or 4096,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()

    def _deepseek_response(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Generate response via DeepSeek API (OpenAI-compatible)."""
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            _root = Path(__file__).resolve().parent.parent.parent
            load_dotenv(_root / ".env")
        except ImportError:
            pass
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY environment variable is not set. "
                "Get a key at https://platform.deepseek.com/api_keys and set it in .env"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "The openai package is required for DeepSeek (OpenAI-compatible). pip install openai"
            )
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=_timeout_seconds(),
            max_retries=2,
        )
        model = (self.model or "").strip() or DEEPSEEK_MODEL
        default_limit = 8192 if model == DEEPSEEK_REASONER_MODEL else 2000
        effective_max = max_tokens or default_limit
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts ontology concepts (classes) and relations from domain text. Your final reply must be valid JSON only: {\"classes\": [{\"label\", \"definition\", \"synonyms\"}], \"relations\": [{\"source\", \"target\", \"label\"}], \"hierarchy\": []}. No other text outside the JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=effective_max,
            temperature=0.3,
        )
        message = response.choices[0].message
        # DeepSeek R1 (deepseek-reasoner) returns reasoning_content (<think>) and content (final answer)
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning and isinstance(reasoning, str) and reasoning.strip():
            preview = reasoning.strip()[:500] + ("..." if len(reasoning) > 500 else "")
            print("[DeepSeek R1 reasoning]", preview, flush=True)
        content = (message.content or "").strip()
        # If content is empty but we have reasoning, some R1 responses put final answer after </think>
        if not content and reasoning and isinstance(reasoning, str):
            if "</think>" in reasoning:
                content = reasoning.split("</think>")[-1].strip()
            else:
                content = reasoning.strip()
        return content

    def _huggingface_response(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Generate response via Hugging Face Router OpenAI-compatible API. base_url=v1, model in body."""
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            _root = Path(__file__).resolve().parent.parent.parent
            load_dotenv(_root / ".env")
        except ImportError:
            pass
        api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not api_key:
            raise RuntimeError(
                "HF_TOKEN or HUGGINGFACEHUB_API_TOKEN is not set. "
                "Get a free token at https://huggingface.co/settings/tokens and set it in .env"
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "The openai package is required for Hugging Face Router (OpenAI-compatible). pip install openai"
            )
        # Router OpenAI-compatible endpoint: base URL must be https://router.huggingface.co/v1
        # Model is passed in the request body, not in the path (avoids 404 from /hf-inference/models/.../v1/chat/completions)
        client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=api_key, timeout=_timeout_seconds(), max_retries=2)
        response = client.chat.completions.create(
            model=self.model or HF_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts ontology concepts (classes) and relations from domain text. Respond with valid JSON only: {\"classes\": [{\"label\", \"definition\", \"synonyms\"}], \"relations\": [{\"source\", \"target\", \"label\"}]}."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens or 4096,
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()

    def _local_transformers_response(self, prompt: str) -> str:
        """Run a local Hugging Face Transformers model from disk with 4-bit quantization when available."""
        model_path = (self.model or "").strip()
        if not model_path:
            raise RuntimeError(
                "Local LLM provider requires a model path (e.g. improvements_llm_model or llm_model)."
            )
        p = Path(model_path)
        if not p.exists():
            raise RuntimeError(f"Local model path not found: {model_path}")
        if not (p / "config.json").exists():
            raise RuntimeError(f"Local model folder is missing config.json: {model_path}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "Local model requires: torch, transformers, accelerate, safetensors. "
                "Install with: pip install -r requirements-local-llm.txt"
            ) from e

        use_4bit = (os.getenv("LOCAL_USE_4BIT") or "1").strip().lower() in ("1", "true", "yes", "y")
        allow_cpu = (os.getenv("LOCAL_ALLOW_CPU") or "1").strip().lower() in ("1", "true", "yes", "y")
        if not torch.cuda.is_available() and not allow_cpu:
            raise RuntimeError("CUDA not detected and LOCAL_ALLOW_CPU=0. Set LOCAL_ALLOW_CPU=1 to run on CPU.")

        cache_key = str(p.resolve()).lower()
        with _LOCAL_MODEL_LOCK:
            cached = _LOCAL_MODEL_CACHE.get(cache_key)
            if cached is None:
                print(
                    "[Local LLM] Loading model from",
                    model_path,
                    "(this can take 2–5 minutes for 8B)...",
                    flush=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(p, local_files_only=True, use_fast=True)
                load_kw: Dict[str, Any] = {
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                }
                if use_4bit and torch.cuda.is_available():
                    try:
                        from transformers import BitsAndBytesConfig
                        load_kw["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        load_kw["device_map"] = "auto"
                    except ImportError:
                        use_4bit = False
                if not use_4bit:
                    if torch.cuda.is_available():
                        load_kw["device_map"] = "auto"
                        load_kw["torch_dtype"] = torch.float16
                    else:
                        dtype_name = (os.getenv("LOCAL_CPU_DTYPE") or "float16").strip().lower()
                        load_kw["torch_dtype"] = torch.float16 if dtype_name == "float16" else torch.float32
                        load_kw["device_map"] = None
                model = AutoModelForCausalLM.from_pretrained(p, **load_kw)
                if not torch.cuda.is_available():
                    model.to("cpu")
                model.eval()
                # Avoid "invalid generation flags: temperature, top_p" when using do_sample=False
                if getattr(model, "generation_config", None) is not None:
                    model.generation_config.do_sample = False
                    if hasattr(model.generation_config, "temperature"):
                        setattr(model.generation_config, "temperature", None)
                    if hasattr(model.generation_config, "top_p"):
                        setattr(model.generation_config, "top_p", None)
                _LOCAL_MODEL_CACHE[cache_key] = (tokenizer, model)
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                print(
                    "[Local LLM] Model loaded. Device=%s. First inference may take several minutes on CPU."
                    % dev,
                    flush=True,
                )
            else:
                tokenizer, model = cached

        text = prompt
        try:
            if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            pass
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        max_new_tokens = _local_max_new_tokens()
        print(
            "[Local LLM] Generating up to %s tokens (may take 1–10+ min on CPU)..."
            % max_new_tokens,
            flush=True,
        )
        if not torch.cuda.is_available():
            n_threads = (os.getenv("LOCAL_CPU_THREADS") or "").strip()
            if n_threads:
                try:
                    torch.set_num_threads(int(n_threads))
                except Exception:
                    pass
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    def _local_gguf_response(self, prompt: str) -> str:
        """Run a local GGUF model via llama-cpp-python."""
        model_path = (self.model or "").strip()
        if not model_path:
            raise RuntimeError("Local GGUF provider requires a model path (e.g. path/to/model.gguf).")
        p = Path(model_path)
        if not p.exists():
            raise RuntimeError(f"GGUF model path not found: {model_path}")
        if p.suffix.lower() != ".gguf":
            raise RuntimeError(f"GGUF model path must point to a .gguf file: {model_path}")
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError(
                "GGUF models require llama-cpp-python. On Windows use a pre-built wheel (Python 3.10–3.12): "
                "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu "
                "GPU: replace /whl/cpu with /whl/cu121 (or cu122). If you use Python 3.13, try a venv with 3.11 or 3.12."
            ) from e
        # Default 16k so LLM Reasoning Layer / schema-guided prompts (~15k+ tokens) fit
        n_ctx = int(os.getenv("LOCAL_GGUF_N_CTX", "16384").strip() or "16384")
        n_gpu_layers = -1  # use GPU for all layers if available
        if (os.getenv("LOCAL_GGUF_CPU") or "").strip().lower() in ("1", "true", "yes", "y"):
            n_gpu_layers = 0
        cache_key = str(p.resolve()).lower()
        with _LOCAL_MODEL_LOCK:
            if cache_key not in _LOCAL_GGUF_CACHE:
                print(
                    "[Local GGUF] Loading model from",
                    model_path,
                    "(this may take a minute)...",
                    flush=True,
                )
                _LOCAL_GGUF_CACHE[cache_key] = Llama(
                    model_path=str(p.resolve()),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
                print("[Local GGUF] Model loaded.", flush=True)
            llm = _LOCAL_GGUF_CACHE[cache_key]
        max_new_tokens = _local_max_new_tokens()
        print(
            "[Local GGUF] Generating up to %s tokens..." % max_new_tokens,
            flush=True,
        )
        out = llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop=None,
            echo=False,
        )
        text = (out.get("choices") or [{}])[0].get("text") or ""
        return text.strip()
