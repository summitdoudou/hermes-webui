"""Regression tests for PR #2473 — /model alias switching to cross-provider custom models.

Pre-fix: `resolve_model_provider("ollama-local/glm-4.7-flash:q4_k_m")` with
`config.model.provider = "deepseek"` (which has a `base_url` set) hit the
`config_base_url` branch first and returned ``("ollama-local/...", "deepseek",
"https://api.deepseek.com/v1")`` — sending an ollama-style model id to the
deepseek API.  The custom_providers loop at the top of the function was
already cleared by the `_skip_custom_providers` guard for non-custom defaults.

Fix: a new branch BEFORE the ``config_base_url`` branch checks whether the
slash prefix matches a ``custom_providers[].name``.  When it does, the
function returns ``(model_id, "custom:<slug>", custom_base_url)`` — routing
to the custom provider rather than the configured default.

The model id is returned WITH the prefix intact, mirroring the local-server
branch at lines 1791-1800 (`_is_local_server_provider` / `_base_url_points_at_local_server`)
which also keeps the prefix.  Local-server endpoints (Ollama, LM Studio,
llama.cpp, vLLM, TabbyAPI) accept namespaced model ids; the user's alias is
expected to encode the model id exactly as the destination provider expects.
"""

from __future__ import annotations

import api.config as config


def _resolve(model_id, **cfg_overrides):
    """Resolve a model_id under a synthesized cfg, isolated from disk state."""
    old_cfg = dict(config.cfg)
    old_mtime = config._cfg_mtime
    config.cfg.clear()
    config.cfg.update(cfg_overrides)
    try:
        config._cfg_mtime = config.Path(config._get_config_path()).stat().st_mtime
    except Exception:
        config._cfg_mtime = 0.0
    try:
        return config.resolve_model_provider(model_id)
    finally:
        config.cfg.clear()
        config.cfg.update(old_cfg)
        config._cfg_mtime = old_mtime


class TestCrossProviderCustomModelSwitch:
    """The reported failure mode: deepseek active + ollama-local custom provider."""

    def test_cross_provider_custom_model_routes_to_custom_not_active(self):
        """The bug-fix case.  Active provider=deepseek (which has base_url) +
        model id with custom-provider prefix must route to the custom provider,
        not fall through to deepseek's base_url branch."""
        model, provider, base_url = _resolve(
            "ollama-local/glm-4.7-flash:q4_k_m",
            model={
                "provider": "deepseek",
                "base_url": "https://api.deepseek.com/v1",
            },
            custom_providers=[
                {"name": "ollama-local", "base_url": "http://127.0.0.1:11434/v1"},
            ],
        )
        assert provider == "custom:ollama-local", (
            f"Pre-fix routed to {provider!r} (likely 'deepseek') because the "
            "config_base_url branch caught the slashed model id before the "
            "custom_providers check.  Post-fix must route to custom:ollama-local."
        )
        assert base_url == "http://127.0.0.1:11434/v1"
        # Per the existing local-server convention (lines 1791-1800), the
        # model id is returned with its prefix intact.  Local-server endpoints
        # accept namespaced ids; the user's alias is expected to encode the
        # model id exactly as the destination provider expects.
        assert model == "ollama-local/glm-4.7-flash:q4_k_m"

    def test_cross_provider_custom_with_no_base_url_routes_to_custom(self):
        """Custom entries don't have to declare a base_url; the slug still wins."""
        model, provider, base_url = _resolve(
            "my-proxy/some-model",
            model={"provider": "deepseek", "base_url": "https://api.deepseek.com/v1"},
            custom_providers=[{"name": "my-proxy"}],
        )
        assert provider == "custom:my-proxy"
        assert base_url is None
        assert model == "my-proxy/some-model"

    def test_unknown_prefix_falls_through_to_existing_behavior(self):
        """If the slash prefix does NOT match any custom_providers entry, the
        new branch must be a no-op — preserving the legacy fall-through."""
        model, provider, base_url = _resolve(
            "unknown-vendor/some-model",
            model={"provider": "deepseek", "base_url": "https://api.deepseek.com/v1"},
            custom_providers=[{"name": "ollama-local"}],
        )
        # Falls through to the existing config_base_url branch — deepseek wins.
        assert provider == "deepseek"
        assert base_url == "https://api.deepseek.com/v1"

    def test_active_custom_provider_same_name_still_resolves_to_custom_slug(self):
        """Sanity: when the active provider IS the custom provider (canonicalized
        to custom:ollama-local via _resolve_configured_provider_id), the new
        branch fires because prefix != canonical config_provider.  The result
        must still route to the custom provider (same destination)."""
        model, provider, base_url = _resolve(
            "ollama-local/glm-4.7-flash:q4_k_m",
            model={
                "provider": "ollama-local",
                "base_url": "http://127.0.0.1:11434/v1",
            },
            custom_providers=[
                {"name": "ollama-local", "base_url": "http://127.0.0.1:11434/v1"},
            ],
        )
        assert provider == "custom:ollama-local"
        assert base_url == "http://127.0.0.1:11434/v1"

    def test_openrouter_active_provider_unaffected(self):
        """OpenRouter active provider must STILL keep the full provider/model
        path (existing #854/#894 invariant at line 1744-1745) — the new branch
        does not run because the function returns at the OpenRouter check first."""
        model, provider, base_url = _resolve(
            "ollama-local/glm-4.7-flash:q4_k_m",
            model={"provider": "openrouter"},
            custom_providers=[{"name": "ollama-local", "base_url": "http://x"}],
        )
        assert provider == "openrouter"
        assert model == "ollama-local/glm-4.7-flash:q4_k_m"

    def test_portal_provider_active_unaffected(self):
        """NVIDIA/Nous/OpenCode portal providers must STILL preserve full path
        (existing #2177 / #854 invariant at lines 1754-1756) — the new branch
        does not run because the portal check returns first."""
        model, provider, base_url = _resolve(
            "ollama-local/glm-4.7-flash:q4_k_m",
            model={"provider": "nvidia", "base_url": "https://integrate.api.nvidia.com/v1"},
            custom_providers=[{"name": "ollama-local", "base_url": "http://x"}],
        )
        assert provider == "nvidia"
        assert model == "ollama-local/glm-4.7-flash:q4_k_m"


class TestAvailableModelsAliases:
    """``get_available_models()`` exposes ``cfg.model.aliases`` so the WebUI
    frontend can resolve user-defined ``/model <alias>`` shortcuts.

    These tests use source-string assertions rather than calling the function
    directly. The function's caching + config-reload semantics are awkward to
    isolate inside a test suite that runs after other tests have mutated the
    in-memory cache fingerprint; the source-level shape check is sufficient
    to catch a maintainer accidentally dropping the aliases key.
    """

    def test_get_available_models_returns_aliases_key(self):
        """The returned dict must include an 'aliases' key the frontend can iterate."""
        import inspect
        src = inspect.getsource(config.get_available_models)
        assert '"aliases": model_aliases' in src, (
            "get_available_models() must include an 'aliases' key in its return "
            "dict so the WebUI frontend can resolve /model <alias> commands "
            "without a separate API call."
        )

    def test_aliases_are_read_from_cfg_model_aliases(self):
        """The aliases dict must come from cfg.model.aliases."""
        import inspect
        src = inspect.getsource(config.get_available_models)
        assert 'cfg.get("model", {}).get("aliases", {})' in src, (
            "get_available_models() must read aliases from cfg.model.aliases. "
            "The frontend's /model <alias> resolver expects this exact key."
        )

    def test_aliases_drop_empty_keys_and_values(self):
        """Empty alias keys or empty target model ids must be filtered out so
        the frontend doesn't accidentally match an empty string."""
        import inspect
        src = inspect.getsource(config.get_available_models)
        # The dict-comprehension must guard with `if k and v` so falsy
        # pairs (None, "", 0) are dropped before reaching the JSON response.
        assert "if k and v" in src, (
            "get_available_models() must drop empty alias keys/values so the "
            "frontend never matches '' against a stray user prompt."
        )

    def test_aliases_malformed_does_not_raise(self):
        """A non-dict ``aliases`` value (e.g. user typo) must default to {}
        rather than raising — the try/except + isinstance guard handles this."""
        import inspect
        src = inspect.getsource(config.get_available_models)
        # The aliases extraction must be wrapped in try/except so a malformed
        # cfg value can't crash /api/models.
        assert "except Exception:" in src, (
            "get_available_models() aliases extraction must be wrapped in "
            "try/except so a malformed cfg.model.aliases value (e.g. a string "
            "instead of a dict) doesn't crash the endpoint."
        )
        # The isinstance check guards against non-dict aliases values silently
        # producing garbage output.
        assert "isinstance(raw_aliases, dict)" in src
