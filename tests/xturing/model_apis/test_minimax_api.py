from unittest.mock import MagicMock, patch

import pytest


def _build_openai_error(error_cls, message):
    if error_cls is Exception:
        return Exception(message)
    # APIStatusError subclasses (RateLimitError) take `response`; APIError takes `request`
    import inspect

    sig = inspect.signature(error_cls.__init__)
    if "response" in sig.parameters:
        mock_response = MagicMock()
        mock_response.status_code = 429
        return error_cls(message, response=mock_response, body=None)
    return error_cls(message, request=MagicMock(), body=None)


class TestMiniMaxTextGenerationAPI:
    """Test suite for MiniMaxTextGenerationAPI"""

    def test_missing_openai_dependency(self):
        """Test that missing openai package raises ModuleNotFoundError"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        with patch.object(
            MiniMaxTextGenerationAPI,
            "_ensure_dependency",
            side_effect=ModuleNotFoundError(
                "The openai SDK is required for MiniMaxTextGenerationAPI."
            ),
        ):
            with pytest.raises(ModuleNotFoundError, match="openai SDK is required"):
                MiniMaxTextGenerationAPI(
                    model="MiniMax-M2.7",
                    api_key="test-key",
                )

    def test_initialization(self):
        """Test MiniMaxTextGenerationAPI initialization"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        with patch("xturing.model_apis.minimax.OpenAI") as mock_openai:
            api = MiniMaxTextGenerationAPI(
                model="MiniMax-M2.7",
                api_key="test-key",
                request_batch_size=5,
            )

            assert api.engine == "MiniMax-M2.7"
            assert api.api_key == "test-key"
            assert api.request_batch_size == 5
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://api.minimax.io/v1",
            )

    def test_minimax_m27_initialization(self):
        """Test MiniMaxM27 convenience class initialization"""
        from xturing.model_apis.minimax import MiniMaxM27

        with patch("xturing.model_apis.minimax.OpenAI"):
            api = MiniMaxM27(api_key="test-key", request_batch_size=3)

            assert api.engine == "MiniMax-M2.7"
            assert api.api_key == "test-key"
            assert api.request_batch_size == 3
            assert api.config_name == "minimax_m2_7"

    def test_minimax_m27_highspeed_initialization(self):
        """Test MiniMaxM27HighSpeed convenience class initialization"""
        from xturing.model_apis.minimax import MiniMaxM27HighSpeed

        with patch("xturing.model_apis.minimax.OpenAI"):
            api = MiniMaxM27HighSpeed(api_key="test-key")

            assert api.engine == "MiniMax-M2.7-highspeed"
            assert api.config_name == "minimax_m2_7_highspeed"

    def test_clamp_temperature_zero(self):
        """Test that temperature 0 is clamped to 0.01"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        with patch("xturing.model_apis.minimax.OpenAI"):
            api = MiniMaxTextGenerationAPI(
                model="MiniMax-M2.7", api_key="test-key"
            )
            assert api._clamp_temperature(0.0) == 0.01
            assert api._clamp_temperature(-0.5) == 0.01
            assert api._clamp_temperature(0.5) == 0.5
            assert api._clamp_temperature(1.0) == 1.0
            assert api._clamp_temperature(None) is None

    def test_make_request_basic(self):
        """Test _make_request with basic parameters"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        with patch("xturing.model_apis.minimax.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            api = MiniMaxTextGenerationAPI(
                model="MiniMax-M2.7", api_key="test-key"
            )

            api._make_request(
                prompt="Hello, world!",
                max_tokens=100,
                temperature=0.7,
                top_p=None,
                stop_sequences=None,
            )

            mock_client.chat.completions.create.assert_called_once_with(
                model="MiniMax-M2.7",
                max_tokens=100,
                temperature=0.7,
                messages=[{"role": "user", "content": "Hello, world!"}],
            )

    def test_make_request_with_optional_params(self):
        """Test _make_request with optional parameters"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        with patch("xturing.model_apis.minimax.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            api = MiniMaxTextGenerationAPI(
                model="MiniMax-M2.7", api_key="test-key"
            )

            api._make_request(
                prompt="Hello, world!",
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                stop_sequences=["STOP", "END"],
            )

            mock_client.chat.completions.create.assert_called_once_with(
                model="MiniMax-M2.7",
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                stop=["STOP", "END"],
                messages=[{"role": "user", "content": "Hello, world!"}],
            )

    def test_make_request_temperature_clamped(self):
        """Test that temperature=0 is clamped in _make_request"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        with patch("xturing.model_apis.minimax.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            api = MiniMaxTextGenerationAPI(
                model="MiniMax-M2.7", api_key="test-key"
            )

            api._make_request(
                prompt="Hello",
                max_tokens=50,
                temperature=0.0,
                top_p=None,
                stop_sequences=None,
            )

            mock_client.chat.completions.create.assert_called_once_with(
                model="MiniMax-M2.7",
                max_tokens=50,
                temperature=0.01,
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_render_response_success(self):
        """Test _render_response with successful response"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a response"
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]

        result = MiniMaxTextGenerationAPI._render_response(mock_response)

        assert result == {
            "choices": [
                {
                    "text": "This is a response",
                    "finish_reason": "stop",
                }
            ]
        }

    def test_render_response_none(self):
        """Test _render_response with None response"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        result = MiniMaxTextGenerationAPI._render_response(None)
        assert result is None

    def test_render_response_empty_choices(self):
        """Test _render_response with empty choices list"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        mock_response = MagicMock()
        mock_response.choices = []

        result = MiniMaxTextGenerationAPI._render_response(mock_response)
        assert result is None

    def test_generate_text_single_prompt(self):
        """Test generate_text with single prompt"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        with patch("xturing.model_apis.minimax.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Generated text"
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]

            mock_client.chat.completions.create.return_value = mock_response

            api = MiniMaxTextGenerationAPI(
                model="MiniMax-M2.7", api_key="test-key"
            )

            results = api.generate_text(
                prompts="Test prompt",
                max_tokens=100,
                temperature=0.7,
            )

            assert len(results) == 1
            assert results[0]["prompt"] == "Test prompt"
            assert results[0]["response"]["choices"][0]["text"] == "Generated text"
            assert "created_at" in results[0]

    def test_generate_text_multiple_prompts(self):
        """Test generate_text with multiple prompts"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        with patch("xturing.model_apis.minimax.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Generated text"
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]

            mock_client.chat.completions.create.return_value = mock_response

            api = MiniMaxTextGenerationAPI(
                model="MiniMax-M2.7", api_key="test-key"
            )

            results = api.generate_text(
                prompts=["Prompt 1", "Prompt 2", "Prompt 3"],
                max_tokens=100,
                temperature=0.7,
            )

            assert len(results) == 3
            assert results[0]["prompt"] == "Prompt 1"
            assert results[1]["prompt"] == "Prompt 2"
            assert results[2]["prompt"] == "Prompt 3"

    def test_generate_text_with_retry(self):
        """Test generate_text retry logic on API errors"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        with patch("xturing.model_apis.minimax.OpenAI") as mock_openai:
            with patch("time.sleep"):
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                mock_response = MagicMock()
                mock_choice = MagicMock()
                mock_choice.message.content = "Generated text"
                mock_choice.finish_reason = "stop"
                mock_response.choices = [mock_choice]

                from xturing.model_apis import minimax as minimax_module

                mock_client.chat.completions.create.side_effect = [
                    _build_openai_error(
                        minimax_module.OpenAIRateLimitError, "Rate limit exceeded"
                    ),
                    mock_response,
                ]

                api = MiniMaxTextGenerationAPI(
                    model="MiniMax-M2.7", api_key="test-key"
                )

                results = api.generate_text(
                    prompts="Test prompt",
                    max_tokens=100,
                    temperature=0.7,
                    retries=3,
                )

                assert len(results) == 1
                assert results[0]["response"]["choices"][0]["text"] == "Generated text"
                assert mock_client.chat.completions.create.call_count == 2

    def test_generate_text_max_retries_exceeded(self):
        """Test generate_text when max retries exceeded"""
        from xturing.model_apis.minimax import MiniMaxTextGenerationAPI

        with patch("xturing.model_apis.minimax.OpenAI") as mock_openai:
            with patch("time.sleep"):
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                from xturing.model_apis import minimax as minimax_module

                mock_client.chat.completions.create.side_effect = _build_openai_error(
                    minimax_module.OpenAIAPIError, "API Error"
                )

                api = MiniMaxTextGenerationAPI(
                    model="MiniMax-M2.7", api_key="test-key"
                )

                results = api.generate_text(
                    prompts="Test prompt",
                    max_tokens=100,
                    temperature=0.7,
                    retries=2,
                )

                assert len(results) == 1
                assert results[0]["prompt"] == "Test prompt"
                assert results[0]["response"] is None
                assert mock_client.chat.completions.create.call_count == 3

    def test_config_names(self):
        """Test that config names are set correctly"""
        from xturing.model_apis.minimax import (
            MiniMaxM27,
            MiniMaxM27HighSpeed,
            MiniMaxTextGenerationAPI,
        )

        assert MiniMaxTextGenerationAPI.config_name == "minimax"
        assert MiniMaxM27.config_name == "minimax_m2_7"
        assert MiniMaxM27HighSpeed.config_name == "minimax_m2_7_highspeed"

    def test_registry_entries(self):
        """Test that MiniMax APIs are registered in BaseApi registry"""
        from xturing.model_apis.base import BaseApi

        assert "minimax" in BaseApi.registry
        assert "minimax_m2_7" in BaseApi.registry
        assert "minimax_m2_7_highspeed" in BaseApi.registry

    def test_base_url_constant(self):
        """Test that the MiniMax base URL is correct"""
        from xturing.model_apis.minimax import _MINIMAX_BASE_URL

        assert _MINIMAX_BASE_URL == "https://api.minimax.io/v1"


class TestMiniMaxIntegration:
    """Integration tests for MiniMax API (require MINIMAX_API_KEY env var)"""

    @pytest.fixture
    def api_key(self):
        import os

        key = os.environ.get("MINIMAX_API_KEY")
        if not key:
            pytest.skip("MINIMAX_API_KEY not set")
        return key

    def test_m27_generate_text(self, api_key):
        """Integration test: generate text with MiniMax M2.7"""
        from xturing.model_apis.minimax import MiniMaxM27

        api = MiniMaxM27(api_key=api_key)
        results = api.generate_text(
            prompts="Say hello in one word.",
            max_tokens=10,
            temperature=0.7,
        )

        assert len(results) == 1
        assert results[0]["response"] is not None
        assert len(results[0]["response"]["choices"]) == 1
        assert len(results[0]["response"]["choices"][0]["text"]) > 0

    def test_m27_highspeed_generate_text(self, api_key):
        """Integration test: generate text with MiniMax M2.7-highspeed"""
        from xturing.model_apis.minimax import MiniMaxM27HighSpeed

        api = MiniMaxM27HighSpeed(api_key=api_key)
        results = api.generate_text(
            prompts="Say hello in one word.",
            max_tokens=10,
            temperature=0.7,
        )

        assert len(results) == 1
        assert results[0]["response"] is not None
        assert len(results[0]["response"]["choices"][0]["text"]) > 0

    def test_temperature_zero_integration(self, api_key):
        """Integration test: verify temperature clamping works end-to-end"""
        from xturing.model_apis.minimax import MiniMaxM27HighSpeed

        api = MiniMaxM27HighSpeed(api_key=api_key)
        results = api.generate_text(
            prompts="What is 1+1? Reply with only the number.",
            max_tokens=5,
            temperature=0.0,
        )

        assert len(results) == 1
        assert results[0]["response"] is not None
