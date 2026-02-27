from unittest.mock import MagicMock, patch

import pytest


def _build_anthropic_error(error_cls, message):
    if error_cls is Exception:
        return Exception(message)
    return error_cls(message, response=None, body=None)


class TestClaudeTextGenerationAPI:
    """Test suite for ClaudeTextGenerationAPI"""

    def test_missing_anthropic_dependency(self):
        """Test that missing anthropic package raises ModuleNotFoundError"""
        with patch.dict("sys.modules", {"anthropic": None}):
            # Force reimport to trigger the import error path
            import sys

            # Remove from cache if present
            if "xturing.model_apis.claude" in sys.modules:
                del sys.modules["xturing.model_apis.claude"]

            # This should work but _ensure_dependency should fail
            from xturing.model_apis.claude import ClaudeTextGenerationAPI

            with pytest.raises(ModuleNotFoundError, match="anthropic SDK is required"):
                ClaudeTextGenerationAPI(
                    model="claude-3-sonnet-20240229",
                    api_key="test-key",
                )

        # Ensure we do not keep a stale submodule object on the package.
        import sys
        import xturing.model_apis as model_apis_pkg

        sys.modules.pop("xturing.model_apis.claude", None)
        if hasattr(model_apis_pkg, "claude"):
            delattr(model_apis_pkg, "claude")

    def test_initialization(self):
        """Test ClaudeTextGenerationAPI initialization"""
        from xturing.model_apis.claude import ClaudeTextGenerationAPI

        with patch("xturing.model_apis.claude.Anthropic") as mock_anthropic:
            api = ClaudeTextGenerationAPI(
                model="claude-3-sonnet-20240229",
                api_key="test-key",
                request_batch_size=5,
            )

            assert api.engine == "claude-3-sonnet-20240229"
            assert api.api_key == "test-key"
            assert api.request_batch_size == 5
            mock_anthropic.assert_called_once_with(api_key="test-key")

    def test_claude_sonnet_initialization(self):
        """Test ClaudeSonnet convenience class initialization"""
        from xturing.model_apis.claude import ClaudeSonnet

        with patch("xturing.model_apis.claude.Anthropic"):
            api = ClaudeSonnet(api_key="test-key", request_batch_size=3)

            assert api.engine == "claude-3-sonnet-20240229"
            assert api.api_key == "test-key"
            assert api.request_batch_size == 3
            assert api.config_name == "claude_3_sonnet"

    def test_make_request_basic(self):
        """Test _make_request with basic parameters"""
        from xturing.model_apis.claude import ClaudeTextGenerationAPI

        with patch("xturing.model_apis.claude.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            api = ClaudeTextGenerationAPI(
                model="claude-3-sonnet-20240229",
                api_key="test-key",
            )

            api._make_request(
                prompt="Hello, world!",
                max_tokens=100,
                temperature=0.7,
                top_p=None,
                stop_sequences=None,
            )

            mock_client.messages.create.assert_called_once_with(
                model="claude-3-sonnet-20240229",
                max_tokens=100,
                temperature=0.7,
                messages=[{"role": "user", "content": "Hello, world!"}],
            )

    def test_make_request_with_optional_params(self):
        """Test _make_request with optional parameters"""
        from xturing.model_apis.claude import ClaudeTextGenerationAPI

        with patch("xturing.model_apis.claude.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            api = ClaudeTextGenerationAPI(
                model="claude-3-sonnet-20240229",
                api_key="test-key",
            )

            api._make_request(
                prompt="Hello, world!",
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                stop_sequences=["STOP", "END"],
            )

            mock_client.messages.create.assert_called_once_with(
                model="claude-3-sonnet-20240229",
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                stop_sequences=["STOP", "END"],
                messages=[{"role": "user", "content": "Hello, world!"}],
            )

    def test_render_response_success(self):
        """Test _render_response with successful response"""
        from xturing.model_apis.claude import ClaudeTextGenerationAPI

        # Mock response object
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "This is a response"
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"

        result = ClaudeTextGenerationAPI._render_response(mock_response)

        assert result == {
            "choices": [
                {
                    "text": "This is a response",
                    "finish_reason": "end_turn",
                }
            ]
        }

    def test_render_response_multiple_blocks(self):
        """Test _render_response with multiple text blocks"""
        from xturing.model_apis.claude import ClaudeTextGenerationAPI

        # Mock response with multiple text blocks
        mock_response = MagicMock()
        mock_block1 = MagicMock()
        mock_block1.type = "text"
        mock_block1.text = "Part 1 "

        mock_block2 = MagicMock()
        mock_block2.type = "text"
        mock_block2.text = "Part 2"

        mock_response.content = [mock_block1, mock_block2]
        mock_response.stop_reason = "max_tokens"

        result = ClaudeTextGenerationAPI._render_response(mock_response)

        assert result == {
            "choices": [
                {
                    "text": "Part 1 Part 2",
                    "finish_reason": "max_tokens",
                }
            ]
        }

    def test_render_response_none(self):
        """Test _render_response with None response"""
        from xturing.model_apis.claude import ClaudeTextGenerationAPI

        result = ClaudeTextGenerationAPI._render_response(None)
        assert result is None

    def test_generate_text_single_prompt(self):
        """Test generate_text with single prompt"""
        from xturing.model_apis.claude import ClaudeTextGenerationAPI

        with patch("xturing.model_apis.claude.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            # Mock response
            mock_response = MagicMock()
            mock_text_block = MagicMock()
            mock_text_block.type = "text"
            mock_text_block.text = "Generated text"
            mock_response.content = [mock_text_block]
            mock_response.stop_reason = "end_turn"

            mock_client.messages.create.return_value = mock_response

            api = ClaudeTextGenerationAPI(
                model="claude-3-sonnet-20240229",
                api_key="test-key",
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
        from xturing.model_apis.claude import ClaudeTextGenerationAPI

        with patch("xturing.model_apis.claude.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            # Mock response
            mock_response = MagicMock()
            mock_text_block = MagicMock()
            mock_text_block.type = "text"
            mock_text_block.text = "Generated text"
            mock_response.content = [mock_text_block]
            mock_response.stop_reason = "end_turn"

            mock_client.messages.create.return_value = mock_response

            api = ClaudeTextGenerationAPI(
                model="claude-3-sonnet-20240229",
                api_key="test-key",
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
        from xturing.model_apis.claude import ClaudeTextGenerationAPI

        with patch("xturing.model_apis.claude.Anthropic") as mock_anthropic:
            with patch("time.sleep"):  # Mock sleep to speed up test
                mock_client = MagicMock()
                mock_anthropic.return_value = mock_client

                # Mock successful response
                mock_response = MagicMock()
                mock_text_block = MagicMock()
                mock_text_block.type = "text"
                mock_text_block.text = "Generated text"
                mock_response.content = [mock_text_block]
                mock_response.stop_reason = "end_turn"

                # First call fails, second succeeds
                from xturing.model_apis import claude as claude_module

                mock_client.messages.create.side_effect = [
                    _build_anthropic_error(
                        claude_module.AnthropicRateLimitError, "Rate limit exceeded"
                    ),
                    mock_response,
                ]

                api = ClaudeTextGenerationAPI(
                    model="claude-3-sonnet-20240229",
                    api_key="test-key",
                )

                results = api.generate_text(
                    prompts="Test prompt",
                    max_tokens=100,
                    temperature=0.7,
                    retries=3,
                )

                assert len(results) == 1
                assert results[0]["response"]["choices"][0]["text"] == "Generated text"
                # Should have been called twice (1 failure + 1 success)
                assert mock_client.messages.create.call_count == 2

    def test_generate_text_max_retries_exceeded(self):
        """Test generate_text when max retries exceeded"""
        from xturing.model_apis.claude import ClaudeTextGenerationAPI

        with patch("xturing.model_apis.claude.Anthropic") as mock_anthropic:
            with patch("time.sleep"):  # Mock sleep to speed up test
                mock_client = MagicMock()
                mock_anthropic.return_value = mock_client

                # Always fail
                from xturing.model_apis import claude as claude_module

                mock_client.messages.create.side_effect = _build_anthropic_error(
                    claude_module.AnthropicAPIError, "API Error"
                )

                api = ClaudeTextGenerationAPI(
                    model="claude-3-sonnet-20240229",
                    api_key="test-key",
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
                # Should have been called 3 times (initial + 2 retries)
                assert mock_client.messages.create.call_count == 3

    def test_config_names(self):
        """Test that config names are set correctly"""
        from xturing.model_apis.claude import ClaudeSonnet, ClaudeTextGenerationAPI

        assert ClaudeTextGenerationAPI.config_name == "claude"
        assert ClaudeSonnet.config_name == "claude_3_sonnet"
