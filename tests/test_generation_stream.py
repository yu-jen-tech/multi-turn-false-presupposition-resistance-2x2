import unittest
from unittest.mock import MagicMock, Mock, patch

from openai import omit
from openai.lib.streaming.chat import ChatCompletionStream
from openai.types.chat import ChatCompletionChunk

from transcript_generation import ChatMessage, OpenAICompatibleChatClient


class GenerationStreamTests(unittest.TestCase):
    def complete_stream(self, finish_reason):
        # Exercise the real SDK's chunk accumulation and finalization locally.
        common = {
            "id": "test-stream",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
        }
        chunks = [
            {**common, "choices": [{"index": 0, "delta": {
                "role": "assistant", "content": "partial answer",
            }, "finish_reason": None}]},
            {**common, "choices": [{"index": 0, "delta": {},
                                    "finish_reason": finish_reason}]},
            {**common, "choices": [], "usage": {
                "prompt_tokens": 1330, "completion_tokens": 2048,
                "total_tokens": 3378,
                "completion_tokens_details": {"reasoning_tokens": 1888},
            }},
        ]
        raw_stream = MagicMock()
        raw_stream.__iter__.return_value = iter(
            ChatCompletionChunk.model_validate(chunk) for chunk in chunks
        )
        stream = ChatCompletionStream(
            raw_stream=raw_stream, response_format=omit, input_tools=omit,
        )
        sdk = Mock()
        sdk.chat.completions.stream.return_value = stream
        with patch("openai.OpenAI", return_value=sdk):
            client = OpenAICompatibleChatClient(api_key="test-key", api_retries=0)
        return client.complete(
            purpose="simulator_baseline", model="test-model",
            messages=[ChatMessage(role="user", content="Test")],
            temperature=0.7, max_tokens=2048, seed=None,
        )

    def test_length_finish_reaches_generator_with_usage(self):
        result = self.complete_stream("length")
        self.assertEqual(result.metadata.finish_reason, "length")
        self.assertEqual(result.metadata.reasoning_tokens, 1888)
        self.assertEqual(result.content, "partial answer")
        self.assertEqual(result.raw_response["usage"]["completion_tokens"], 2048)

    def test_normal_finish_preserves_content(self):
        result = self.complete_stream("stop")
        self.assertEqual(result.metadata.finish_reason, "stop")
        self.assertEqual(result.content, "partial answer")

    def test_missing_finish_reason_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "without a finish reason"):
            self.complete_stream(None)
