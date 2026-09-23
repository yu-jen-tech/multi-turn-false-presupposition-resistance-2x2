from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx2
from openai import InternalServerError, OpenAI

from tof_nof_evaluation.models import SUPPORTED_JUDGES, default_judge_config
from tof_nof_evaluation.providers import OpenAICompatibleJudgeClient
from transcript_generation import ChatMessage


DECISION = '{"decision_type":"rejects_or_corrects","reason":"Correction."}'


def completion_response() -> httpx2.Response:
    return httpx2.Response(200, json={
        "id": "judge-response", "object": "chat.completion", "created": 1,
        "model": "openai/qwen3.8",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": DECISION,
        }}],
    })


def streamed_response(*, finished=True) -> httpx2.Response:
    chunks = []
    for delta, finish in (
        ({"role": "assistant", "reasoning_content": "Private "}, None),
        ({"reasoning_content": "reasoning.", "content": DECISION[:25]}, None),
        ({"content": DECISION[25:]}, "stop" if finished else None),
    ):
        chunks.append({
            "id": "judge-response", "object": "chat.completion.chunk",
            "created": 1, "model": "openai/qwen3.8",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })
    chunks.append({
        "id": "judge-response", "object": "chat.completion.chunk",
        "created": 1, "model": "openai/qwen3.8", "choices": [],
        "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50,
                  "completion_tokens_details": {"reasoning_tokens": 12}},
    })
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"},
        text="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
        + "data: [DONE]\n\n",
    )


class JudgeTransportTests(unittest.TestCase):
    def complete(self, handler, *, retries=1):
        with httpx2.Client(transport=httpx2.MockTransport(handler)) as http:
            with OpenAI(api_key="test", http_client=http, max_retries=0) as sdk:
                with patch("openai.OpenAI", return_value=sdk):
                    client = OpenAICompatibleJudgeClient(api_key="test", api_retries=retries)
                return client.complete(
                    config=default_judge_config(SUPPORTED_JUDGES["qwen3.8"]),
                    messages=[ChatMessage(role="user", content="Evaluate this response.")],
                )

    def test_524_honors_server_backoff_before_retrying_same_request(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                return httpx2.Response(524, json={
                    "error_code": 524, "retry_after": 120,
                    "error_name": "origin_response_timeout",
                })
            return streamed_response() if requests[-1].get("stream") else completion_response()

        with patch("tof_nof_evaluation.providers.time.sleep") as sleep:
            result = self.complete(handler)
        self.assertEqual(DECISION, result.content)
        self.assertEqual([120], [call.args[0] for call in sleep.call_args_list])
        self.assertEqual(2, len(requests))
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(1, result.metadata["retry_count"])

    def test_streams_and_preserves_visible_answer_reasoning_and_usage(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return streamed_response() if requests[-1].get("stream") else completion_response()

        result = self.complete(handler, retries=0)
        self.assertTrue(requests[0].get("stream"), "KI4BUW must receive a streaming request")
        self.assertEqual({"include_usage": True}, requests[0]["stream_options"])
        self.assertEqual(DECISION, result.content)
        self.assertEqual("stop", result.metadata["finish_reason"])
        self.assertEqual(12, result.metadata["reasoning_tokens"])
        self.assertEqual("Private reasoning.",
                         result.raw_response["choices"][0]["message"]["reasoning_content"])

    def test_incomplete_stream_is_retried_instead_of_scoring_partial_output(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return streamed_response(finished=len(requests) > 1)

        with patch("tof_nof_evaluation.providers.time.sleep"):
            result = self.complete(handler)
        self.assertEqual(2, len(requests), "Missing finish_reason must not become a decision")
        self.assertEqual(DECISION, result.content)
        self.assertEqual("stop", result.metadata["finish_reason"])

    def test_524_backoff_accepts_headers_and_handles_missing_or_invalid_hints(self):
        for headers, body, expected in (
            ({"Retry-After": "180"}, {"retry_after": 120}, 180),
            ({"Retry-After": "Thu, 01 Jan 1970 00:03:00 GMT"}, {}, 180),
            ({}, {}, 120),
            ({"Retry-After": "invalid"}, {"retry_after": "nan"}, 120),
            ({"Retry-After": "-1"}, {"retry_after": True}, 120),
        ):
            with self.subTest(headers=headers, body=body):
                attempts = []

                def handler(request):
                    attempts.append(request)
                    if len(attempts) == 1:
                        return httpx2.Response(524, headers=headers, json=body)
                    return streamed_response()

                with patch("tof_nof_evaluation.providers.time.sleep") as sleep, \
                        patch("tof_nof_evaluation.providers.time.time", return_value=0):
                    result = self.complete(handler)
                self.assertEqual(expected, sleep.call_args.args[0])
                self.assertEqual(DECISION, result.content)

    def test_persistent_524_stops_at_retry_limit_with_original_status(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx2.Response(524, json={"retry_after": 120})

        with patch("tof_nof_evaluation.providers.time.sleep") as sleep:
            with self.assertRaises(InternalServerError) as caught:
                self.complete(handler, retries=2)
        self.assertEqual(524, caught.exception.status_code)
        self.assertEqual(3, len(requests))
        self.assertEqual([120, 240], [call.args[0] for call in sleep.call_args_list])

    def test_interrupted_stream_closes_connection_and_discards_partial_answer(self):
        class BrokenStream(httpx2.SyncByteStream):
            closed = False

            def __iter__(self):
                chunk = {
                    "id": "broken", "object": "chat.completion.chunk", "created": 1,
                    "model": "openai/qwen3.8", "choices": [{
                        "index": 0, "finish_reason": None,
                        "delta": {"role": "assistant", "content": "discard me"},
                    }],
                }
                yield ("data: " + json.dumps(chunk) + "\n\n").encode()
                raise httpx2.ReadError("connection interrupted")

            def close(self):
                self.closed = True

        broken = BrokenStream()
        requests = []

        def handler(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                      stream=broken)
            return streamed_response()

        with patch("tof_nof_evaluation.providers.time.sleep"):
            result = self.complete(handler)
        self.assertTrue(broken.closed)
        self.assertEqual(2, len(requests))
        self.assertEqual(DECISION, result.content)
        self.assertNotIn("discard me", json.dumps(result.raw_response))
