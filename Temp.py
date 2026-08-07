import os

os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("AWS_ACCOUNT_ID", "611184449569")
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("CONFIG_S3_BUCKET", "test-bucket")
os.environ.setdefault("BUSINESS_UNIT", "BUK")
os.environ.setdefault("CHANNEL_TYPE", "PHONE")


"""Unit tests for Lambda handler — Kinesis consumer; parses events (Contact Lens utterances).

Covers:
- Event decoding (base64 + raw JSON)
- Utterance extraction from Contact Lens segments
- Session creation, loading, and schema merging
- Rule evaluation (conditions and orchestrator triggers)
- Contact lifecycle (CONTACT_INITIATED → SEGMENTS → CONTACT_DISCONNECTED)
- Lambda handler (Kinesis batch and direct invoke)
- AgentCore orchestrator invocation
- Error handling and edge cases
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from ail_connect_adapter import handler
from ail_connect_adapter.transformer import evaluate_condition, should_invoke_orchestrator
from ail_connect_adapter.error_codes import ErrorCode, ProcessingError

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_redis(monkeypatch):
    """Mock redis.Redis client."""
    mock_client = MagicMock()
    monkeypatch.setattr(handler, "redis_client", mock_client)
    return mock_client


@pytest.fixture
def mock_s3(monkeypatch):
    """Mock boto3 S3 client."""
    mock_client = MagicMock()
    monkeypatch.setattr(handler, "s3_client", mock_client)
    return mock_client


@pytest.fixture
def mock_call_agentcore_api(monkeypatch):
    """Mock the AgentCore HTTP call so no real SigV4-signed network request is made."""
    mock_fn = MagicMock(return_value={"status": "accepted"})
    monkeypatch.setattr(handler, "_call_agentcore_api", mock_fn)
    return mock_fn


@pytest.fixture
def env_vars(monkeypatch):
    """Set up required environment variables."""
    monkeypatch.setenv("REDIS_HOST", "localhost")
    monkeypatch.setenv("REDIS_PORT", "6379")
    monkeypatch.setenv("BUSINESS_UNIT", "BUK")
    monkeypatch.setenv("CHANNEL_TYPE", "PHONE")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "86400")
    monkeypatch.setenv("CONFIG_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("AGENTCORE_RUNTIME_NAME", "test-runtime")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCOUNT_ID", "611184449569")


@pytest.fixture(autouse=True)
def reset_caches():
    handler._state_schema_cache = None
    handler._rules_cache = None
    handler._event_config_cache = None
    handler._transformer_cache = {}
    yield
    handler._state_schema_cache = None
    handler._rules_cache = None
    handler._event_config_cache = None
    handler._transformer_cache = {}


@pytest.fixture
def mock_context():
    """Mock Lambda context object."""
    context = Mock()
    context.aws_request_id = "test-request-123"
    return context


@pytest.fixture
def sample_utterance() -> dict:
    """Sample Contact Lens utterance."""
    return {
        "speaker": "CUSTOMER",
        "text": "I want to complain about my account",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_segment(sample_utterance: dict) -> dict:
    """Sample Contact Lens segment."""
    return {
        "Transcript": {
            "Content": sample_utterance["text"],
            "ParticipantRole": "CUSTOMER",
        },
        "MatchedCategories": ["dissatisfaction", "complaint"],
    }


# ============================================================================
# TESTS: Helpers
# ============================================================================


class TestUtcNowIso:
    def test_returns_iso_format_string(self):
        result = handler.utc_now_iso()
        assert isinstance(result, str)
        assert "T" in result
        assert "+" in result or "Z" in result

    def test_returns_valid_datetime(self):
        result = handler.utc_now_iso()
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
        assert parsed is not None


class TestSafeJsonDumps:
    def test_serializes_dict(self):
        obj = {"key": "value", "number": 42}
        result = handler.safe_json_dumps(obj)
        assert json.loads(result) == obj

    def test_handles_datetime(self):
        dt = datetime.now(timezone.utc)
        obj = {"timestamp": dt}
        result = handler.safe_json_dumps(obj)
        assert "timestamp" in result

    def test_handles_non_serializable(self):
        class CustomObj:
            def __str__(self):
                return "custom"

        obj = {"custom": CustomObj()}
        result = handler.safe_json_dumps(obj)
        assert json.loads(result) is not None


class TestDecodeKinesisData:
    """decode_kinesis_data() raises ProcessingError (not builtin exceptions) on bad input."""

    def test_decodes_base64_encoded_json(self):
        data = {"EventType": "SEGMENTS", "ContactId": "123"}
        encoded = base64.b64encode(json.dumps(data).encode()).decode()
        result = handler.decode_kinesis_data(encoded)
        assert result == data

    def test_decodes_double_base64_encoded_json(self):
        data = {"EventType": "SEGMENTS", "ContactId": "123"}
        first_encoded = base64.b64encode(json.dumps(data).encode()).decode()
        second_encoded = base64.b64encode(first_encoded.encode()).decode()
        result = handler.decode_kinesis_data(second_encoded)
        assert result == data

    def test_raises_on_invalid_payload(self):
        with pytest.raises(ProcessingError) as exc_info:
            handler.decode_kinesis_data("not-valid-json-or-base64")
        assert exc_info.value.code == ErrorCode.MALFORMED_KINESIS_PAYLOAD

    def test_handles_empty_string(self):
        with pytest.raises(ProcessingError) as exc_info:
            handler.decode_kinesis_data("")
        assert exc_info.value.code == ErrorCode.EMPTY_RECORD_PAYLOAD


class TestNormaliseEventType:
    """normalise_event_type() lives on the transformer, reached via get_transformer()."""

    def test_normalises_started_to_contact_initiated(self, env_vars, mock_s3):
        handler._event_config_cache = {
            "event_type_aliases": {"STARTED": "CONTACT_INITIATED", "DISCONNECTED": "CONTACT_DISCONNECTED"},
            "valid_event_types": [],
        }
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        assert transformer.normalise_event_type("STARTED") == "CONTACT_INITIATED"

    def test_normalises_disconnected(self, env_vars, mock_s3):
        handler._event_config_cache = {
            "event_type_aliases": {"STARTED": "CONTACT_INITIATED", "DISCONNECTED": "CONTACT_DISCONNECTED"},
            "valid_event_types": [],
        }
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        assert transformer.normalise_event_type("DISCONNECTED") == "CONTACT_DISCONNECTED"

    def test_normalises_segments(self, env_vars, mock_s3):
        handler._event_config_cache = {"event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        assert transformer.normalise_event_type("SEGMENTS") == "SEGMENTS"

    def test_normalises_case_insensitive(self, env_vars, mock_s3):
        handler._event_config_cache = {"event_type_aliases": {"STARTED": "CONTACT_INITIATED"}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        assert transformer.normalise_event_type("started") == "CONTACT_INITIATED"

    def test_returns_empty_for_none(self, env_vars, mock_s3):
        handler._event_config_cache = {"event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        assert transformer.normalise_event_type(None) == ""


class TestExtractUtterancesFromSegments:
    def test_extracts_single_utterance(self, sample_segment, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {"AGENT": "COLLEAGUE"}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        utterances, categories = transformer.extract_utterances_from_segments([sample_segment])
        assert len(utterances) == 1
        assert utterances[0]["text"] == sample_segment["Transcript"]["Content"]
        assert utterances[0]["speaker"] == "CUSTOMER"

    def test_extracts_matched_categories(self, sample_segment, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        _, categories = transformer.extract_utterances_from_segments([sample_segment])
        assert categories == ["complaint", "dissatisfaction"]

    def test_maps_agent_to_colleague(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {"AGENT": "COLLEAGUE"}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        segments = [{"Transcript": {"Content": "How can I help?", "ParticipantRole": "AGENT"}}]
        utterances, _ = transformer.extract_utterances_from_segments(segments)
        assert utterances[0]["speaker"] == "COLLEAGUE"

    def test_handles_empty_segments_list(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        utterances, categories = transformer.extract_utterances_from_segments([])
        assert utterances == []
        assert categories == []

    def test_skips_empty_transcripts(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        segments = [{"Transcript": {"Content": "", "ParticipantRole": "CUSTOMER"}}]
        utterances, _ = transformer.extract_utterances_from_segments(segments)
        assert len(utterances) == 0

    def test_handles_missing_transcript(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        segments = [{"MatchedCategories": ["dissatisfaction"]}]
        utterances, categories = transformer.extract_utterances_from_segments(segments)
        assert len(utterances) == 0
        assert categories == ["dissatisfaction"]

    def test_extracts_multiple_segments(self, sample_segment, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        utterances, _ = transformer.extract_utterances_from_segments([sample_segment, sample_segment])
        assert len(utterances) == 2

    def test_handles_transcript_without_content(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        segments = [{"Transcript": {"ParticipantRole": "CUSTOMER"}}]
        utterances, categories = transformer.extract_utterances_from_segments(segments)
        assert utterances == []
        assert categories == []


# ============================================================================
# TESTS: Rule Evaluation
# ============================================================================


class TestEvaluateCondition:
    def test_evaluates_eq_condition(self):
        assert evaluate_condition({"field": "current_intent", "op": "eq", "value": "COMPLAINT"}, {"current_intent": "COMPLAINT"}) is True

    def test_evaluates_neq_condition(self):
        assert evaluate_condition({"field": "current_intent", "op": "neq", "value": "QUERY"}, {"current_intent": "COMPLAINT"}) is True

    def test_evaluates_gt_condition(self):
        assert evaluate_condition({"field": "utterance_count", "op": "gt", "value": 5}, {"utterance_count": 10}) is True

    def test_evaluates_gte_condition(self):
        assert evaluate_condition({"field": "utterance_count", "op": "gte", "value": 10}, {"utterance_count": 10}) is True

    def test_evaluates_lt_condition(self):
        assert evaluate_condition({"field": "utterance_count", "op": "lt", "value": 10}, {"utterance_count": 5}) is True

    def test_evaluates_lte_condition(self):
        assert evaluate_condition({"field": "utterance_count", "op": "lte", "value": 5}, {"utterance_count": 5}) is True

    def test_evaluates_mod_condition(self):
        condition = {"field": "utterance_count", "op": "mod", "value": 2, "equals": 0}
        assert evaluate_condition(condition, {"utterance_count": 4}) is True

    def test_evaluates_is_null(self):
        assert evaluate_condition({"field": "contact_ended_at", "op": "is_null"}, {"contact_ended_at": None}) is True

    def test_evaluates_not_null(self):
        assert evaluate_condition({"field": "contact_started_at", "op": "not_null"}, {"contact_started_at": "2026-01-01"}) is True

    def test_evaluates_contains_condition(self):
        condition = {"field": "matched_categories", "op": "contains", "value": "dissatisfaction"}
        context = {"matched_categories": ["dissatisfaction", "complaint"]}
        assert evaluate_condition(condition, context) is True

    def test_evaluates_not_empty_condition(self):
        assert evaluate_condition({"field": "contact_id", "op": "not_empty"}, {"contact_id": "123"}) is True

    def test_evaluates_empty_condition(self):
        assert evaluate_condition({"field": "contact_id", "op": "empty"}, {"contact_id": None}) is True

    def test_raises_for_unknown_op(self):
        """Unsupported operator raises ProcessingError, not a builtin ValueError."""
        condition = {"field": "status", "op": "unknown_op", "value": "test"}
        with pytest.raises(ProcessingError) as exc_info:
            evaluate_condition(condition, {"status": "test"})
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_RULE_OPERATOR

    def test_raises_on_invalid_comparison(self):
        """Type mismatches on gt/lt/etc. are not caught — they propagate as TypeError."""
        condition = {"field": "number", "op": "gt", "value": "not_a_number"}
        with pytest.raises(TypeError):
            evaluate_condition(condition, {"number": 10})


class TestShouldInvokeOrchestrator:
    def test_returns_false_when_no_rules_and_no_fallback(self):
        session = {
            "session_id": "123",
            "qualification_context": {"utterance_count": 0},
        }
        invoke, _ = should_invoke_orchestrator(session, [], "")
        assert invoke is False

    def test_returns_false_when_no_rules_match_and_no_direct_trigger(self):
        session = {
            "session_id": "123",
            "qualification_context": {"utterance_count": 0},
        }
        invoke, reason = should_invoke_orchestrator(session, [], "contact_initiated")
        assert invoke is False
        assert reason == "contact_initiated"

    def test_matches_rule_with_all_conditions(self):
        rules = [
            {
                "name": "dissatisfaction_rule",
                "conditions": [
                    {"field": "matched_categories", "op": "contains", "value": "dissatisfaction"},
                    {"field": "utterance_count", "op": "gte", "value": 3},
                ],
            }
        ]
        session = {"session_id": "123", "qualification_context": {"utterance_count": 5}}
        invoke, reason = should_invoke_orchestrator(session, rules, "", event_matched_categories=["dissatisfaction", "complaint"])
        assert invoke is True
        assert reason == "dissatisfaction_rule"

    def test_does_not_match_rule_with_failed_condition(self):
        rules = [
            {
                "name": "dissatisfaction_rule",
                "conditions": [
                    {"field": "matched_categories", "op": "contains", "value": "dissatisfaction"},
                    {"field": "utterance_count", "op": "gte", "value": 10},
                ],
            }
        ]
        session = {"session_id": "123", "qualification_context": {"utterance_count": 5}}
        invoke, _ = should_invoke_orchestrator(session, rules, "", event_matched_categories=["dissatisfaction"])
        assert invoke is False


# ============================================================================
# TESTS: Session Management
# ============================================================================


class TestMergeWithSchema:
    def test_adds_missing_fields_from_schema(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"session_id": "default", "bu": "BUK", "colleague_id": "SYSTEM"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        result = transformer.merge_with_schema({"session_id": "123"})
        assert result["bu"] == "BUK"
        assert result["colleague_id"] == "SYSTEM"

    def test_does_not_overwrite_existing_fields(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"session_id": "default", "bu": "BUK"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        result = transformer.merge_with_schema({"session_id": "123", "bu": "CUSTOM"})
        assert result["bu"] == "CUSTOM"

    def test_merges_nested_dicts(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"qualification_context": {"channel_type": "PHONE", "utterances": []}}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")
        result = transformer.merge_with_schema({"qualification_context": {"utterances": []}})
        assert result["qualification_context"]["channel_type"] == "PHONE"
        assert result["qualification_context"]["utterances"] == []


class TestCreateSession:
    def test_creates_session_with_defaults(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK", "session_id": "default", "current_phase": "CONTACT_OPEN"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")

        now = handler.utc_now_iso()
        session = transformer.create_session("contact-123", now)

        assert session["session_id"] == "contact-123"
        assert session["bu"] == "BUK"
        assert session["current_phase"] == "CONTACT_OPEN"
        assert session["qualification_context"]["channel_type"] == "PHONE"
        assert session["qualification_context"]["utterance_count"] == 0


class TestLoadSession:
    def test_loads_existing_session_from_redis(self, mock_redis, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK", "session_id": "default"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")

        session_data = {"session_id": "contact-123", "current_phase": "PRODUCT_IDENTIFICATION_OPEN"}
        mock_redis.get.return_value = json.dumps(session_data)

        result, is_existing = handler.load_session("ail:contact-123", "contact-123", transformer)

        assert is_existing is True
        assert result["session_id"] == "contact-123"
        assert result["current_phase"] == "PRODUCT_IDENTIFICATION_OPEN"

    def test_creates_new_session_when_not_found(self, mock_redis, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {"bu": "BUK", "session_id": "default", "current_phase": "CONTACT_OPEN"}
        mock_s3.get_object.side_effect = Exception("No config")
        transformer = handler.get_transformer("PHONE")

        mock_redis.get.return_value = None

        result, is_existing = handler.load_session("ail:contact-123", "contact-123", transformer)

        assert is_existing is False
        assert result["session_id"] == "contact-123"


class TestRedisHelpers:
    def test_redis_get_decodes_bytes(self, mock_redis):
        mock_redis.get.return_value = b'{"key":"value"}'
        assert handler.redis_get("test-key") == '{"key":"value"}'

    def test_redis_get_returns_string(self, mock_redis):
        mock_redis.get.return_value = '{"key":"value"}'
        assert handler.redis_get("test-key") == '{"key":"value"}'

    def test_redis_set_unsafe_with_ttl(self, mock_redis):
        """redis_set_unsafe() is a full overwrite — the safe path is redis_atomic_merge()."""
        handler.redis_set_unsafe("test-key", "test-value", ttl=60)
        mock_redis.set.assert_called_once_with("test-key", "test-value", ex=60)


# ============================================================================
# TESTS: Core Processing
# ============================================================================


class TestProcessEvent:
    def test_processes_direct_event(self, env_vars, mock_redis, mock_s3, mock_call_agentcore_api):
        """New session bootstrapped by a non-INITIATED event still processes and invokes orchestrator."""
        handler._state_schema_cache = {"bu": "BUK", "session_id": "default", "current_phase": "CONTACT_OPEN"}
        handler._event_config_cache = {"speaker_map": {"AGENT": "COLLEAGUE"}, "event_type_aliases": {}, "valid_event_types": []}
        mock_s3.get_object.side_effect = Exception("No config")
        mock_redis.get.return_value = None
        handler._rules_cache = []

        event = {
            "EventType": "SEGMENTS",
            "ContactId": "contact-123",
            "colleague_id": "COLLEAGUE_001",
            "Segments": [{"Transcript": {"Content": "I want to complain", "ParticipantRole": "CUSTOMER"}}],
        }

        result = handler.process_event(event, request_id="req-123")

        assert result["contact_id"] == "contact-123"
        mock_call_agentcore_api.assert_called()

    def test_raises_on_missing_contact_id(self, env_vars):
        """validate_event() raises ProcessingError, not KeyError, when ContactId is absent."""
        handler._state_schema_cache = {"bu": "BUK"}
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        with pytest.raises(ProcessingError) as exc_info:
            handler.process_event({"EventType": "SEGMENTS"})
        assert exc_info.value.code == ErrorCode.MISSING_CONTACT_ID


class TestCreateOrUpdateSession:
    """create_or_update_session() returns a 6-tuple: (..., event_type)."""

    def test_creates_new_session_on_contact_initiated(self, env_vars, mock_redis, mock_s3):
        handler._state_schema_cache = {"bu": "BUK", "session_id": "default", "current_phase": "CONTACT_OPEN"}
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {"STARTED": "CONTACT_INITIATED"}, "valid_event_types": []}
        mock_s3.get_object.side_effect = Exception("No config")
        mock_redis.get.return_value = None
        handler._rules_cache = []

        payload = {"EventType": "CONTACT_INITIATED", "ContactId": "contact-123", "colleague_id": "COLLEAGUE_001"}

        session, existed, reason, transformer, matched_categories, event_type = handler.create_or_update_session(payload, "req-123")

        assert session["session_id"] == "contact-123"
        assert existed is False
        assert session["bu"] == "BUK"
        assert matched_categories == []
        assert event_type == "CONTACT_INITIATED"

    def test_loads_existing_session(self, env_vars, mock_redis, mock_s3):
        handler._state_schema_cache = {"bu": "BUK"}
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        mock_s3.get_object.side_effect = Exception("No config")

        session_data = {
            "session_id": "contact-123",
            "current_phase": "PRODUCT_IDENTIFICATION_OPEN",
            "qualification_context": {"utterance_count": 3},
        }
        mock_redis.get.return_value = json.dumps(session_data)

        payload = {"EventType": "SEGMENTS", "ContactId": "contact-123", "Segments": []}

        session, existed, reason, transformer, matched_categories, event_type = handler.create_or_update_session(payload)

        assert session["session_id"] == "contact-123"
        assert existed is True
        assert isinstance(matched_categories, list)
        assert event_type == "SEGMENTS"

    def test_appends_utterances_to_session(self, env_vars, mock_redis, mock_s3):
        handler._state_schema_cache = {"bu": "BUK", "session_id": "default"}
        handler._event_config_cache = {"speaker_map": {"AGENT": "COLLEAGUE"}, "event_type_aliases": {}, "valid_event_types": []}
        mock_s3.get_object.side_effect = Exception("No config")

        session_data = {
            "session_id": "contact-123",
            "current_phase": "CONTACT_OPEN",
            "qualification_context": {"utterances": [], "utterance_count": 0},
        }
        mock_redis.get.return_value = json.dumps(session_data)

        payload = {
            "EventType": "SEGMENTS",
            "ContactId": "contact-123",
            "Segments": [{"Transcript": {"Content": "I have a complaint", "ParticipantRole": "CUSTOMER"}}],
        }

        session, existed, reason, transformer, matched_categories, event_type = handler.create_or_update_session(payload)

        assert session["qualification_context"]["utterance_count"] == 1
        assert len(session["qualification_context"]["utterances"]) == 1
        assert matched_categories == []


# ============================================================================
# TESTS: Lambda Handler
# ============================================================================


class TestLambdaHandler:
    def test_handles_kinesis_batch_event(self, env_vars, mock_redis, mock_s3, mock_call_agentcore_api, mock_context):
        handler._state_schema_cache = {"bu": "BUK", "session_id": "default", "current_phase": "CONTACT_OPEN"}
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        mock_s3.get_object.side_effect = Exception("No config")
        mock_redis.get.return_value = None
        handler._rules_cache = []

        kinesis_data = {
            "EventType": "SEGMENTS",
            "ContactId": "contact-123",
            "colleague_id": "COLLEAGUE_001",
            "Segments": [{"Transcript": {"Content": "Test utterance", "ParticipantRole": "CUSTOMER"}}],
        }
        encoded = base64.b64encode(json.dumps(kinesis_data).encode()).decode()

        event = {
            "Records": [
                {"kinesis": {"data": encoded, "sequenceNumber": "seq-1"}},
                {"kinesis": {"data": encoded, "sequenceNumber": "seq-2"}},
            ]
        }

        result = handler.lambda_handler(event, mock_context)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["processed"] == 2
        assert len(body["results"]) == 2

    def test_handles_direct_invoke(self, env_vars, mock_redis, mock_s3, mock_call_agentcore_api, mock_context):
        handler._state_schema_cache = {"bu": "BUK", "session_id": "default", "current_phase": "CONTACT_OPEN"}
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        mock_s3.get_object.side_effect = Exception("No config")
        mock_redis.get.return_value = None
        handler._rules_cache = []

        event = {"EventType": "CONTACT_INITIATED", "ContactId": "contact-123", "colleague_id": "COLLEAGUE_001"}

        result = handler.lambda_handler(event, mock_context)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["processed"] == 1
        assert body["results"][0]["contact_id"] == "contact-123"

    def test_handles_lambda_context_missing(self, env_vars, mock_redis, mock_s3, mock_call_agentcore_api):
        handler._state_schema_cache = {"bu": "BUK", "session_id": "default", "current_phase": "CONTACT_OPEN"}
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        mock_s3.get_object.side_effect = Exception("No config")
        mock_redis.get.return_value = None
        handler._rules_cache = []

        event = {"EventType": "CONTACT_INITIATED", "ContactId": "contact-123"}

        result = handler.lambda_handler(event, None)
        assert result["statusCode"] == 200

    def test_returns_error_result_for_invalid_record(self, env_vars, mock_redis, mock_context):
        """lambda_handler never raises for a bad direct-invoke payload — it reports an error result."""
        handler._state_schema_cache = {"bu": "BUK"}
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._rules_cache = []

        event = {"EventType": "SEGMENTS"}  # missing ContactId

        result = handler.lambda_handler(event, mock_context)

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["results"][0]["status"] == "error"
        assert body["results"][0]["error_code"] == ErrorCode.MISSING_CONTACT_ID.value

    def test_handles_empty_batch(self, mock_context):
        event = {"Records": []}
        result = handler.lambda_handler(event, mock_context)
        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["processed"] == 0


# ============================================================================
# TESTS: S3 Config Loading
# ============================================================================


class TestLoadStateSchema:
    def test_loads_schema_from_s3(self, env_vars, mock_s3, monkeypatch):
        monkeypatch.setenv("CONFIG_S3_BUCKET", "test-bucket")
        handler.CONFIG_S3_BUCKET = "test-bucket"

        schema = {"field1": "value1", "field2": {}}
        mock_s3.get_object.return_value = {"Body": Mock(read=Mock(return_value=json.dumps(schema).encode()))}
        handler._state_schema_cache = None

        assert handler.load_state_schema() == schema

    def test_caches_schema_on_repeated_calls(self, env_vars, mock_s3, monkeypatch):
        monkeypatch.setenv("CONFIG_S3_BUCKET", "test-bucket")
        handler.CONFIG_S3_BUCKET = "test-bucket"

        schema = {"field1": "value1"}
        mock_s3.get_object.return_value = {"Body": Mock(read=Mock(return_value=json.dumps(schema).encode()))}
        handler._state_schema_cache = None

        handler.load_state_schema()
        handler.load_state_schema()

        assert mock_s3.get_object.call_count == 1

    def test_returns_default_schema_when_bucket_not_set(self, monkeypatch):
        original_bucket = handler.CONFIG_S3_BUCKET
        handler.CONFIG_S3_BUCKET = None
        handler._state_schema_cache = None

        with pytest.raises(ValueError, match="CONFIG_S3_BUCKET"):
            handler.load_state_schema()

        handler.CONFIG_S3_BUCKET = original_bucket

    def test_returns_cached_schema(self):
        handler._state_schema_cache = {"cached": True}
        assert handler.load_state_schema() == {"cached": True}


class TestLoadOrchestratorRules:
    def test_loads_rules_from_s3(self, env_vars, mock_s3, monkeypatch):
        monkeypatch.setenv("CONFIG_S3_BUCKET", "test-bucket")
        handler.CONFIG_S3_BUCKET = "test-bucket"

        rules = [{"name": "rule1", "conditions": [{"field": "status", "op": "eq", "value": "active"}]}]
        mock_s3.get_object.return_value = {"Body": Mock(read=Mock(return_value=json.dumps(rules).encode()))}
        handler._rules_cache = None

        assert handler.load_orchestrator_rules() == rules

    def test_caches_rules_on_repeated_calls(self, env_vars, mock_s3, monkeypatch):
        monkeypatch.setenv("CONFIG_S3_BUCKET", "test-bucket")
        handler.CONFIG_S3_BUCKET = "test-bucket"

        rules = [{"name": "rule1", "conditions": []}]
        mock_s3.get_object.return_value = {"Body": Mock(read=Mock(return_value=json.dumps(rules).encode()))}
        handler._rules_cache = None

        handler.load_orchestrator_rules()
        handler.load_orchestrator_rules()

        assert mock_s3.get_object.call_count == 1

    def test_returns_cached_rules(self):
        handler._rules_cache = [{"name": "cached-rule"}]
        assert handler.load_orchestrator_rules() == [{"name": "cached-rule"}]


# ============================================================================
# TESTS: AgentCore Invocation / Redis auth token
# ============================================================================


class TestInvokeAiOrchestrator:
    def test_logs_orchestrator_invocation(self, env_vars, mock_call_agentcore_api, caplog):
        session = {
            "session_id": "contact-123",
            "colleague_id": "COLLEAGUE_001",
            "current_phase": "CONTACT_OPEN",
            "bu": "BUK",
            "qualification_context": {"channel_type": "PHONE"},
        }

        handler.invoke_ai_orchestrator(session, "contact-123", event_category="COLLEAGUE", event_action="UPDATE_ISSUE")

        assert "Orchestrator invoked" in caplog.text
        assert "contact-123" in caplog.text
        assert "COLLEAGUE" in caplog.text
        assert "UPDATE_ISSUE" in caplog.text
        mock_call_agentcore_api.assert_called_once()

    def test_raises_processing_error_when_agentcore_call_fails(self, env_vars, mock_call_agentcore_api):
        """invoke_ai_orchestrator re-raises AgentCore failures as ProcessingError, it does not swallow them."""
        mock_call_agentcore_api.side_effect = Exception("AgentCore failure")

        session = {
            "session_id": "contact-123",
            "colleague_id": "COLLEAGUE_001",
            "current_phase": "CONTACT_OPEN",
            "bu": "BUK",
            "qualification_context": {"channel_type": "PHONE"},
        }

        with pytest.raises(ProcessingError) as exc_info:
            handler.invoke_ai_orchestrator(session, "contact-123", event_category="LIFECYCLE", event_action="RULE_BASED_EVENT")

        assert exc_info.value.code == ErrorCode.ORCHESTRATOR_INVOCATION_FAILED

    def test_returns_early_when_runtime_name_missing(self, monkeypatch, mock_call_agentcore_api, caplog):
        """Missing AGENTCORE_RUNTIME_NAME short-circuits before any network call."""
        monkeypatch.delenv("AGENTCORE_RUNTIME_NAME", raising=False)

        handler.invoke_ai_orchestrator({"session_id": "contact-123"}, "contact-123", event_category="COLLEAGUE", event_action="TEST")

        mock_call_agentcore_api.assert_not_called()
        assert "Missing env var" in caplog.text


class TestFetchAuthToken:
    def test_missing_secret_arn_returns_none(self, monkeypatch):
        monkeypatch.delenv("REDIS_SECRET_ARN", raising=False)
        assert handler._fetch_auth_token() is None

    def test_returns_none_when_auth_token_missing(self, monkeypatch):
        monkeypatch.setenv("REDIS_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:123:secret:redis-token")

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": '{"password": "wrong-key-name"}'}

        with patch("ail_connect_adapter.handler.boto3.client", return_value=mock_client):
            result = handler._fetch_auth_token()

        assert result is None

    def test_returns_none_on_invalid_json(self, monkeypatch):
        monkeypatch.setenv("REDIS_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:123:secret:redis-token")

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "not-valid-json"}

        with patch("ail_connect_adapter.handler.boto3.client", return_value=mock_client):
            result = handler._fetch_auth_token()

        assert result is None

    def test_creates_client_with_region_from_env(self, monkeypatch):
        monkeypatch.setenv("REDIS_SECRET_ARN", "arn:aws:secretsmanager:us-east-1:123:secret:test")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-2")

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": '{"authToken": "token-456"}'}

        with patch("ail_connect_adapter.handler.boto3.client", return_value=mock_client) as mock_boto:
            result = handler._fetch_auth_token()

        assert result == "token-456"
        _, kwargs = mock_boto.call_args
        assert kwargs["region_name"] == "eu-west-2"


class TestContactLifecycle:
    def test_complete_contact_lifecycle(self, env_vars, mock_redis, mock_s3, mock_call_agentcore_api, mock_context):
        """CONTACT_INITIATED → SEGMENTS end to end through lambda_handler."""
        handler._state_schema_cache = {"bu": "BUK", "session_id": "default", "current_phase": "CONTACT_OPEN"}
        handler._event_config_cache = {"speaker_map": {"AGENT": "COLLEAGUE"}, "event_type_aliases": {}, "valid_event_types": []}
        mock_s3.get_object.side_effect = Exception("No config")
        mock_redis.get.return_value = None
        handler._rules_cache = []

        event_initiated = {"EventType": "CONTACT_INITIATED", "ContactId": "contact-123", "colleague_id": "COLLEAGUE_001"}
        result_initiated = handler.lambda_handler(event_initiated, mock_context)
        assert result_initiated["statusCode"] == 200
        assert json.loads(result_initiated["body"])["processed"] == 1

        session_after_init = {
            "session_id": "contact-123",
            "colleague_id": "COLLEAGUE_001",
            "current_phase": "CONTACT_OPEN",
            "qualification_context": {"utterances": [], "utterance_count": 0},
        }
        mock_redis.get.return_value = json.dumps(session_after_init)

        kinesis_data = {
            "EventType": "SEGMENTS",
            "ContactId": "contact-123",
            "Segments": [
                {
                    "Transcript": {"Content": "I'm unhappy with my service", "ParticipantRole": "CUSTOMER"},
                    "Categories": {"MatchedCategories": ["dissatisfaction"]},
                }
            ],
        }
        encoded = base64.b64encode(json.dumps(kinesis_data).encode()).decode()
        event_segments = {"Records": [{"kinesis": {"data": encoded, "sequenceNumber": "seq-1"}}]}
        result_segments = handler.lambda_handler(event_segments, mock_context)
        assert result_segments["statusCode"] == 200

        assert mock_call_agentcore_api.call_count >= 1


# ============================================================================
# Additional coverage: transformer operators, validation fallbacks, edge cases
# ============================================================================


class TestEvaluateConditionExtended:
    def test_evaluates_mod_operator(self):
        condition = {"field": "utterance_count", "op": "mod", "value": 5, "equals": 0}
        assert evaluate_condition(condition, {"utterance_count": 10}) is True

    def test_evaluates_mod_operator_not_equals(self):
        condition = {"field": "utterance_count", "op": "mod", "value": 5, "equals": 0}
        assert evaluate_condition(condition, {"utterance_count": 12}) is False

    def test_evaluates_gte_delta_or_null_reference_null(self):
        condition = {"field": "current_count", "op": "gte_delta_or_null", "reference_field": "baseline", "value": 5}
        assert evaluate_condition(condition, {"current_count": 10, "baseline": None}) is True

    def test_evaluates_gte_delta_or_null_type_mismatch(self):
        condition = {"field": "current_count", "op": "gte_delta_or_null", "reference_field": "baseline", "value": 5}
        assert evaluate_condition(condition, {"current_count": "not_int", "baseline": 5}) is False

    def test_evaluates_gte_delta_or_null_valid_delta(self):
        condition = {"field": "current_count", "op": "gte_delta_or_null", "reference_field": "baseline", "value": 5}
        assert evaluate_condition(condition, {"current_count": 15, "baseline": 10}) is True

    def test_evaluates_not_contains_operator(self):
        condition = {"field": "matched_categories", "op": "not_contains", "value": "billing"}
        assert evaluate_condition(condition, {"matched_categories": ["complaint"]}) is True

    def test_not_contains_type_mismatch_returns_false(self):
        condition = {"field": "matched_categories", "op": "not_contains", "value": "billing"}
        assert evaluate_condition(condition, {"matched_categories": "not_a_list"}) is False


class TestValidateEventFallbacks:
    def test_validate_event_contactid_lowercase_fallback(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {}
        mock_s3.get_object.side_effect = Exception("No config")

        transformer = handler.get_transformer("PHONE")
        result = transformer.validate_event({"contactId": "contact-456", "EventType": "SEGMENTS"})
        assert result["contact_id"] == "contact-456"

    def test_validate_event_eventtype_lowercase_fallback(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": ["TEST_EVENT"]}
        handler._state_schema_cache = {}
        mock_s3.get_object.side_effect = Exception("No config")

        transformer = handler.get_transformer("PHONE")
        result = transformer.validate_event({"contactId": "contact-123", "eventType": "test_event"})
        assert result["event_type"] == "TEST_EVENT"

    def test_validate_event_unsupported_event_type_raises(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": ["ALLOWED_TYPE"]}
        handler._state_schema_cache = {}
        mock_s3.get_object.side_effect = Exception("No config")

        transformer = handler.get_transformer("PHONE")
        with pytest.raises(ProcessingError) as exc_info:
            transformer.validate_event({"contactId": "contact-123", "EventType": "UNSUPPORTED"})
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_EVENT_TYPE


class TestApplyEventSessionMutations:
    def test_apply_event_sets_contact_ended_at_on_disconnect(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {}
        mock_s3.get_object.side_effect = Exception("No config")

        transformer = handler.get_transformer("PHONE")
        session = {"session_id": "123", "qualification_context": {}}
        transformed = {"event_type": "CONTACT_DISCONNECTED", "utterances": [], "matched_categories": []}
        now = "2026-01-01T00:00:00Z"

        result = transformer.apply_event_to_session(session, transformed, now)
        assert result["contact_ended_at"] == now

    def test_apply_event_recovers_corrupted_utterances_list(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {}
        mock_s3.get_object.side_effect = Exception("No config")

        transformer = handler.get_transformer("PHONE")
        session = {"session_id": "123", "qualification_context": {"utterances": "corrupted_not_a_list"}}
        transformed = {
            "event_type": "SEGMENTS",
            "utterances": [{"speaker": "CUSTOMER", "text": "test"}],
            "matched_categories": [],
        }

        result = transformer.apply_event_to_session(session, transformed, "2026-01-01T00:00:00Z")
        assert isinstance(result["qualification_context"]["utterances"], list)
        assert len(result["qualification_context"]["utterances"]) == 1


class TestExtractUtterancesEdgeCases:
    def test_extract_utterances_categories_direct_list(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {}
        mock_s3.get_object.side_effect = Exception("No config")

        transformer = handler.get_transformer("PHONE")
        segments = [
            {
                "Transcript": {"Content": "test", "ParticipantRole": "CUSTOMER"},
                "MatchedCategories": ["complaint", "dissatisfaction"],
            }
        ]

        _, categories = transformer.extract_utterances_from_segments(segments)
        assert "complaint" in categories
        assert "dissatisfaction" in categories

    def test_extract_utterances_invalid_categories_type_handled(self, env_vars, mock_s3):
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "valid_event_types": []}
        handler._state_schema_cache = {}
        mock_s3.get_object.side_effect = Exception("No config")

        transformer = handler.get_transformer("PHONE")
        segments = [
            {
                "Transcript": {"Content": "test", "ParticipantRole": "CUSTOMER"},
                "MatchedCategories": "invalid_string",
            }
        ]

        _, categories = transformer.extract_utterances_from_segments(segments)
        assert categories == []


class TestHandlerUncoveredBranches:
    def test_redis_atomic_merge_fallback_when_no_lua_sha(self, env_vars, mock_redis, monkeypatch):
        """When no atomic-merge script is registered, redis_atomic_merge() must take the manual GET-MERGE-SET path."""
        monkeypatch.setattr(handler, "_atomic_merge_sha", None)
        mock_redis.get.return_value = None

        handler.redis_atomic_merge("ail:test-123", {"field": "value"})

        assert mock_redis.evalsha.called is False
        assert mock_redis.setex.called

    def test_redis_atomic_merge_falls_back_on_lua_failure(self, env_vars, mock_redis, monkeypatch):
        """When the registered Lua script errors at runtime, fall back to the manual merge instead of losing the write."""
        monkeypatch.setattr(handler, "_atomic_merge_sha", "fakesha123")
        mock_redis.evalsha.side_effect = Exception("Lua error")

        handler.redis_atomic_merge("ail:test-123", {"field": "value"})

        mock_redis.evalsha.assert_called_once()
        assert mock_redis.setex.called

    def test_send_to_orchestrator_if_required_invoke_false(self, env_vars, mock_s3):
        handler._rules_cache = [{"name": "rule1", "conditions": [{"field": "utterance_count", "op": "gte", "value": 100}]}]
        handler._event_config_cache = {"speaker_map": {}, "event_type_aliases": {}, "rule_payload_overrides": {}, "valid_event_types": []}
        handler._state_schema_cache = {}
        mock_s3.get_object.side_effect = Exception("No config")

        transformer = handler.get_transformer("PHONE")
        session = {"session_id": "123", "qualification_context": {"utterance_count": 5}}

        invoke, reason = handler.send_to_orchestrator_if_required(session, "fallback", transformer, [])

        assert invoke is False
        assert reason == "fallback"
