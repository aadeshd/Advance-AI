import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
import redis

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 86400))

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_SSL = os.environ.get("REDIS_SSL", "false").lower() == "true"

BUSINESS_UNIT = os.environ.get("BUSINESS_UNIT", "BUK")
CHANNEL_TYPE = os.environ.get("CHANNEL_TYPE", "PHONE")

# S3 config
CONFIG_S3_BUCKET = os.environ.get("CONFIG_S3_BUCKET")
STATE_SCHEMA_S3_KEY = os.environ.get("STATE_SCHEMA_S3_KEY", "config/state_schema.json")
RULES_S3_KEY = os.environ.get("RULES_S3_KEY", "config/orchestrator_rules.json")

# FIX 3: Account ID for ExpectedBucketOwner on all S3 calls
AWS_ACCOUNT_ID = os.environ["AWS_ACCOUNT_ID"]

# FIX 2: Explicit timeouts for all S3 network calls — prevents hanging Lambda executions
S3_CLIENT_CONFIG = Config(connect_timeout=5, read_timeout=10)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

SPEAKER_MAP = {"AGENT": "COLLEAGUE", "CUSTOMER": "CUSTOMER", "SYSTEM": "SYSTEM"}

EVENT_TYPE_ALIASES = {
    "STARTED": "CONTACT_INITIATED",
    "CONTACT_INITIATED": "CONTACT_INITIATED",
    "DISCONNECTED": "CONTACT_DISCONNECTED",
    "ENDED": "CONTACT_DISCONNECTED",
    "CONTACT_DISCONNECTED": "CONTACT_DISCONNECTED",
    "SEGMENTS": "SEGMENTS",
}

PHASE_EVENT_MAP = {
    "CONTACT_OPEN": "CONTACT_INITIATED",
    "PRODUCT_IDENTIFICATION_OPEN": "PRODUCT_IDENTIFICATION_OPEN",
    "PRODUCT_DETERMINED": "PRODUCT_DETERMINED",
    "RULE_BASED_EVENT": "RULE_BASED_EVENT",
    "GENERATE_SUMMARY": "GENERATE_SUMMARY",
}

RULE_PAYLOAD_OVERRIDES: Dict[str, Dict[str, str]] = {
    "compaction_trigger": {
        "event_category": "COMPACTION",
        "event_action": "COMPACT_SESSION",
    }
}


# --------------------------------------------------------------------------------------
# Module-level clients
# --------------------------------------------------------------------------------------

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    ssl=REDIS_SSL,
    socket_connect_timeout=3,
    socket_timeout=3,
    retry_on_timeout=True,
    decode_responses=True,
)

# FIX 2: S3 client uses explicit timeout config
s3_client = boto3.client("s3", config=S3_CLIENT_CONFIG)

_agentcore_endpoint = os.environ.get("AGENTCORE_ENDPOINT")
agent_client = boto3.client(
    "bedrock-agentcore",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    **({"endpoint_url": f"https://{_agentcore_endpoint}"} if _agentcore_endpoint else {}),
)

logger.info(f"Agent client created {agent_client}")


# --------------------------------------------------------------------------------------
# S3 config — cached at module level, loaded once per cold start
# --------------------------------------------------------------------------------------

_state_schema_cache: Optional[Dict] = None
_rules_cache: Optional[List] = None


def load_state_schema() -> Dict:
    global _state_schema_cache
    if _state_schema_cache is not None:
        return _state_schema_cache

    if not CONFIG_S3_BUCKET:
        raise EnvironmentError(
            "CONFIG_S3_BUCKET env var is not set — cannot load state schema"
        )

    try:
        # FIX 3a: ExpectedBucketOwner added — first S3 get_object call
        response = s3_client.get_object(
            Bucket=CONFIG_S3_BUCKET,
            Key=STATE_SCHEMA_S3_KEY,
            ExpectedBucketOwner=AWS_ACCOUNT_ID,
        )
        _state_schema_cache = json.loads(response["Body"].read().decode("utf-8"))
        logger.info(
            "State schema loaded | bucket=%s | key=%s",
            CONFIG_S3_BUCKET,
            STATE_SCHEMA_S3_KEY,
        )
        return _state_schema_cache

    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(
            f"State schema not found in S3 | bucket={CONFIG_S3_BUCKET} | key={STATE_SCHEMA_S3_KEY}"
        )
    except EnvironmentError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to load state schema from S3: {exc}") from exc


def load_orchestrator_rules() -> Optional[List]:
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache
    if not CONFIG_S3_BUCKET:
        logger.warning("CONFIG_S3_BUCKET not set — fallback: matched_categories only")
        return None
    try:
        # FIX 3b: ExpectedBucketOwner added — second S3 get_object call
        response = s3_client.get_object(
            Bucket=CONFIG_S3_BUCKET,
            Key=RULES_S3_KEY,
            ExpectedBucketOwner=AWS_ACCOUNT_ID,
        )
        _rules_cache = json.loads(response["Body"].read().decode("utf-8"))
        logger.info(
            "Orchestrator rules loaded | bucket=%s | key=%s | count=%s",
            CONFIG_S3_BUCKET,
            RULES_S3_KEY,
            len(_rules_cache),
        )
        return _rules_cache
    except Exception:
        logger.exception("Failed to load orchestrator rules from S3 — fallback: matched_categories only")
        return None


# --------------------------------------------------------------------------------------
# Rule evaluation
# --------------------------------------------------------------------------------------

def evaluate_condition(condition: Dict, context: Dict) -> bool:
    field = condition.get("field")
    op = condition.get("op")
    value = condition.get("value")
    field_value = context.get(field)
    try:
        if op == "eq":          return field_value == value
        if op == "neq":         return field_value != value
        if op == "gt":          return field_value > value
        if op == "gte":         return field_value >= value
        if op == "lt":          return field_value < value
        if op == "lte":         return field_value <= value
        if op == "mod":         return isinstance(field_value, int) and field_value % value == condition.get("equals", 0)
        if op == "not_empty":   return bool(field_value)
        if op == "empty":       return not bool(field_value)
        if op == "contains":    return value in field_value
        logger.warning("Unknown rule op=%s | condition=%s", op, condition)
        return False
    except Exception:
        logger.exception(
            "Condition evaluation failed | condition=%s | field=%s | value=%s",
            condition, field, field_value,
        )
        return False


def should_invoke_orchestrator(
    session: Dict,
    prev_count: int,
    curr_count: int,
    matched_categories: List[str],
    speaker: str,
) -> Tuple[bool, str]:
    rules = load_orchestrator_rules()

    context = {
        "current_phase": session.get("current_phase", "CONTACT_OPEN"),
        "utterance_count": curr_count,
        "prev_utterance_count": prev_count,
        "matched_categories": matched_categories,
        "speaker": speaker,
        "contact_id": session.get("session_id", ""),
        "current_intent": session.get("current_intent", ""),
        **{k: v for k, v in session.items() if k != "qualification_context"},
    }

    if rules is None:
        if "dissatisfaction" in matched_categories:
            return True, "fallback_dissatisfaction_category"
        return False, ""

    for rule in rules:
        rule_name = rule.get("name", "unnamed")
        conditions = rule.get("conditions", [])
        if not conditions:
            continue
        if all(evaluate_condition(c, context) for c in conditions):
            logger.info(
                "Rule matched | rule=%s | contact_id=%s",
                rule_name, context.get("contact_id"),
            )
            return True, rule_name

    return False, ""


# --------------------------------------------------------------------------------------
# Redis helpers
# --------------------------------------------------------------------------------------

def redis_get(key: str) -> Optional[str]:
    return redis_client.get(key)


def redis_set(key: str, value: str, ttl: int = SESSION_TTL_SECONDS):
    redis_client.set(key, value, ex=ttl)


# --------------------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------------------

def merge_with_schema(session: Dict, schema: Dict) -> Dict:
    for key, default_value in schema.items():
        if key not in session:
            session[key] = default_value
        elif isinstance(default_value, dict) and isinstance(session.get(key), dict):
            session[key] = merge_with_schema(session[key], default_value)
    return session


def create_new_session(contact_id: str, now: str) -> Dict:
    schema = load_state_schema()
    session = {**schema}
    session.update({
        "session_id": contact_id,
        "created_at": now,
        "updated_at": now,
        "contact_started_at": now,
        "ttl_seconds": SESSION_TTL_SECONDS,
    })
    session.setdefault("contact_ended_at", None)
    qc = session.setdefault("qualification_context", {})
    qc.setdefault("utterances", [])
    qc.setdefault("utterance_count", 0)
    qc.setdefault("channel_type", CHANNEL_TYPE)
    logger.info("Session created from S3 schema | contact_id=%s", contact_id)
    return session


def load_session(session_key: str, contact_id: str, now: str) -> Tuple[Dict, bool]:
    raw = redis_get(session_key)
    if not raw:
        logger.warning("No session found, creating new | contact_id=%s", contact_id)
        return create_new_session(contact_id, now), False
    session = json.loads(raw)
    schema = load_state_schema()
    session = merge_with_schema(session, schema)
    return session, True


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


# FIX 1: decode_kinesis_data — base64 path removed entirely.
# Kinesis records are now always raw JSON strings from the sender lambda.
def decode_kinesis_data(kinesis_data: str) -> Dict:
    try:
        return json.loads(kinesis_data)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Kinesis data is not valid JSON: {kinesis_data[:80]}"
        ) from exc


def normalise_event_type(event_type: Optional[str]) -> str:
    if not event_type:
        return ""
    return EVENT_TYPE_ALIASES.get(str(event_type).upper(), str(event_type).upper())


def extract_utterances_from_segments(segments: List[Dict]) -> Tuple[List[Dict], List[str]]:
    utterances = []
    matched_categories = []

    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        if "Categories" in seg and isinstance(seg["Categories"], dict):
            matched_categories = seg["Categories"].get("MatchedCategories", []) or []
        transcript = seg.get("Transcript")
        if isinstance(transcript, dict):
            content = transcript.get("Content", "") or ""
            if content.strip():
                role = transcript.get("ParticipantRole") or transcript.get("ParticipantId") or "CUSTOMER"
                speaker = SPEAKER_MAP.get(role.upper(), "CUSTOMER")
                utterances.append({
                    "speaker": speaker,
                    "text": content.strip(),
                    "timestamp": utc_now_iso(),
                })

    return utterances, matched_categories


# --------------------------------------------------------------------------------------
# Lambda entry
# --------------------------------------------------------------------------------------

# FIX 4: Extracted from lambda_handler to reduce its Cognitive Complexity.
# Decodes and parses one Kinesis record into structured fields.
# Raises on bad data so the caller can append to batch_failures cleanly.
def _decode_record(record: Dict) -> Tuple[str, str, List[Dict], List[str]]:
    payload = decode_kinesis_data(record["kinesis"]["data"])
    event_type = normalise_event_type(payload.get("EventType"))
    contact_id = payload.get("ContactId") or payload.get("contactId")

    if not contact_id:
        raise ValueError(f"No contact_id | keys={list(payload.keys())}")

    utterances: List[Dict] = []
    matched_categories: List[str] = []

    if event_type == "SEGMENTS":
        segments = payload.get("Segments") or payload.get("segments") or []
        utterances, matched_categories = extract_utterances_from_segments(segments)

    return contact_id, event_type, utterances, matched_categories


# FIX 5: Extracted from lambda_handler to reduce its Cognitive Complexity.
# Merges a decoded record into the per-contact accumulator dict.
def _merge_into_contacts(
    contacts: Dict,
    contact_id: str,
    event_type: str,
    utterances: List[Dict],
    matched_categories: List[str],
) -> None:
    if contact_id not in contacts:
        contacts[contact_id] = {
            "utterances": [],
            "matched_categories": [],
            "event_type": event_type,
        }
    contacts[contact_id]["utterances"].extend(utterances)
    if matched_categories:
        contacts[contact_id]["matched_categories"] = matched_categories


def lambda_handler(event: Dict, context):
    request_id = getattr(context, "aws_request_id", "") if context else ""

    if not isinstance(event, dict) or "Records" not in event:
        try:
            result = process_event(event, request_id=request_id)
            return {"statusCode": 200, "body": safe_json_dumps(result)}
        except Exception as exc:
            logger.exception("Direct invoke failed | request_id=%s", request_id)
            return {"statusCode": 500, "body": safe_json_dumps({"error": str(exc)})}

    records = event.get("Records", [])
    batch_failures = []
    contacts: Dict[str, Dict] = {}

    logger.info("Kinesis batch started | records=%s | request_id=%s", len(records), request_id)

    for record in records:
        seq = record.get("kinesis", {}).get("sequenceNumber", "")
        try:
            contact_id, event_type, utterances, matched_categories = _decode_record(record)
            _merge_into_contacts(contacts, contact_id, event_type, utterances, matched_categories)
            logger.info("Record decoded | seq=%s | eventType=%s | contactId=%s", seq, event_type, contact_id)
        except Exception:
            logger.exception("Record decode failed | seq=%s | request_id=%s", seq, request_id)
            if seq:
                batch_failures.append({"itemIdentifier": seq})

    for contact_id, data in contacts.items():
        try:
            process_contact(
                contact_id=contact_id,
                event_type=data["event_type"],
                new_utterances=data["utterances"],
                matched_categories=data["matched_categories"],
                request_id=request_id,
            )
        except Exception:
            logger.exception(
                "Contact processing failed | contact_id=%s | request_id=%s",
                contact_id, request_id,
            )

    logger.info(
        "Kinesis batch complete | total=%s | failures=%s | request_id=%s",
        len(records), len(batch_failures), request_id,
    )
    return {"batchItemFailures": batch_failures}


# --------------------------------------------------------------------------------------
# Core processing
# --------------------------------------------------------------------------------------

def process_event(event: Dict, request_id: str = "") -> Dict:
    event_type = normalise_event_type(event.get("EventType"))
    contact_id = event.get("ContactId")
    if not contact_id:
        raise KeyError("Missing required field: ContactId")
    segments = event.get("Segments") or []
    utterances, matched_categories = extract_utterances_from_segments(segments)
    return process_contact(
        contact_id=contact_id,
        event_type=event_type,
        new_utterances=utterances,
        matched_categories=matched_categories,
        request_id=request_id,
    )


def process_contact(
    contact_id: str,
    event_type: str,
    new_utterances: List[Dict],
    matched_categories: List[str],
    request_id: str = "",
) -> Dict:
    now = utc_now_iso()
    session_key = f"ail:{contact_id}"

    logger.info(
        "Processing contact | event_type=%s | contact_id=%s | request_id=%s",
        event_type, contact_id, request_id,
    )

    if event_type == "CONTACT_INITIATED":
        session = create_new_session(contact_id, now)
        redis_set(session_key, safe_json_dumps(session))
        logger.info("Session created | contact_id=%s | ttl=%ss", contact_id, SESSION_TTL_SECONDS)
        return {"contact_id": contact_id, "event_type": event_type}

    if event_type == "CONTACT_DISCONNECTED":
        session, _ = load_session(session_key, contact_id, now)
        session["contact_ended_at"] = now
        session["updated_at"] = now
        redis_set(session_key, safe_json_dumps(session))
        logger.info(
            "Session closed | contact_id=%s | total_utterances=%s",
            contact_id, session["qualification_context"].get("utterance_count", 0),
        )
        return {"contact_id": contact_id, "event_type": event_type}

    if not new_utterances:
        logger.info("No utterances in segment | contact_id=%s", contact_id)
        return {"contact_id": contact_id, "event_type": event_type, "utterances": 0}

    session, _ = load_session(session_key, contact_id, now)
    qc = session["qualification_context"]
    prev_count = qc.get("utterance_count", 0)

    qc["utterances"].extend(new_utterances)
    qc["utterance_count"] = len(qc["utterances"])
    curr_count = qc["utterance_count"]

    if matched_categories:
        existing = session.get("categorization_result") or []
        for cat in matched_categories:
            if cat not in existing:
                existing.append(cat)
        session["categorization_result"] = existing
        logger.info("Categories matched | contact_id=%s | categories=%s", contact_id, matched_categories)

    session["updated_at"] = now
    redis_set(session_key, safe_json_dumps(session))

    logger.info(
        "Utterances stored | contact_id=%s | added=%s | total=%s | speakers=%s",
        contact_id, len(new_utterances), curr_count,
        [u["speaker"] for u in new_utterances],
    )

    speaker = new_utterances[-1]["speaker"] if new_utterances else "CUSTOMER"
    invoke, reason = should_invoke_orchestrator(session, prev_count, curr_count, matched_categories, speaker)

    if invoke:
        logger.info(
            "Orchestrator triggered | contact_id=%s | reason=%s | utterance_count=%s",
            contact_id, reason, curr_count,
        )
        invoke_ai_orchestrator(session, contact_id, reason=reason)

    return {
        "contact_id": contact_id,
        "event_type": event_type,
        "utterances_added": len(new_utterances),
        "utterance_count": curr_count,
        "matched_categories": matched_categories,
        "orchestrator_invoked": invoke,
    }


# --------------------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------------------

def invoke_ai_orchestrator(session: Dict, contact_id: str, reason: str = "") -> None:
    runtime_arn = os.environ.get("AGENTCORE_RUNTIME_ARN") or os.environ.get("AGENT_RUNTIME_ARN")
    if not runtime_arn:
        logger.error("Missing env var: AGENTCORE_RUNTIME_ARN | contact_id=%s", contact_id)
        return

    try:
        sid = re.sub(r"[^a-zA-Z0-9_-]", "_", contact_id)
        sid = sid.ljust(33, "0")

        phase = session.get("current_phase", "CONTACT_OPEN")
        default_event_action = PHASE_EVENT_MAP.get(phase, "CONTACT_INITIATED")

        overrides = RULE_PAYLOAD_OVERRIDES.get(reason, {})
        event_category = overrides.get("event_category", "LIFECYCLE")
        event_action = overrides.get("event_action", default_event_action)

        payload = {
            "session_id": contact_id,
            "colleague_id": session.get(
                "colleague_id",
                "arn:aws:connect:eu-west-1:123456789012:instance/abc/agent/COL001",
            ),
            "event_category": event_category,
            "event_action": event_action,
            "payload": {},
            "metadata": {
                "source": "CONNECT",
                "channel": "PHONE",
                "bu": "BUK",
                "correlation_id": "corr-002a",
                "timestamp": utc_now_iso(),
            },
            "api_version": "1.0",
        }

        logger.info(
            "AgentCore invoke started | contact_id=%s | phase=%s | event=%s | category=%s | reason=%s",
            contact_id, phase, event_action, event_category, reason,
        )

        response = agent_client.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            contentType="application/json",
            accept="application/json",
            runtimeSessionId=sid[:64],
            payload=json.dumps(payload).encode(),
        )

        body = response.get("response") or response.get("body")
        result = body.read().decode("utf-8") if hasattr(body, "read") else str(body)
        logger.info(
            "AgentCore invoke successful | contact_id=%s | response=%s",
            contact_id, result[:200],
        )

    except Exception:
        logger.exception("AgentCore invoke failed | contact_id=%s", contact_id)
