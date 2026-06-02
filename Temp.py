import base64
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
import redis

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 86400))

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))

BUSINESS_UNIT = os.environ.get("BUSINESS_UNIT", "BUK")
CHANNEL_TYPE = os.environ.get("CHANNEL_TYPE", "PHONE")

UTTERANCE_THRESHOLD = int(os.environ.get("UTTERANCE_THRESHOLD", "5"))

# S3 config
CONFIG_S3_BUCKET = os.environ.get("CONFIG_S3_BUCKET")
STATE_SCHEMA_S3_KEY = os.environ.get("STATE_SCHEMA_S3_KEY", "config/state_schema.json")
RULES_S3_KEY = os.environ.get("RULES_S3_KEY", "config/orchestrator_rules.json")

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

SPEAKER_MAP = {"AGENT": "COLLEAGUE", "CUSTOMER": "CUSTOMER", "SYSTEM": "SYSTEM"}

PHASE_EVENT_MAP = {
    "CONTACT_OPEN": "CONTACT_INITIATED",
    "PRODUCT_IDENTIFICATION_OPEN": "PRODUCT_IDENTIFICATION_OPEN",
    "PRODUCT_DETERMINED": "PRODUCT_DETERMINED",
    "RULE_BASED_EVENT": "RULE_BASED_EVENT",
    "GENERATE_SUMMARY": "GENERATE_SUMMARY",
}

EVENT_TYPE_ALIASES = {
    "STARTED": "CONTACT_INITIATED",
    "CONTACT_INITIATED": "CONTACT_INITIATED",
    "DISCONNECTED": "CONTACT_DISCONNECTED",
    "ENDED": "CONTACT_DISCONNECTED",
    "CONTACT_DISCONNECTED": "CONTACT_DISCONNECTED",
    "SEGMENTS": "SEGMENTS",
}

# --------------------------------------------------------------------------------------
# Module-level clients
# --------------------------------------------------------------------------------------

# SSL false — Redis is within VPC, no TLS needed
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    ssl=False,
    socket_connect_timeout=3,
    socket_timeout=3,
    retry_on_timeout=True,
    decode_responses=True,
)

s3_client = boto3.client("s3")

_agentcore_endpoint = os.environ.get("AGENTCORE_ENDPOINT")
agent_client = boto3.client(
    "bedrock-agentcore",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    **({"endpoint_url": f"https://{_agentcore_endpoint}"} if _agentcore_endpoint else {}),
)

# --------------------------------------------------------------------------------------
# S3 config — cached at module level, loaded once per cold start
# --------------------------------------------------------------------------------------

_state_schema_cache: Optional[Dict] = None
_rules_cache: Optional[List] = None


def load_state_schema() -> Optional[Dict]:
    global _state_schema_cache
    if _state_schema_cache is not None:
        return _state_schema_cache
    if not CONFIG_S3_BUCKET:
        logger.warning("CONFIG_S3_BUCKET not set — using hardcoded state schema")
        return None
    try:
        response = s3_client.get_object(Bucket=CONFIG_S3_BUCKET, Key=STATE_SCHEMA_S3_KEY)
        _state_schema_cache = json.loads(response["Body"].read().decode("utf-8"))
        logger.info("State schema loaded | bucket=%s | key=%s", CONFIG_S3_BUCKET, STATE_SCHEMA_S3_KEY)
        return _state_schema_cache
    except Exception:
        logger.exception("Failed to load state schema from S3 — using hardcoded defaults")
        return None


def load_orchestrator_rules() -> Optional[List]:
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache
    if not CONFIG_S3_BUCKET:
        logger.warning("CONFIG_S3_BUCKET not set — fallback: matched_categories only")
        return None
    try:
        response = s3_client.get_object(Bucket=CONFIG_S3_BUCKET, Key=RULES_S3_KEY)
        _rules_cache = json.loads(response["Body"].read().decode("utf-8"))
        logger.info("Orchestrator rules loaded | bucket=%s | key=%s | count=%s",
                    CONFIG_S3_BUCKET, RULES_S3_KEY, len(_rules_cache))
        return _rules_cache
    except Exception:
        logger.exception("Failed to load orchestrator rules from S3 — fallback: matched_categories only")
        return None


# --------------------------------------------------------------------------------------
# Rule evaluation
# --------------------------------------------------------------------------------------

def evaluate_condition(condition: Dict, context: Dict) -> bool:
    """
    Supported ops:
      eq, neq, gt, gte, lt, lte  — standard comparisons
      mod                         — field % value == equals
      not_empty, empty            — truthy/falsy check
      contains                    — value in field
    """
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
        logger.exception("Condition evaluation failed | condition=%s | field=%s | value=%s", condition, field, field_value)
        return False


def should_invoke_orchestrator(
    session: Dict,
    prev_count: int,
    curr_count: int,
    matched_categories: List[str],
    speaker: str,
) -> Tuple[bool, str]:
    """
    Evaluate orchestrator rules from S3.
    ANY rule matching triggers (OR between rules).
    ALL conditions within a rule must match (AND within rule).

    Bucket-based threshold check:
      prev_bucket = prev_count // THRESHOLD
      curr_bucket = curr_count // THRESHOLD
      triggers when bucket changes and count >= THRESHOLD
    This correctly handles batches of records crossing the threshold at once.

    Falls back to matched_categories only if S3 rules missing.
    """
    rules = load_orchestrator_rules()

    # Build context for rule evaluation
    context = {
        "current_phase": session.get("current_phase", "CONTACT_OPEN"),
        "utterance_count": curr_count,
        "prev_utterance_count": prev_count,
        "matched_categories": matched_categories,
        "speaker": speaker,
        "contact_id": session.get("session_id", ""),
        "current_intent": session.get("current_intent", ""),
        "prev_bucket": prev_count // UTTERANCE_THRESHOLD,
        "curr_bucket": curr_count // UTTERANCE_THRESHOLD,
        "threshold_crossed": (curr_count // UTTERANCE_THRESHOLD > prev_count // UTTERANCE_THRESHOLD)
                              and curr_count >= UTTERANCE_THRESHOLD,
        **{k: v for k, v in session.items() if k != "qualification_context"},
    }

    if rules is None:
        # Fallback: only matched categories trigger
        if matched_categories:
            return True, f"fallback_matched_categories={matched_categories}"
        return False, ""

    for rule in rules:
        rule_name = rule.get("name", "unnamed")
        conditions = rule.get("conditions", [])
        if not conditions:
            continue
        if all(evaluate_condition(c, context) for c in conditions):
            logger.info("Rule matched | rule=%s | contact_id=%s", rule_name, context.get("contact_id"))
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
    """Add any fields from schema missing in session. Never overwrites existing values."""
    for key, default_value in schema.items():
        if key not in session:
            session[key] = default_value
        elif isinstance(default_value, dict) and isinstance(session.get(key), dict):
            session[key] = merge_with_schema(session[key], default_value)
    return session


def create_new_session(contact_id: str, now: str) -> Dict:
    schema = load_state_schema()

    hardcoded_defaults = {
        "session_id": contact_id,
        "bu": BUSINESS_UNIT,
        "colleague_id": None,
        "contact_started_at": now,
        "contact_ended_at": None,
        "current_phase": "CONTACT_OPEN",
        "phase_history": [],
        "qualification_context": {
            "channel_type": CHANNEL_TYPE,
            "utterances": [],
            "utterance_count": 0,
            "compacted_summary": None,
            "compaction_cursor": None,
            "compacted_at": None,
            "structured_inputs": {},
        },
        "current_intent": "UNABLE_TO_DETERMINE",
        "intent_history": [],
        "product_identification_result": [],
        "intent_detection_result": [],
        "summarization_result": [],
        "categorization_result": [],
        "colleague_decisions": [],
        "audit_trail": [],
        "ttl_seconds": SESSION_TTL_SECONDS,
        "created_at": now,
        "updated_at": now,
    }

    if schema is None:
        return hardcoded_defaults

    session = {**schema}
    # Runtime fields always set from code, never from schema
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
    """Load from Redis and merge with schema. Returns (session, is_existing)."""
    raw = redis_get(session_key)
    if not raw:
        logger.warning("No session found, creating new | contact_id=%s", contact_id)
        return create_new_session(contact_id, now), False

    session = json.loads(raw)
    schema = load_state_schema()
    if schema:
        session = merge_with_schema(session, schema)
    return session, True


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)


def redact_text_for_logs(text: str, max_len: int = 120) -> str:
    if not text:
        return ""
    snippet = text[:max_len]
    masked, run = [], 0
    for ch in snippet:
        if ch.isdigit():
            run += 1
            masked.append("x" if run >= 3 else ch)
        else:
            run = 0
            masked.append(ch)
    return "".join(masked)


def decode_kinesis_data(kinesis_data_b64: str) -> Dict:
    raw = base64.b64decode(kinesis_data_b64).decode("utf-8", errors="replace")
    return json.loads(raw)


def normalise_event_type(event_type: Optional[str]) -> str:
    if not event_type:
        return ""
    return EVENT_TYPE_ALIASES.get(str(event_type).upper(), str(event_type).upper())


def extract_utterances_from_segments(segments: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Extract all utterances and matched categories from a SEGMENTS payload.
    Returns (utterances, matched_categories).
    """
    utterances = []
    matched_categories = []

    for seg in segments or []:
        if not isinstance(seg, dict):
            continue

        # Categories
        if "Categories" in seg and isinstance(seg["Categories"], dict):
            matched_categories = seg["Categories"].get("MatchedCategories", []) or []

        # Transcript segment
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

        # Utterance segment
        utterance = seg.get("Utterance")
        if isinstance(utterance, dict):
            content = (utterance.get("ParticipantContent")
                       or utterance.get("PartialContent")
                       or utterance.get("Content") or "")
            if content.strip():
                role = utterance.get("ParticipantRole") or utterance.get("ParticipantId") or "CUSTOMER"
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

def lambda_handler(event: Dict, context):
    request_id = getattr(context, "aws_request_id", "") if context else ""

    if not isinstance(event, dict) or "Records" not in event:
        # Direct invoke
        try:
            result = process_event(event, request_id=request_id)
            return {"statusCode": 200, "body": safe_json_dumps(result)}
        except Exception as exc:
            logger.exception("Direct invoke failed | request_id=%s", request_id)
            return {"statusCode": 500, "body": safe_json_dumps({"error": str(exc)})}

    records = event.get("Records", [])
    batch_failures = []
    logger.info("Kinesis batch started | records=%s | request_id=%s", len(records), request_id)

    # Group records by contact_id for efficient Redis access
    contacts: Dict[str, Dict] = {}

    for i, record in enumerate(records):
        seq = record.get("kinesis", {}).get("sequenceNumber", "")
        try:
            payload = decode_kinesis_data(record["kinesis"]["data"])
            event_type = normalise_event_type(payload.get("EventType"))
            contact_id = payload.get("ContactId") or payload.get("contactId")

            if not contact_id:
                logger.warning("No contact_id | seq=%s | keys=%s", seq, list(payload.keys()))
                continue

            if contact_id not in contacts:
                contacts[contact_id] = {
                    "utterances": [],
                    "matched_categories": [],
                    "event_type": event_type,
                    "payload": payload,
                }

            if event_type == "SEGMENTS":
                segments = payload.get("Segments") or payload.get("segments") or []
                utterances, matched_categories = extract_utterances_from_segments(segments)
                contacts[contact_id]["utterances"].extend(utterances)
                if matched_categories:
                    contacts[contact_id]["matched_categories"] = matched_categories
            else:
                # For CONTACT_INITIATED / CONTACT_DISCONNECTED keep latest event type
                contacts[contact_id]["event_type"] = event_type

            logger.info("Record decoded | seq=%s | eventType=%s | contactId=%s", seq, event_type, contact_id)

        except Exception:
            logger.exception("Record decode failed | seq=%s | request_id=%s", seq, request_id)
            if seq:
                batch_failures.append({"itemIdentifier": seq})

    # Process each contact
    for contact_id, data in contacts.items():
        try:
            process_contact(
                contact_id=contact_id,
                event_type=data["event_type"],
                new_utterances=data["utterances"],
                matched_categories=data["matched_categories"],
                payload=data["payload"],
                request_id=request_id,
            )
        except Exception:
            logger.exception("Contact processing failed | contact_id=%s | request_id=%s", contact_id, request_id)

    logger.info("Kinesis batch complete | total=%s | failures=%s | request_id=%s",
                len(records), len(batch_failures), request_id)
    return {"batchItemFailures": batch_failures}


# --------------------------------------------------------------------------------------
# Core processing
# --------------------------------------------------------------------------------------

def process_event(event: Dict, request_id: str = "") -> Dict:
    """Direct invoke handler — single event."""
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
        payload=event,
        request_id=request_id,
    )


def process_contact(
    contact_id: str,
    event_type: str,
    new_utterances: List[Dict],
    matched_categories: List[str],
    payload: Dict,
    request_id: str = "",
) -> Dict:
    now = utc_now_iso()
    session_key = f"session:{contact_id}"

    logger.info("Processing contact | event_type=%s | contact_id=%s | request_id=%s",
                event_type, contact_id, request_id)

    # ── CONTACT_INITIATED ──────────────────────────────────────────────
    if event_type == "CONTACT_INITIATED":
        session = create_new_session(contact_id, now)
        redis_set(session_key, safe_json_dumps(session))
        logger.info("Session created | contact_id=%s | ttl=%ss", contact_id, SESSION_TTL_SECONDS)
        return {"contact_id": contact_id, "event_type": event_type}

    # ── CONTACT_DISCONNECTED ───────────────────────────────────────────
    if event_type == "CONTACT_DISCONNECTED":
        session, is_existing = load_session(session_key, contact_id, now)
        session["contact_ended_at"] = now
        session["updated_at"] = now
        session["current_phase"] = "CONTACT_CLOSED"
        redis_set(session_key, safe_json_dumps(session))
        logger.info("Session closed | contact_id=%s | total_utterances=%s",
                    contact_id, session["qualification_context"].get("utterance_count", 0))
        return {"contact_id": contact_id, "event_type": event_type}

    # ── SEGMENTS ───────────────────────────────────────────────────────
    if not new_utterances:
        logger.info("No utterances in segment | contact_id=%s", contact_id)
        return {"contact_id": contact_id, "event_type": event_type, "utterances": 0}

    session, _ = load_session(session_key, contact_id, now)
    qc = session["qualification_context"]
    prev_count = qc.get("utterance_count", 0)

    # Append utterances
    qc["utterances"].extend(new_utterances)
    qc["utterance_count"] = len(qc["utterances"])
    curr_count = qc["utterance_count"]

    # Merge matched categories
    if matched_categories:
        existing = session.get("categorization_result") or []
        for cat in matched_categories:
            if cat not in existing:
                existing.append(cat)
        session["categorization_result"] = existing
        logger.info("Categories matched | contact_id=%s | categories=%s", contact_id, matched_categories)

    session["updated_at"] = now
    redis_set(session_key, safe_json_dumps(session))

    logger.info("Utterances stored | contact_id=%s | added=%s | total=%s | speakers=%s",
                contact_id, len(new_utterances), curr_count,
                [u["speaker"] for u in new_utterances])

    # Orchestrator check
    speaker = new_utterances[-1]["speaker"] if new_utterances else "CUSTOMER"
    invoke, reason = should_invoke_orchestrator(session, prev_count, curr_count, matched_categories, speaker)

    if invoke:
        logger.info("Orchestrator triggered | contact_id=%s | reason=%s | utterance_count=%s",
                    contact_id, reason, curr_count)
        invoke_ai_orchestrator(session, contact_id)

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

def invoke_ai_orchestrator(session: Dict, contact_id: str) -> None:
    runtime_arn = os.environ.get("AGENTCORE_RUNTIME_ARN") or os.environ.get("AGENT_RUNTIME_ARN")
    if not runtime_arn:
        logger.error("Missing env var: AGENTCORE_RUNTIME_ARN | contact_id=%s", contact_id)
        return

    try:
        sid = re.sub(r"[^a-zA-Z0-9_-]", "_", contact_id)
        if len(sid) < 33:
            sid = sid + "-" + "0" * (33 - len(sid) - 1)

        phase = session.get("current_phase", "CONTACT_OPEN")
        event_name = PHASE_EVENT_MAP.get(phase, "CONTACT_INITIATED")
        payload = {
            "session_id": contact_id,
            "event": event_name,
            "current_phase": phase,
        }

        logger.info("AgentCore invoke started | contact_id=%s | phase=%s | event=%s",
                    contact_id, phase, event_name)

        response = agent_client.invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            contentType="application/json",
            accept="application/json",
            runtimeSessionId=sid[:64],
            payload=json.dumps(payload).encode(),
        )

        body = response.get("response") or response.get("body")
        result = body.read().decode("utf-8") if hasattr(body, "read") else str(body)
        logger.info("AgentCore invoke successful | contact_id=%s | response=%s", contact_id, result[:200])

    except Exception:
        logger.exception("AgentCore invoke failed | contact_id=%s"
