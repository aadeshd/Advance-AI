import base64
import binascii
import json
import logging
import os
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
import redis
from botocore.config import Config

from transformer import TransformerFactory, should_invoke_orchestrator


# --------------------------------------------------------------------------------------
# Runtime configuration only.
# Business rules, state schema, event list and transformer mappings are outside Lambda.
# --------------------------------------------------------------------------------------

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "86400"))

REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
# NOTE: template sets REDIS_TLS (not REDIS_SSL) - was previously reading the wrong
# env var name here, which silently forced plaintext connections regardless of the
# actual cluster's transit-encryption setting.
REDIS_TLS = os.environ.get("REDIS_TLS", "false").lower() == "true"
REDIS_AUTH_SECRET = os.environ.get("REDIS_AUTH_SECRET", "")

BUSINESS_UNIT = os.environ.get("BUSINESS_UNIT", "BUK")
CHANNEL_TYPE = os.environ.get("CHANNEL_TYPE", "PHONE")
AWS_ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID")
AWS_REGION = os.environ.get("AWS_REGION", "eu-central-1")

CONFIG_S3_BUCKET = os.environ.get("CONFIG_S3_BUCKET")
STATE_SCHEMA_S3_KEY = os.environ.get(
    "StateSchemaS3Key",
    "lambda/interaction-stream/config/dev/state_schema.json",
)
RULES_S3_KEY = os.environ.get(
    "RulesS3Key",
    "lambda/interaction-stream/config/dev/orchestrator_rules.json",
)
EVENT_CONFIG_S3_KEY = os.environ.get(
    "EVENT_CONFIG_S3_KEY",
    "lambda/interaction-stream/config/dev/event_config.json",
)

AGENTCORE_RUNTIME_ARN = os.environ.get("AGENTCORE_RUNTIME_ARN") or os.environ.get(
    "AGENT_RUNTIME_ARN"
)
AGENTCORE_HOST = f"bedrock-agentcore.{AWS_REGION}.amazonaws.com"

if not AWS_ACCOUNT_ID:
    raise ValueError("AWS_ACCOUNT_ID environment variable is required")

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

S3_CLIENT_CONFIG = Config(connect_timeout=5, read_timeout=10)
AGENTCORE_CLIENT_CONFIG = Config(
    connect_timeout=5,
    read_timeout=10,
    retries={"max_attempts": 1},  # keep at 1 while diagnosing; raise once stable
)

_cold_start_t0 = time.time()

logger.info(
    "COLD START | region=%s | redis_host=%s | redis_port=%s | redis_tls=%s | "
    "config_bucket=%s | agentcore_runtime_arn=%s",
    AWS_REGION,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_TLS,
    CONFIG_S3_BUCKET,
    AGENTCORE_RUNTIME_ARN,
)

secretsmanager_client = boto3.client("secretsmanager", region_name=AWS_REGION)

_t = time.time()
try:
    _secret_response = secretsmanager_client.get_secret_value(SecretId=REDIS_AUTH_SECRET)
    _secret = json.loads(_secret_response["SecretString"])
    redis_auth_token = _secret["authToken"]
    logger.info(
        "Redis auth secret fetched | secret_arn=%s | elapsed=%.3fs",
        REDIS_AUTH_SECRET,
        time.time() - _t,
    )
except Exception:
    logger.exception(
        "Failed to fetch Redis auth secret | secret_arn=%s | elapsed=%.3fs",
        REDIS_AUTH_SECRET,
        time.time() - _t,
    )
    raise

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    ssl=REDIS_TLS,
    password=redis_auth_token,
    socket_connect_timeout=3,
    socket_timeout=3,
    retry_on_timeout=True,
    decode_responses=True,
)

s3_client = boto3.client("s3", region_name=AWS_REGION, config=S3_CLIENT_CONFIG)

agent_client = boto3.client(
    "bedrock-agentcore",
    region_name=AWS_REGION,
    config=AGENTCORE_CLIENT_CONFIG,
)

logger.info("COLD START complete | elapsed=%.3fs", time.time() - _cold_start_t0)

_state_schema_cache: Optional[Dict[str, Any]] = None
_rules_cache: Optional[List[Dict[str, Any]]] = None
_event_config_cache: Optional[Dict[str, Any]] = None
_transformer_cache: Dict[str, Any] = {}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))


def check_dns(hostname: str) -> None:
    """
    Diagnostic only: resolves a hostname and logs how long it took and what it
    resolved to. Helps distinguish a DNS-resolution hang from a TCP-connect hang,
    since botocore's connect_timeout does not always bound getaddrinfo() calls.
    """
    t0 = time.time()
    try:
        resolved_ip = socket.gethostbyname(hostname)
        logger.info(
            "DNS check OK | host=%s | resolved_ip=%s | elapsed=%.3fs",
            hostname,
            resolved_ip,
            time.time() - t0,
        )
    except Exception:
        logger.exception(
            "DNS check FAILED | host=%s | elapsed=%.3fs",
            hostname,
            time.time() - t0,
        )


# --------------------------------------------------------------------------------------
# Reading input/configuration data
# --------------------------------------------------------------------------------------

def read_json_from_s3(key: str) -> Any:
    if not CONFIG_S3_BUCKET:
        raise ValueError(
            "CONFIG_S3_BUCKET environment variable is required to load configuration"
        )

    t0 = time.time()
    try:
        response = s3_client.get_object(
            Bucket=CONFIG_S3_BUCKET,
            Key=key,
            ExpectedBucketOwner=AWS_ACCOUNT_ID,
        )
        data = json.loads(response["Body"].read().decode("utf-8"))
        logger.info(
            "S3 object loaded | bucket=%s | key=%s | elapsed=%.3fs",
            CONFIG_S3_BUCKET,
            key,
            time.time() - t0,
        )
        return data
    except Exception:
        logger.exception(
            "S3 object load FAILED | bucket=%s | key=%s | elapsed=%.3fs",
            CONFIG_S3_BUCKET,
            key,
            time.time() - t0,
        )
        raise


def load_state_schema() -> Dict[str, Any]:
    global _state_schema_cache

    if _state_schema_cache is None:
        _state_schema_cache = read_json_from_s3(STATE_SCHEMA_S3_KEY)

    return _state_schema_cache


def load_orchestrator_rules() -> List[Dict[str, Any]]:
    global _rules_cache

    if _rules_cache is None:
        _rules_cache = read_json_from_s3(RULES_S3_KEY)
        logger.info("Orchestrator rules cached | count=%s", len(_rules_cache))

    return _rules_cache


def load_event_config() -> Dict[str, Any]:
    global _event_config_cache

    if _event_config_cache is None:
        _event_config_cache = read_json_from_s3(EVENT_CONFIG_S3_KEY)

    return _event_config_cache


def get_transformer(channel_type: str):
    channel_key = (channel_type or CHANNEL_TYPE).upper()

    if channel_key not in _transformer_cache:
        _transformer_cache[channel_key] = TransformerFactory.create(
            channel_type=channel_key,
            event_config=load_event_config(),
            state_schema=load_state_schema(),
            session_ttl_seconds=SESSION_TTL_SECONDS,
            business_unit=BUSINESS_UNIT,
        )

    return _transformer_cache[channel_key]


def decode_kinesis_data(kinesis_data: str) -> Dict[str, Any]:
    """
    Decode Kinesis payload.

    Handles:
    1. Standard base64 encoded JSON
    2. Double base64 encoded JSON
    """
    try:
        decoded = base64.b64decode(kinesis_data).decode("utf-8")

        try:
            return json.loads(decoded)
        except json.JSONDecodeError:
            decoded_twice = base64.b64decode(decoded).decode("utf-8")
            return json.loads(decoded_twice)

    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.exception("Invalid Kinesis record payload")
        raise ValueError("Invalid Kinesis record payload") from exc


def read_records(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Read incoming Lambda event.

    Supports:
    1. Direct payload invocation
    2. Kinesis Records batch
    """
    if "Records" not in event:
        return [event]

    payloads: List[Dict[str, Any]] = []

    for record in event.get("Records", []):
        payloads.append(decode_kinesis_data(record["kinesis"]["data"]))

    logger.info("Kinesis batch decoded | record_count=%s", len(payloads))

    return payloads


# --------------------------------------------------------------------------------------
# Validating_event
# --------------------------------------------------------------------------------------

def validate_event(payload: Dict[str, Any], transformer) -> Dict[str, Any]:
    return transformer.validate_event(payload)


# --------------------------------------------------------------------------------------
# Redis operations
# --------------------------------------------------------------------------------------

def redis_get(key: str) -> Optional[str]:
    t0 = time.time()
    try:
        value = redis_client.get(key)
        logger.info("Redis GET | key=%s | hit=%s | elapsed=%.3fs", key, value is not None, time.time() - t0)
    except Exception:
        logger.exception("Redis GET FAILED | key=%s | elapsed=%.3fs", key, time.time() - t0)
        raise

    if isinstance(value, bytes):
        return value.decode("utf-8")

    return value


def redis_set(key: str, value: str, ttl: int = SESSION_TTL_SECONDS) -> None:
    t0 = time.time()
    try:
        redis_client.set(key, value, ex=ttl)
        logger.info("Redis SET | key=%s | ttl=%s | elapsed=%.3fs", key, ttl, time.time() - t0)
    except Exception:
        logger.exception("Redis SET FAILED | key=%s | elapsed=%.3fs", key, time.time() - t0)
        raise


def load_session(
    session_key: str,
    contact_id: str,
    transformer,
) -> Tuple[Dict[str, Any], bool]:
    now = utc_now_iso()
    raw = redis_get(session_key)

    if not raw:
        logger.info(
            "No existing session found; creating session | contact_id=%s",
            contact_id,
        )
        return transformer.create_session(contact_id=contact_id, now=now), False

    session = json.loads(raw)
    return transformer.merge_with_schema(session), True


def create_or_update_session(
    payload: Dict[str, Any],
    request_id: str = "",
) -> Tuple[Dict[str, Any], bool, str, Any]:
    channel_type = payload.get("Channel") or payload.get("ChannelType") or CHANNEL_TYPE
    transformer = get_transformer(channel_type)

    validated_event = validate_event(payload=payload, transformer=transformer)

    contact_id = validated_event["contact_id"]
    session_key = f"ail:{contact_id}"

    session, existed = load_session(
        session_key=session_key,
        contact_id=contact_id,
        transformer=transformer,
    )

    transformed = transformer.transform_event(
        validated_event=validated_event,
        request_id=request_id,
    )

    updated_session = transformer.apply_event_to_session(
        session=session,
        transformed=transformed,
        now=utc_now_iso(),
    )

    redis_set(key=session_key, value=safe_json_dumps(updated_session))

    return (
        updated_session,
        existed,
        transformed.get("orchestrator_reason", ""),
        transformer,
    )


# --------------------------------------------------------------------------------------
# Sending data to the Orchestrator
# --------------------------------------------------------------------------------------

def invoke_ai_orchestrator(
    session: Dict[str, Any],
    contact_id: str,
    event_category: str,
    event_action: str,
) -> None:
    if not AGENTCORE_RUNTIME_ARN:
        logger.error(
            "Missing env var: AGENTCORE_RUNTIME_ARN | contact_id=%s",
            contact_id,
        )
        return

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

    # One-time-per-cold-start DNS diagnostic - safe to remove once the timeout
    # root cause is confirmed and fixed.
    check_dns(AGENTCORE_HOST)

    t0 = time.time()
    logger.info(
        "Invoking AgentCore | contact_id=%s | runtime_arn=%s | event_category=%s | event_action=%s",
        contact_id,
        AGENTCORE_RUNTIME_ARN,
        event_category,
        event_action,
    )

    try:
        response = agent_client.invoke_agent_runtime(
            agentRuntimeArn=AGENTCORE_RUNTIME_ARN,
            payload=json.dumps(payload).encode("utf-8"),
            contentType="application/json",
        )
        logger.info(
            "AgentCore invoke SUCCEEDED | contact_id=%s | elapsed=%.3fs | response_keys=%s",
            contact_id,
            time.time() - t0,
            list(response.keys()) if isinstance(response, dict) else type(response),
        )
    except Exception:
        logger.exception(
            "AgentCore invoke FAILED | contact_id=%s | event_category=%s | event_action=%s | elapsed=%.3fs",
            contact_id,
            event_category,
            event_action,
            time.time() - t0,
        )


def send_to_orchestrator_if_required(
    session: Dict[str, Any],
    reason: str,
    transformer,
) -> bool:
    contact_id = session["session_id"]
    rules = load_orchestrator_rules()

    invoke, matched_reason = should_invoke_orchestrator(
        session=session,
        rules=rules,
        fallback_reason=reason,
    )

    if not invoke:
        logger.info("Orchestrator NOT invoked | contact_id=%s | reason=%s", contact_id, reason)
        return False

    orchestrator_event = transformer.get_orchestrator_event(matched_reason=matched_reason)

    invoke_ai_orchestrator(
        session=session,
        contact_id=contact_id,
        event_category=orchestrator_event["event_category"],
        event_action=orchestrator_event["event_action"],
    )

    return True


# --------------------------------------------------------------------------------------
# Lambda entrypoint - intentionally thin orchestration only
# --------------------------------------------------------------------------------------

def process_event(payload: Dict[str, Any], request_id: str = "") -> Dict[str, Any]:
    session, existed, reason, transformer = create_or_update_session(
        payload=payload,
        request_id=request_id,
    )

    orchestrator_invoked = send_to_orchestrator_if_required(
        session=session,
        reason=reason,
        transformer=transformer,
    )

    return {
        "contact_id": session["session_id"],
        "session_existed": existed,
        "orchestrator_invoked": orchestrator_invoked,
        "utterance_count": session.get("qualification_context", {}).get("utterance_count", 0),
        "current_phase": session.get("current_phase"),
    }


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    request_id = getattr(context, "aws_request_id", "") if context else ""
    remaining_ms = context.get_remaining_time_in_millis() if context else None
    logger.info(
        "Invocation START | request_id=%s | remaining_time_ms=%s",
        request_id,
        remaining_ms,
    )

    t0 = time.time()
    results: List[Dict[str, Any]] = []

    for payload in read_records(event):
        results.append(process_event(payload=payload, request_id=request_id))

    logger.info(
        "Invocation COMPLETE | request_id=%s | processed=%s | elapsed=%.3fs",
        request_id,
        len(results),
        time.time() - t0,
    )

    return {
        "statusCode": 200,
        "body": json.dumps({"processed": len(results), "results": results}),
    }
