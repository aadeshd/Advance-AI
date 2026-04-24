import boto3
import json
from urllib.parse import unquote_plus  # ← handles URL encoding

s3_client = boto3.client("s3")

def lambda_handler(event, context):
    
    # ── Extract bucket and key ────────────────────────────────
    bucket = event["detail"]["bucket"]["name"]
    key = unquote_plus(                          # ← fixes URL encoding
        event["detail"]["object"]["key"]
    )
    
    print(f"📂 Reading: s3://{bucket}/{key}")    # ← log exact path
    
    # ── Read from S3 ──────────────────────────────────────────
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    
    # ── Decode safely ─────────────────────────────────────────
    # Strip BOM if present (common in Windows-saved files)
    content = raw.decode("utf-8-sig").strip()
    
    print(f"📄 First 200 chars: {content[:200]}")  # ← log raw content
    print(f"📄 Last 50 chars:   {content[-50:]}")  # ← check for trailing garbage
    
    # ── Parse JSON ────────────────────────────────────────────
    try:
        utterances = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse failed at: {e}")
        print(f"   Around char {e.pos}: ...{content[max(0,e.pos-50):e.pos+50]}...")
        raise

    contact_id = utterances[0].get("ContactId", "unknown") if utterances else "unknown"
    print(f"✅ Loaded {len(utterances)} utterances for {contact_id}")

    return {
        "utterances": utterances,
        "contact_id": contact_id,
        "total": len(utterances)
    }
