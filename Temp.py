# Average milliseconds per word (tune these to your data)
AVG_MS_PER_WORD = {
    "AGENT": 280,       # agents speak slightly faster, scripted
    "CUSTOMER": 320,    # customers speak slower, natural speech
    "DEFAULT": 300      # fallback
}

def calculate_delay(utterance: dict) -> float:
    """
    Calculate delay in seconds based on word count and speaker.
    Returns delay in seconds (float) for time.sleep().
    """
    content = utterance.get("content", "") or ""
    participant = utterance.get("participantid", "DEFAULT").upper()

    word_count = len(content.split())
    ms_per_word = AVG_MS_PER_WORD.get(participant, AVG_MS_PER_WORD["DEFAULT"])

    total_ms = word_count * ms_per_word

    # Optional: use actual offset difference if available as a sanity cap
    begin = utterance.get("beginoffsetmillis", 0)
    end = utterance.get("endoffsetmillis", 0)
    actual_duration_ms = end - begin if end > begin else total_ms

    # Use whichever is smaller — don't wait longer than actual duration
    final_ms = min(total_ms, actual_duration_ms) if actual_duration_ms > 0 else total_ms

    return final_ms / 1000.0   # convert to seconds

import boto3
import json
import time

# ── Kinesis client setup ──────────────────────────────────────────────────────
kinesis_client = boto3.client(
    "kinesis",
    region_name="your-region",          # e.g. "us-east-1"
    aws_access_key_id="YOUR_KEY",       # or use IAM role / env vars
    aws_secret_access_key="YOUR_SECRET"
)

STREAM_NAME = "your-kinesis-stream-name"


def send_to_kinesis(utterance_json: dict):
    """
    Send a single utterance JSON to Kinesis stream.
    Partition key = ContactId (keeps one conversation on same shard).
    """
    contact_id = utterance_json.get("ContactId", "unknown")
    payload = json.dumps(utterance_json)

    response = kinesis_client.put_record(
        StreamName=STREAM_NAME,
        Data=payload.encode("utf-8"),
        PartitionKey=contact_id         # same shard for same conversation
    )

    shard = response.get("ShardId")
    seq = response.get("SequenceNumber")
    print(f"  ✅ Sent → Shard: {shard} | Seq: {seq[-6:]}")
    return response


def stream_utterances_to_kinesis(utterances_json_path: str):
    """
    Read utterance list JSON and stream to Kinesis with paced delays.
    """
    with open(utterances_json_path) as f:
        utterance_list = json.load(f)

    total = len(utterance_list)
    contact_id = utterance_list[0].get("ContactId", "unknown") if utterance_list else "unknown"

    print(f"\n🚀 Streaming {total} utterances for Contact: {contact_id}")
    print("=" * 60)

    for item in utterance_list:
        utterance = item.get("Transcript", {})
        idx = item.get("UtteranceIndex", 0)
        participant = utterance.get("participantid", "?").upper()
        content = utterance.get("content", "")
        word_count = len(content.split())

        # Calculate delay
        delay_seconds = calculate_delay(utterance)

        print(f"\n[{idx+1}/{total}] {participant}")
        print(f"  Words     : {word_count}")
        print(f"  Delay     : {delay_seconds:.2f}s")
        print(f"  Content   : {content[:60]}{'...' if len(content)>60 else ''}")

        # ⏳ Wait (simulate speech duration)
        time.sleep(delay_seconds)

        # 📤 Send to Kinesis
        send_to_kinesis(item)

    print(f"\n✅ Done — all {total} utterances streamed for {contact_id}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
import glob

utterance_files = glob.glob("output/*_utterances.json")

for file_path in utterance_files:
    stream_utterances_to_kinesis(file_path)
    print("\n⏸️  Conversation complete. Starting next...\n")
    time.sleep(2)   # small gap between conversations


def dry_run(utterances_json_path: str):
    """Test delay logic without sending to Kinesis."""
    with open(utterances_json_path) as f:
        utterance_list = json.load(f)

    total_simulated_time = 0

    for item in utterance_list:
        utterance = item.get("Transcript", {})
        content = utterance.get("content", "")
        participant = utterance.get("participantid", "?").upper()
        delay = calculate_delay(utterance)
        total_simulated_time += delay

        print(f"{participant:10} | {len(content.split()):3} words "
              f"| {delay:.2f}s delay | {content[:50]}")

    print(f"\n⏱️  Total simulated conversation time: {total_simulated_time:.1f}s "
          f"({total_simulated_time/60:.1f} mins)")

# Run this first!
dry_run("output/abc-123_utterances.json")
