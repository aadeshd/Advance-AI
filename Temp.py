def decode_kinesis_data(kinesis_data: str) -> Dict:
    """
    Kinesis always base64-encodes the data field on the stream,
    regardless of what the producer wrote. Decode that first,
    then parse the resulting JSON string.
    """
    try:
        raw = base64.b64decode(kinesis_data).decode("utf-8")
        return json.loads(raw)
    except (base64.binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to base64-decode Kinesis record: {kinesis_data[:80]}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Kinesis data is not valid JSON after base64 decode: {kinesis_data[:80]}") from exc
