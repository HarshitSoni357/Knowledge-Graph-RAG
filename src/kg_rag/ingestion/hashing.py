import hashlib


def create_document_id(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def create_chunk_id(
    document_id: str,
    chunk_index: int,
) -> str:
    raw = f"{document_id}:{chunk_index}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()