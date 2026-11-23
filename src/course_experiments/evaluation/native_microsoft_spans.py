from __future__ import annotations

import bisect
import hashlib
import json


def token_windows(text, encoding, *, size=600, overlap=100, trailing_overlap=False):
    """保留官方token窗口；引用仅取窗口字节范围内完整的原文字符。"""
    if type(size) is not int or type(overlap) is not int or not 0 <= overlap < size:
        raise ValueError("Explicit valid official token size/overlap required")
    tokens = encoding.encode(text)
    pieces = [encoding.decode_single_token_bytes(token) for token in tokens]
    source = text.encode("utf-8")
    if b"".join(pieces) != source:
        raise ValueError("Token bytes do not reconstruct the exact original UTF-8 source")
    token_bytes = [0]
    for piece in pieces:
        token_bytes.append(token_bytes[-1] + len(piece))
    character_bytes = [0]
    for character in text:
        character_bytes.append(character_bytes[-1] + len(character.encode("utf-8")))
    rows = []
    for start in range(0, len(tokens), size - overlap):
        end = min(start + size, len(tokens))
        left, right = token_bytes[start], token_bytes[end]
        char_left = bisect.bisect_left(character_bytes, left)
        char_right = bisect.bisect_right(character_bytes, right) - 1
        native_text = encoding.decode(tokens[start:end])
        if native_text != source[left:right].decode("utf-8", errors="replace"):
            raise ValueError("Official token decoder differs from preserved window bytes")
        rows.append({
            "token_start": start, "token_end": end, "n_tokens": end - start,
            "byte_start": left, "byte_end": right,
            "passage_start": char_left, "passage_end": max(char_left, char_right),
            "covered_character_start": bisect.bisect_right(character_bytes, left) - 1,
            "covered_character_end": bisect.bisect_left(character_bytes, right),
            "left_partial_character": left not in character_bytes,
            "right_partial_character": right not in character_bytes,
            "decoded_text_is_exact_quote": left in character_bytes and right in character_bytes,
            "native_text": native_text,
            "source_utf8_sha256": hashlib.sha256(source).hexdigest(),
            "window_bytes_sha256": hashlib.sha256(source[left:right]).hexdigest(),
            "token_ids_sha256": hashlib.sha256(json.dumps(tokens[start:end], separators=(",", ":")).encode()).hexdigest(),
            "quote_policy": "complete_original_UTF8_characters_inside_indexed_token_bytes",
        })
        if end == len(tokens) and not trailing_overlap:
            break
    return rows


def locate_native_text(text, native_text, encoding, *, size=600, overlap=100):
    # 重复窗口保留全部位置，不以首次字符串命中代替真实分词边界。
    return [row for row in token_windows(text, encoding, size=size, overlap=overlap)
            if row["native_text"] == native_text]


def evidence_intersections(passage, window):
    text = passage["text"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != window["source_utf8_sha256"]:
        raise ValueError("Token mapping belongs to another source passage")
    rows = []
    for span in passage["evidence_spans"]:
        left = max(span["start"], window["passage_start"])
        right = min(span["end"], window["passage_end"])
        if left >= right:
            continue
        rows.append({"evidence_id": span["evidence_id"], "text": text[left:right],
                     "passage_start": left, "passage_end": right,
                     "evidence_start": left - span["start"], "evidence_end": right - span["start"],
                     "source_byte_start": len(text[:left].encode("utf-8")),
                     "source_byte_end": len(text[:right].encode("utf-8")),
                     "source_kind": "original_evidence_fragment",
                     "native_token_window": {key: value for key, value in window.items() if key != "native_text"},
                     "partial_native_character_bytes_are_not_quoted": True})
    return rows
