"""Deterministic, completely local H3 Context-IR compilation pipeline."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from .model import (
    ArtifactRecord,
    AudioReferencePolicy,
    AutoAdjustment,
    COMPILER_VERSION,
    CompilationResult,
    CompilationStatus,
    CompileRequest,
    ContextIRDocument,
    Diagnostic,
    DialogueEvent,
    EmbeddedVideoAudioPolicy,
    GenerationMode,
    MusicPolicy,
    Provenance,
    ReferenceKind,
    ReferenceLabel,
    ReferenceOrigin,
    ReferenceSpec,
    RetentionEntry,
    Severity,
    SourceOutfitPolicy,
    SubjectDefinition,
    stable_sha256,
)
from .parser import (
    assign_dialogue_to_shots,
    discover_reference_labels,
    is_h3_context_ir,
    parse_dialogue,
    split_prompt_into_shots,
)
from .renderer import render_context_ir
from .validator import has_fatal, validate_document, validate_rendered_ir


STYLE_DIRECTIONS = {
    "natural": "",
    "cinematic": (
        "Cinematic composition, expressive camera movement, natural temporal motion, "
        "and detailed lighting."
    ),
    "photoreal": (
        "Photorealistic presentation with physically plausible lighting, natural materials, "
        "and coherent fine detail."
    ),
    "anime": (
        "Polished anime film style with expressive motion, clean linework, and rich color design."
    ),
    "illustration": (
        "Painterly editorial illustration with tactile textures, art-directed color, and "
        "graceful motion."
    ),
    "product": (
        "Premium commercial product film with controlled studio lighting, precise materials, "
        "and elegant camera work."
    ),
    "moody": (
        "Atmospheric presentation with dramatic shadows, restrained color, and cinematic motion."
    ),
}

AUDIO_DIRECTIONS = {
    "auto": "",
    "dialogue": (
        "Prioritize clear, natural, intelligible dialogue synchronized only with visible speech."
    ),
    "ambience": (
        "Prioritize a detailed spatial soundscape with convincing room tone and environmental "
        "ambience."
    ),
    "effects": (
        "Prioritize synchronized foley and physical sound effects with natural perspective."
    ),
    "music": "Prioritize the explicitly requested musical layer without inventing vocals.",
    "quiet": (
        "Use a restrained soundscape with subtle room tone and only necessary diegetic sounds."
    ),
}

_VIDEO_AUDIO_SOURCE_PATTERN = (
    r"(?:(?:reference\s+)?video audio|(?:reference\s+)?video soundtrack|"
    r"embedded video audio|original video audio|"
    r"Video\s+[1-9][0-9]*\s*(?:'s\s*)?(?:audio|soundtrack)|"
    r"<Video\s+[1-9][0-9]*>\s*(?:'s\s*)?(?:audio|soundtrack)|"
    r"(?:audio|soundtrack)\s+(?:from|of)\s+(?:the\s+)?(?:reference\s+)?video|"
    r"動画(?:の|に含まれる|内の)(?:埋め込み)?音声(?:トラック)?|動画音声|"
    r"動画\s*[1-9][0-9]*\s*(?:の|に含まれる|内の)?音声(?:トラック)?|"
    r"元動画(?:の)?音声(?:トラック)?|"
    r"<Video\s+[1-9][0-9]*>\s*の?音声(?:トラック)?)"
)
_VIDEO_AUDIO_REUSE_RE = re.compile(
    _VIDEO_AUDIO_SOURCE_PATTERN
    + r".{0,24}(?:reuse|copy|preserve|keep|そのまま|再利用|保持|コピー)",
    re.IGNORECASE,
)
_VIDEO_AUDIO_REUSE_PREFIX_RE = re.compile(
    r"(?:reuse|copy|preserve|keep)\s+(?:the\s+)?"
    + _VIDEO_AUDIO_SOURCE_PATTERN
    + r"|(?:再利用|保持|コピー)(?:する|して)?\s*"
    + _VIDEO_AUDIO_SOURCE_PATTERN,
    re.IGNORECASE,
)
_VIDEO_AUDIO_REFERENCE_RE = re.compile(
    _VIDEO_AUDIO_SOURCE_PATTERN
    + r".{0,48}(?:as\s+(?:a\s+)?(?:voice\s+|timbre\s+)?reference|"
    r"timbre|voice\s+reference|声質|話し方|参照(?:に|として))|"
    r"\buse\b\s+(?:the\s+)?"
    + _VIDEO_AUDIO_SOURCE_PATTERN,
    re.IGNORECASE,
)
_VIDEO_AUDIO_HARD_IGNORE_RE = re.compile(
    r"(?:ignore|exclude|do\s+not\s+use|don't\s+use|without)\s+(?:the\s+)?"
    + _VIDEO_AUDIO_SOURCE_PATTERN
    + r"|"
    + _VIDEO_AUDIO_SOURCE_PATTERN
    + r".{0,40}(?:ignore|exclude|do\s+not\s+use|don't\s+use|無視|使わない|参照しない)|"
    r"(?:動画(?:の|に含まれる|内の)?音声|元動画(?:の)?音声).{0,24}"
    r"(?:無視|使わない|参照しない)",
    re.IGNORECASE,
)
_VIDEO_AUDIO_NEGATED_REUSE_RE = re.compile(
    r"(?:do\s+not|don't|never)\s+(?:reuse|copy|preserve|keep)\s+(?:the\s+)?"
    + _VIDEO_AUDIO_SOURCE_PATTERN
    + r"|"
    + _VIDEO_AUDIO_SOURCE_PATTERN
    + r".{0,40}(?:do\s+not|don't|never)\s+(?:reuse|copy|preserve|keep)|"
    + _VIDEO_AUDIO_SOURCE_PATTERN
    + r".{0,32}(?:再利用しない|コピーしない|保持しない)",
    re.IGNORECASE,
)
_ANY_REFERENCE_TAG_RE = re.compile(
    r"<\s*(?:Picture|Video|Audio)\s+[1-9][0-9]*\s*>",
    re.IGNORECASE,
)
_PICTURE_TAG_PATTERN = r"<\s*Picture\s+[1-9][0-9]*\s*>"
_PICTURE_LIST_CONNECTOR_PATTERN = (
    r"(?:\s*,\s*(?:(?:and|or|&)\s+)?|\s+(?:and|or|&)\s+|"
    r"\s*(?:、|と|および|及び)\s*)"
)
_COORDINATED_PICTURE_LIST_RE = re.compile(
    rf"(?P<labels>{_PICTURE_TAG_PATTERN}(?:{_PICTURE_LIST_CONNECTOR_PATTERN}"
    rf"{_PICTURE_TAG_PATTERN})+)(?P<tail>[^\n.;。；！？!?]{{0,180}})",
    re.IGNORECASE,
)
_ROLE_CLAUSE_SEPARATOR_RE = re.compile(
    r"(?:[;\n。；！？!?]+|(?<!\d)\.(?!\d)|,\s*(?:while|whereas|but|and)\b\s*|、)",
    re.IGNORECASE,
)
_NEGATIVE_IDENTITY_RE = re.compile(
    r"(?i)(?:do\s+not|don't|never)\s+(?:copy|preserve|keep|use|reference).{0,48}"
    r"(?:person|character|identity|face|body)|"
    r"(?:person|character|identity|face|body).{0,32}(?:must\s+not|should\s+not)|"
    r"(?:人物|キャラクター|本人|顔|体形|体型|身体|外見).{0,24}"
    r"(?:参照しない|使わない|保持しない|維持しない|コピーしない)"
)
_EXPLICIT_PICTURE_ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "clothing",
        re.compile(
            r"(?i)(?:as|for)\s+(?:an?\s+)?(?:clothing|outfit|wardrobe|costume|garment|fashion)"
            r"(?:\s+design)?\s+reference|(?:衣装|服装|服|コスチューム|ファッション)"
            r"(?:だけ|のみ)?(?:の|として|用の)?(?:デザイン)?参照"
        ),
    ),
    (
        "style",
        re.compile(
            r"(?i)(?:as|for)\s+(?:an?\s+)?(?:art\s+|visual\s+)?style\s+reference|"
            r"(?:画風|作風|描画スタイル|ビジュアルスタイル)(?:だけ|のみ)?(?:の|として|用の)?参照"
        ),
    ),
    (
        "background",
        re.compile(
            r"(?i)(?:as|for)\s+(?:an?\s+)?(?:background|location|environment|setting|scene)"
            r"\s+reference|(?:背景|場所|風景|環境|舞台)(?:だけ|のみ)?(?:の|として|用の)?参照"
        ),
    ),
    (
        "object",
        re.compile(
            r"(?i)(?:as|for)\s+(?:an?\s+)?(?:object|product|prop|item|vehicle|architecture)"
            r"(?:\s+design)?\s+reference|(?:商品|製品|小物|物体|道具|乗り物|建物)"
            r"(?:だけ|のみ)?(?:の|として|用の)?(?:デザイン)?参照"
        ),
    ),
    (
        "character",
        re.compile(
            r"(?i)(?:as|for)\s+(?:an?\s+)?(?:character|person|identity|face|body)\s+reference|"
            r"(?:人物|キャラクター|本人|顔|体形|体型|外見)(?:の|として|用の)?参照"
        ),
    ),
)
_PICTURE_ROLE_KEYWORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "character",
        re.compile(
            r"(?i)\b(?:character|person|identity|face|facial|hairstyle|hair|body|wears|wearing)\b|"
            r"人物|キャラクター|本人|顔|髪|体形|体型|身体|外見|眼鏡|着て"
        ),
    ),
    (
        "clothing",
        re.compile(
            r"(?i)\b(?:clothing|outfit|wardrobe|costume|garment|fashion|dress|jacket)\b|"
            r"衣装|服装|コスチューム|ファッション|ドレス|ジャケット"
        ),
    ),
    (
        "style",
        re.compile(r"(?i)\b(?:art\s+style|visual\s+style|aesthetic|styling)\b|画風|作風"),
    ),
    (
        "background",
        re.compile(r"(?i)\b(?:background|location|environment|setting|scenery)\b|背景|場所|風景|舞台"),
    ),
    (
        "object",
        re.compile(r"(?i)\b(?:object|product|prop|item|vehicle|architecture)\b|商品|製品|小物|物体|道具|乗り物|建物"),
    ),
)
_USER_CONTROL_TOKENS = (
    "<d>",
    "</d>",
    "<|cutoff|>",
    "<|lyrics_start|>",
    "<|lyrics_end|>",
    "<|caption_start|>",
    "<|caption_end|>",
)


def _reserved_control_token(
    request: CompileRequest,
    *,
    raw_ir: bool,
) -> tuple[str, str] | None:
    tokens = _USER_CONTROL_TOKENS[2:] if raw_ir else _USER_CONTROL_TOKENS
    fields = (
        ("prompt", request.prompt),
        ("soundscape", request.soundscape),
        ("audio_direction", request.audio_direction),
        ("music_direction", request.music_direction),
        ("style_direction", request.style_direction),
    )
    for field_name, value in fields:
        lowered = value.lower()
        for token in tokens:
            if token.lower() in lowered:
                return field_name, token
    return None


def _picture_role_from_context(context: str) -> str:
    value = context.strip()
    if not value:
        return "neutral"
    for role, pattern in _EXPLICIT_PICTURE_ROLE_PATTERNS:
        if pattern.search(value):
            if role == "character" and _NEGATIVE_IDENTITY_RE.search(value):
                continue
            return role
    negative_identity = bool(_NEGATIVE_IDENTITY_RE.search(value))
    for role, pattern in _PICTURE_ROLE_KEYWORDS:
        if role == "character" and negative_identity:
            continue
        if pattern.search(value):
            return role
    return "neutral"


def _coordinated_picture_roles(
    reference: ReferenceSpec, text: str
) -> tuple[tuple[int, int, str], ...]:
    resolved: list[tuple[int, int, str]] = []
    for match in _COORDINATED_PICTURE_LIST_RE.finditer(text):
        labels = {
            ReferenceLabel.parse(item.group()).text
            for item in re.finditer(_PICTURE_TAG_PATTERN, match.group("labels"), re.IGNORECASE)
        }
        if reference.label.text not in labels:
            continue
        tail = match.group("tail")
        next_reference = _ANY_REFERENCE_TAG_RE.search(tail)
        exception = re.search(
            r"(?i),?\s*(?:but|except|however)\b|(?:ただし|しかし|一方で)",
            tail,
        )
        boundaries = [
            item.start()
            for item in (next_reference, exception)
            if item is not None
        ]
        if boundaries:
            tail = tail[: min(boundaries)]
        tail_role = _picture_role_from_context(tail)
        if tail_role != "neutral":
            resolved.append(
                (match.start("labels"), match.end("labels"), tail_role)
            )
            continue
        boundary = max(
            text.rfind("\n", 0, match.start()),
            text.rfind(";", 0, match.start()),
            text.rfind(".", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("?", 0, match.start()),
            text.rfind("。", 0, match.start()),
            text.rfind("！", 0, match.start()),
            text.rfind("？", 0, match.start()),
        )
        head = text[max(boundary + 1, match.start() - 180) : match.start()]
        head_role = _picture_role_from_context(head)
        if head_role != "neutral":
            resolved.append(
                (match.start("labels"), match.end("labels"), head_role)
            )
    return tuple(resolved)


def _picture_identity_is_explicitly_negated(
    reference: ReferenceSpec, text: str
) -> bool:
    separators = tuple(_ROLE_CLAUSE_SEPARATOR_RE.finditer(text))
    for match in _ANY_REFERENCE_TAG_RE.finditer(text):
        if ReferenceLabel.parse(match.group()).text != reference.label.text:
            continue
        clause_start = max(
            (item.end() for item in separators if item.end() <= match.start()),
            default=0,
        )
        clause_end = next(
            (item.start() for item in separators if item.start() >= match.end()),
            len(text),
        )
        if _NEGATIVE_IDENTITY_RE.search(text[clause_start:clause_end]):
            return True
    return False


def _picture_role(
    reference: ReferenceSpec,
    request: CompileRequest,
    *,
    dialogue_speaker_labels: frozenset[str] = frozenset(),
) -> str:
    if request.source_outfit_policy is SourceOutfitPolicy.PRESERVE:
        return "character"
    text = request.prompt
    coordinated_roles = _coordinated_picture_roles(reference, text)
    matches = tuple(_ANY_REFERENCE_TAG_RE.finditer(text))
    label_matches = tuple(
        match for match in matches if ReferenceLabel.parse(match.group()).text == reference.label.text
    )
    for match in reversed(label_matches):
        if any(
            start <= match.start() and match.end() <= end
            for start, end, _ in coordinated_roles
        ):
            continue
        separators = tuple(_ROLE_CLAUSE_SEPARATOR_RE.finditer(text))
        clause_start = max(
            (item.end() for item in separators if item.end() <= match.start()),
            default=0,
        )
        clause_end = next(
            (item.start() for item in separators if item.start() >= match.end()),
            len(text),
        )
        if _NEGATIVE_IDENTITY_RE.search(text[clause_start:clause_end]):
            continue
        clause_matches = tuple(
            item
            for item in matches
            if clause_start <= item.start() and item.end() <= clause_end
        )
        next_start = next(
            (item.start() for item in clause_matches if item.start() > match.start()),
            min(clause_end, match.end() + 240),
        )
        after = text[match.end() : min(next_start, match.end() + 240)]
        previous_end = max(
            (item.end() for item in clause_matches if item.end() < match.end()),
            default=max(clause_start, match.start() - 180),
        )
        before = text[max(previous_end, match.start() - 180) : match.start()]
        before_role = _picture_role_from_context(before)
        after_role = _picture_role_from_context(after)
        has_next_reference = any(
            item.start() > match.start() for item in clause_matches
        )
        after_binds_next = has_next_reference and bool(
            re.search(
                r"(?i)(?:from|in|using|based\s+on|according\s+to|から|より|は|を|に)\s*$",
                after,
            )
        )
        if after_binds_next:
            after_role = "neutral"
        before_binds = bool(
            re.search(
                r"(?i)(?:from|using|based\s+on|according\s+to|参照(?:元|先)?|"
                r"から|より|は|を|に|として)\s*$",
                before,
            )
        )
        after_binds = bool(
            re.match(
                r"(?i)^\s*(?:is\b|as\b|for\b|shows?\b|depicts?\b|"
                r"は|が|を|の|から|として)",
                after,
            )
        )
        if before_binds and before_role != "neutral":
            return before_role
        if after_binds and after_role != "neutral":
            return after_role
        if after_role != "neutral" and before_role == "neutral":
            return after_role
        if before_role != "neutral" and after_role == "neutral":
            return before_role
        if after_role != "neutral":
            return after_role
        if before_role != "neutral":
            return before_role
        if len(clause_matches) > 1 and not re.fullmatch(
            r"(?i)\s*(?:and|&|と|および|及び)?\s*",
            after,
        ):
            continue
        broad = text[
            max(clause_start, match.start() - 180) : min(clause_end, match.end() + 240)
        ]
        role = _picture_role_from_context(broad)
        if role != "neutral":
            return role
    if _picture_identity_is_explicitly_negated(reference, text):
        return "neutral"
    if coordinated_roles:
        return coordinated_roles[-1][2]
    if (
        reference.label.text in dialogue_speaker_labels
        and not _picture_identity_is_explicitly_negated(reference, text)
    ):
        return "character"
    return "neutral"


def _request_instruction_text(request: Mapping[str, Any]) -> str:
    return "\n".join(
        str(request.get(key, "") or "")
        for key in (
            "prompt",
            "dialogue",
            "soundscape",
            "audio_direction",
            "music_direction",
        )
    )


_AUDIO_LABEL_TOKEN_RE = re.compile(
    r"<\s*Audio\s+(?P<index>[1-9][0-9]*)\s*>",
    re.IGNORECASE,
)
_AUDIO_TAG_PATTERN = r"<\s*Audio\s+[1-9][0-9]*\s*>"
_AUDIO_LIST_CONNECTOR_PATTERN = (
    r"(?:\s*,\s*(?:(?:and|&)\s+)?|\s+(?:and|&)\s+|\s*(?:、|と|および|及び)\s*)"
)
_AUDIO_GROUP_PATTERN = (
    rf"(?P<labels>{_AUDIO_TAG_PATTERN}(?:{_AUDIO_LIST_CONNECTOR_PATTERN}"
    rf"{_AUDIO_TAG_PATTERN})+)"
)
_NEGATIVE_AUDIO_GROUP_PREFIX_RE = re.compile(
    rf"(?i)(?:do\s+not|don't|never)\s+(?:reuse|copy|preserve|keep)\s+"
    rf"(?:the\s+)?{_AUDIO_GROUP_PATTERN}"
)
_POSITIVE_AUDIO_GROUP_PREFIX_RE = re.compile(
    rf"(?i)(?:reuse|copy|preserve|keep)\s+(?:the\s+)?{_AUDIO_GROUP_PATTERN}"
)
_NEGATIVE_AUDIO_GROUP_SUFFIX_RE = re.compile(
    rf"{_AUDIO_GROUP_PATTERN}\s*(?:"
    rf"(?:must|should)\s+not\s+be\s+(?:reused|copied|preserved|kept)|"
    rf"(?:do\s+not|don't|never)\s+(?:reuse|copy|preserve|keep)|"
    rf"(?:を|は)?\s*(?:再利用しない|コピーしない|保持しない))",
    re.IGNORECASE,
)
_POSITIVE_AUDIO_GROUP_SUFFIX_RE = re.compile(
    rf"{_AUDIO_GROUP_PATTERN}\s*(?:"
    rf"(?:must|should)\s+be\s+(?:reused|copied|preserved|kept)|"
    rf"(?:are|is)\s+(?:reused|copied|preserved|kept)|"
    rf"(?:を|は)?\s*(?:そのまま)?(?:再利用|コピー|保持)(?:する|して)?)",
    re.IGNORECASE,
)


def _shift_standalone_audio_labels(value: str, offset: int) -> str:
    if offset <= 0 or not value:
        return value
    return _AUDIO_LABEL_TOKEN_RE.sub(
        lambda match: f"<Audio {int(match.group('index')) + offset}>",
        value,
    )


def _grouped_standalone_audio_candidates(
    label: ReferenceLabel, text: str
) -> tuple[tuple[int, int, AudioReferencePolicy], ...]:
    def includes(match: re.Match[str]) -> bool:
        return any(
            int(item.group("index")) == label.index
            for item in _AUDIO_LABEL_TOKEN_RE.finditer(match.group("labels"))
        )

    candidates: list[tuple[int, int, AudioReferencePolicy]] = []
    for pattern in (
        _NEGATIVE_AUDIO_GROUP_PREFIX_RE,
        _NEGATIVE_AUDIO_GROUP_SUFFIX_RE,
    ):
        candidates.extend(
            (match.start(), 1, AudioReferencePolicy.TIMBRE)
            for match in pattern.finditer(text)
            if includes(match)
        )
    for pattern in (
        _POSITIVE_AUDIO_GROUP_PREFIX_RE,
        _POSITIVE_AUDIO_GROUP_SUFFIX_RE,
    ):
        for match in pattern.finditer(text):
            # A positive prefix regex is also a substring of "do not reuse ...".
            # The negative group above handles the complete form; reject any
            # leftover negated prefix defensively instead of upgrading it.
            prefix = text[max(0, match.start() - 12) : match.start()]
            if re.search(r"(?i)(?:do\s+not|don't|never)\s*$", prefix):
                continue
            if includes(match):
                candidates.append((match.start(), 1, AudioReferencePolicy.REUSE))
    return tuple(candidates)


def _standalone_audio_policy(
    label: ReferenceLabel,
    request: Mapping[str, Any],
    *,
    default: AudioReferencePolicy,
) -> AudioReferencePolicy:
    text = _request_instruction_text(request)
    candidates = list(_grouped_standalone_audio_candidates(label, text))
    tag = rf"<\s*Audio\s+{label.index}\s*>"
    negative_reuse = re.compile(
        rf"(?i)(?:do\s+not|don't|never)\s+(?:reuse|copy|preserve|keep).{{0,40}}{tag}|"
        rf"{tag}.{{0,56}}(?:must\s+not|should\s+not|do\s+not|don't|never)\s+"
        rf"(?:be\s+)?(?:reused|copied|preserved|kept)|"
        rf"{tag}.{{0,56}}(?:再利用しない|コピーしない|そのまま使わない|保持しない)"
    )
    positive_reuse = re.compile(
        rf"(?i)(?:reuse|copy|preserve|keep)\s+(?:the\s+)?{tag}|"
        rf"{tag}.{{0,64}}(?:reuse|copy|preserve|keep|exactly\s+as\s+(?:the\s+)?"
        rf"(?:target\s+)?soundtrack|そのまま|再利用|コピー|音源として使う)"
    )
    explicit_timbre = re.compile(
        rf"(?i){tag}.{{0,56}}(?:voice\s+)?(?:timbre|delivery|voice\s+reference|"
        rf"声質|話し方|声の参照)"
    )
    separators = tuple(_ROLE_CLAUSE_SEPARATOR_RE.finditer(text))
    audio_tags = tuple(_AUDIO_LABEL_TOKEN_RE.finditer(text))
    matching_tags = tuple(
        item for item in audio_tags if int(item.group("index")) == label.index
    )
    for match in matching_tags:
        clause_start = max(
            (item.end() for item in separators if item.end() <= match.start()),
            default=0,
        )
        clause_end = next(
            (item.start() for item in separators if item.start() >= match.end()),
            len(text),
        )
        previous_audio_end = max(
            (item.end() for item in audio_tags if item.end() <= match.start()),
            default=clause_start,
        )
        next_audio_start = next(
            (item.start() for item in audio_tags if item.start() >= match.end()),
            clause_end,
        )
        segment_start = max(clause_start, previous_audio_end)
        segment_end = min(clause_end, next_audio_start)
        segment = text[segment_start:segment_end]
        negative_matches = tuple(negative_reuse.finditer(segment))
        negative_spans = tuple(item.span() for item in negative_matches)
        candidates.extend(
            (
                segment_start + item.start(),
                3,
                AudioReferencePolicy.TIMBRE,
            )
            for item in negative_matches
        )
        for item in positive_reuse.finditer(segment):
            if any(start <= item.start() < end for start, end in negative_spans):
                continue
            prefix = segment[max(0, item.start() - 12) : item.start()]
            if re.search(r"(?i)(?:do\s+not|don't|never)\s*$", prefix):
                continue
            candidates.append(
                (
                    segment_start + item.start(),
                    3,
                    AudioReferencePolicy.REUSE,
                )
            )
        candidates.extend(
            (
                segment_start + item.start(),
                2,
                AudioReferencePolicy.TIMBRE,
            )
            for item in explicit_timbre.finditer(segment)
        )
    if not candidates:
        return default
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _deduplicate_diagnostics(values: Sequence[Diagnostic]) -> tuple[Diagnostic, ...]:
    result: list[Diagnostic] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in values:
        key = (item.severity.value, item.code, item.message, item.path)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)


def _provenance(
    source: Any,
    *,
    output: str | None,
    embedded_policy: EmbeddedVideoAudioPolicy,
) -> Provenance:
    return Provenance(
        compiler_version=COMPILER_VERSION,
        source_sha256=stable_sha256(source),
        output_sha256=stable_sha256(output) if output is not None else None,
        embedded_video_audio_policy=embedded_policy.value,
    )


def _blocked(
    *,
    code: str,
    message: str,
    source: Any,
    embedded_policy: EmbeddedVideoAudioPolicy = EmbeddedVideoAudioPolicy.IGNORE,
) -> CompilationResult:
    diagnostic = Diagnostic(Severity.FATAL, code, message)
    return CompilationResult(
        ir_text=None,
        status=CompilationStatus.BLOCKED,
        document=None,
        diagnostics=(diagnostic,),
        auto_adjustments=(),
        provenance=_provenance(source, output=None, embedded_policy=embedded_policy),
    )


def _alignment_instruction(request: CompileRequest, shot_count: int) -> str | None:
    if request.mode is GenerationMode.I2V:
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    if request.mode is GenerationMode.FIRST_LAST:
        return (
            "How the reference pictures align with the target video — Picture 1 "
            "(from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {shot_count}) aligns with the "
            f"{request.duration_seconds:.2f}-second mark of the target video."
        )
    return None


def _verbatim_inline(label: str, value: str) -> str:
    return (
        f"{label} is preserved verbatim in the user's original language: "
        f"{value.strip()}"
    )


def _soundscape(request: CompileRequest, *, has_dialogue: bool) -> str:
    parts: list[str] = []
    if request.audio_direction.strip():
        parts.append(_verbatim_inline("User-authored audio priority", request.audio_direction))
    if request.soundscape.strip():
        parts.append(
            _verbatim_inline(
                "User-authored ambience, foley, and physical sound direction",
                request.soundscape,
            )
        )
    if has_dialogue:
        parts.append(
            "Keep only the explicitly tagged dialogue audible and synchronized with its visible "
            "speaker; do not add narration or unspecified words."
        )
    else:
        parts.append(
            "Do not add narration, voice-over, dialogue, lyrics, or unintended human voices."
        )
    if not request.audio_direction.strip() and not request.soundscape.strip():
        parts.append(
            "Use restrained diegetic ambience and physical sounds only where implied by the "
            "unchanged user instruction."
        )
    return " ".join(parts)


def _music(request: CompileRequest) -> tuple[str, AutoAdjustment | None]:
    if request.music_policy is MusicPolicy.NONE:
        return "N/A", None
    if request.music_direction.strip():
        return _verbatim_inline("User-authored non-diegetic music direction", request.music_direction), None
    if request.music_policy is MusicPolicy.SUBTLE:
        return (
            "Sparse instrumental music at a restrained volume and steady slow-to-moderate tempo, "
            "kept beneath dialogue and important diegetic sound.",
            None,
        )
    if request.music_policy is MusicPolicy.PROMINENT:
        return (
            "A prominent instrumental score with clearly defined rhythm and dynamic development, "
            "without unrequested singing or lyrics.",
            None,
        )
    return (
        "N/A",
        AutoAdjustment(
            "MUSIC_AUTO_DEFAULTED_NA",
            "No separate music direction was supplied, so non-diegetic music was set to N/A.",
        ),
    )


def _audio_policy(reference: ReferenceSpec, request: CompileRequest) -> AudioReferencePolicy:
    return reference.audio_policy or request.audio_reference_policy


def _dialogue_speaker_picture_labels(
    dialogue_events: Sequence[DialogueEvent],
) -> frozenset[str]:
    return frozenset(
        event.speaker_reference.text
        for event in dialogue_events
        if event.speaker_reference is not None
        and event.speaker_reference.kind is ReferenceKind.PICTURE
    )


def _dialogue_speaker_reference_labels(
    dialogue_events: Sequence[DialogueEvent],
) -> frozenset[str]:
    return frozenset(
        event.speaker_reference.text
        for event in dialogue_events
        if event.speaker_reference is not None
    )


def _dialogue_speaker_role_conflict_diagnostics(
    document: ContextIRDocument,
    request: CompileRequest,
    resolved_picture_roles: Sequence[tuple[str, str]],
) -> tuple[Diagnostic, ...]:
    """Warn when an explicit Picture speaker binding contradicts its declared role."""

    role_by_label = dict(resolved_picture_roles)
    picture_by_label = {
        reference.label.text: reference
        for reference in request.references
        if reference.label.kind is ReferenceKind.PICTURE
    }
    conflicting_roles = frozenset({"clothing", "style", "background", "object"})
    emitted_labels: set[str] = set()
    issues: list[Diagnostic] = []

    for shot in document.shots:
        for event_index, event in enumerate(shot.dialogue_events):
            reference_label = event.speaker_reference
            if (
                reference_label is None
                or reference_label.kind is not ReferenceKind.PICTURE
                or reference_label.text in emitted_labels
            ):
                continue
            role = role_by_label.get(reference_label.text)
            reference = picture_by_label.get(reference_label.text)
            negative_identity = (
                reference is not None
                and _picture_identity_is_explicitly_negated(reference, request.prompt)
            )
            if role not in conflicting_roles and not negative_identity:
                continue

            emitted_labels.add(reference_label.text)
            conflict = (
                "explicitly excluded from character identity use"
                if negative_identity
                else f"resolved as a {role} reference"
            )
            issues.append(
                Diagnostic(
                    Severity.WARNING,
                    "IR_DIALOGUE_SPEAKER_ROLE_CONFLICT",
                    f"Dialogue directly binds {reference_label.text} as the visible speaker, "
                    f"but that picture is {conflict}. The compiler retained the direct "
                    "speaker binding without remapping or inventing a different subject.",
                    f"/shots/{shot.index - 1}/dialogue_events/"
                    f"{event_index}/speaker_reference",
                )
            )
    return tuple(issues)


def _dialogue_audio_targets(
    dialogue_events: Sequence[DialogueEvent],
) -> dict[str, tuple[str, ...]]:
    collected: dict[str, list[str]] = {}
    for event in dialogue_events:
        if event.audio_reference is None:
            continue
        if event.speaker_reference is not None:
            target = (
                "the visible speaking character identified by "
                f"{event.speaker_reference.text}"
            )
        else:
            target = f"<Subject {event.speaker_id}> (S{event.speaker_id})"
        values = collected.setdefault(event.audio_reference.text, [])
        if target not in values:
            values.append(target)
    return {key: tuple(values) for key, values in collected.items()}


def _definitions_and_retention(
    request: CompileRequest,
    dialogue_events: Sequence[DialogueEvent] = (),
) -> tuple[tuple[SubjectDefinition, ...], tuple[RetentionEntry, ...]]:
    pictures = tuple(
        item for item in request.references if item.label.kind is ReferenceKind.PICTURE
    )
    videos = tuple(
        item for item in request.references if item.label.kind is ReferenceKind.VIDEO
    )
    audios = tuple(
        item for item in request.references if item.label.kind is ReferenceKind.AUDIO
    )
    definitions: list[SubjectDefinition] = []
    retention: list[RetentionEntry] = []

    dialogue_speaker_labels = _dialogue_speaker_picture_labels(dialogue_events)
    dialogue_speaker_reference_labels = _dialogue_speaker_reference_labels(
        dialogue_events
    )
    dialogue_audio_targets = _dialogue_audio_targets(dialogue_events)
    picture_roles = tuple(
        (
            reference,
            _picture_role(
                reference,
                request,
                dialogue_speaker_labels=dialogue_speaker_labels,
            ),
        )
        for reference in pictures
    )
    subject_index = 0
    for reference, role in picture_roles:
        if role == "character":
            subject_index += 1
            if request.source_outfit_policy is SourceOutfitPolicy.PRESERVE:
                outfit_definition = "Preserve the source clothing as part of the character."
                relationship = "fully_preserved"
                outfit_retention = "Preserve the source outfit."
            elif request.source_outfit_policy is SourceOutfitPolicy.AS_PROMPTED:
                outfit_definition = (
                    "Handle source clothing only as directed by the unchanged user instruction."
                )
                relationship = "partially_preserved"
                outfit_retention = (
                    "Apply the user's wardrobe direction without adding an assumption."
                )
            else:
                outfit_definition = (
                    "The source outfit and source background are not target requirements when "
                    "the user specifies replacement clothing or a different setting."
                )
                relationship = "partially_preserved"
                outfit_retention = (
                    "Preserve identity, face, hair, eye traits, proportions, and body shape. "
                    "When target clothing is specified, replace the source outfit completely; "
                    "the source outfit must not override the target wardrobe."
                )
            text = (
                f"<Subject {subject_index}> is the character identity and body-shape reference "
                f"derived from {reference.label.text}. Preserve identity-defining facial "
                f"features, hairstyle, eye traits, proportions, and body shape. "
                f"{outfit_definition}"
            )
            definitions.append(
                SubjectDefinition(
                    text=text,
                    subject_index=subject_index,
                    source_label=reference.label,
                )
            )
            retention.append(
                RetentionEntry(
                    label=f"<Subject {subject_index}> (from {reference.label.text})",
                    relationship=relationship,
                    detail=outfit_retention,
                )
            )
        elif role == "clothing":
            definitions.append(
                SubjectDefinition(
                    text=(
                        f"{reference.label.text} is a clothing/costume design reference only. "
                        "Use garment silhouette, construction, materials, colors, trims, and "
                        "accessories only as explicitly requested. Do not copy or infer the "
                        "pictured person's identity, face, or body shape."
                    ),
                    source_label=reference.label,
                )
            )
            retention.append(
                RetentionEntry(
                    label=reference.label.text,
                    relationship="reference",
                    detail=(
                        "Retain only the requested clothing design; the pictured wearer and "
                        "background are not target identity requirements."
                    ),
                )
            )
        elif role == "style":
            definitions.append(
                SubjectDefinition(
                    text=(
                        f"{reference.label.text} is a visual-style reference. Apply only the "
                        "explicitly requested medium, rendering language, palette, texture, or "
                        "lighting; do not copy depicted identities or scene content."
                    ),
                    source_label=reference.label,
                )
            )
            retention.append(
                RetentionEntry(
                    label=reference.label.text,
                    relationship="weak_reference",
                    detail="Use style traits only where the unchanged user instruction requests them.",
                )
            )
        elif role == "background":
            definitions.append(
                SubjectDefinition(
                    text=(
                        f"{reference.label.text} is a background/location reference. Use only "
                        "the requested setting, spatial layout, ambience, or environmental "
                        "details; do not adopt unrelated people or foreground objects."
                    ),
                    source_label=reference.label,
                )
            )
            retention.append(
                RetentionEntry(
                    label=reference.label.text,
                    relationship="reference",
                    detail="Retain only setting elements explicitly assigned to this reference.",
                )
            )
        elif role == "object":
            definitions.append(
                SubjectDefinition(
                    text=(
                        f"{reference.label.text} is an object/product design reference. Preserve "
                        "only explicitly requested shape, construction, materials, markings, and "
                        "functional details; do not infer a pictured person's identity."
                    ),
                    source_label=reference.label,
                )
            )
            retention.append(
                RetentionEntry(
                    label=reference.label.text,
                    relationship="reference",
                    detail="Retain only the explicitly requested object or product attributes.",
                )
            )
        else:
            definitions.append(
                SubjectDefinition(
                    text=(
                        f"{reference.label.text} is a neutral visual reference. Use only visual "
                        "elements explicitly assigned to it by the unchanged user instruction; "
                        "do not assume character identity, body shape, wardrobe, background, or "
                        "style from the image."
                    ),
                    source_label=reference.label,
                )
            )
            retention.append(
                RetentionEntry(
                    label=reference.label.text,
                    relationship="weak_reference",
                    detail="Do not retain unspecified visual attributes from this reference.",
                )
            )

    video_speaker_labels = frozenset(
        label
        for label in dialogue_speaker_reference_labels
        if label.lower().startswith("<video ")
    )
    if subject_index == 0 and not video_speaker_labels:
        definitions.append(
            SubjectDefinition(
                text=(
                    "<Subject 1> is the primary visible subject described in the unchanged "
                    "user instruction."
                ),
                subject_index=1,
            )
        )

    for reference in videos:
        speaker_video = reference.label.text in video_speaker_labels
        definitions.append(
            SubjectDefinition(
                text=(
                    (
                        f"{reference.label.text} is the explicit visual identity and appearance "
                        "reference for a visible speaking character. Preserve that speaker's "
                        "identity-defining appearance while using only the motion, timing, "
                        "camera, or composition requested by the user."
                    )
                    if speaker_video
                    else (
                        f"{reference.label.text} is a visual motion, temporal-structure, and "
                        "composition reference; it is not a source edit unless the unchanged "
                        "user instruction explicitly says so."
                    )
                ),
                source_label=reference.label,
            )
        )
        retention.append(
            RetentionEntry(
                label=reference.label.text,
                relationship="partially_preserved" if speaker_video else "weak_reference",
                detail=(
                    (
                        "Preserve the explicitly selected visible speaker's identity-defining "
                        "appearance and only the requested motion/timing/camera traits."
                    )
                    if speaker_video
                    else (
                        "Use only motion, timing, camera, or composition explicitly requested by "
                        "the user; do not silently copy unrelated source content."
                    )
                ),
            )
        )

    for reference in audios:
        policy = _audio_policy(reference, request)
        if reference.origin is ReferenceOrigin.EMBEDDED_VIDEO_AUDIO:
            if policy is AudioReferencePolicy.REUSE:
                definition = (
                    f"{reference.label.text} is the synchronized soundtrack from its source "
                    "reference video and is explicitly reused in the target video."
                )
            else:
                definition = (
                    f"{reference.label.text} is the synchronized soundtrack from its source "
                    "reference video and supplies audio characteristics without copying its "
                    "original signal."
                )
        elif policy is AudioReferencePolicy.TIMBRE:
            targets = dialogue_audio_targets.get(reference.label.text, ())
            if targets:
                target_text = ", ".join(targets)
                definition = (
                    f"{reference.label.text} is a voice-timbre and delivery reference for "
                    f"{target_text}; it is not source dialogue to copy or continue."
                )
            else:
                definition = (
                    f"{reference.label.text} is a voice-timbre and delivery reference for only "
                    "the visible speaker explicitly assigned by the unchanged user instruction; "
                    "it is not source dialogue to copy or continue."
                )
        elif policy is AudioReferencePolicy.REUSE:
            definition = (
                f"{reference.label.text} is an audio source explicitly selected for partial "
                "signal reuse according to the unchanged user instruction."
            )
        else:
            definition = (
                f"{reference.label.text} is an audio reference whose role follows only the "
                "unchanged user instruction; it is not copied unless explicitly requested."
            )
        definitions.append(
            SubjectDefinition(text=definition, source_label=reference.label)
        )

        if policy is AudioReferencePolicy.REUSE:
            retention.append(
                RetentionEntry(
                    label=reference.label.text,
                    relationship="partially_copy",
                    detail=(
                        "Reuse only the explicitly requested audio portion or layer, without "
                        "inventing an additional voice source."
                    ),
                )
            )
        elif policy is AudioReferencePolicy.TIMBRE:
            retention.append(
                RetentionEntry(
                    label=reference.label.text,
                    relationship="reference",
                    detail=(
                        "Use only speaker timbre and delivery characteristics. Do not copy, "
                        "continue, paraphrase, or reuse words, timing, background audio, or "
                        "music from the source recording."
                    ),
                )
            )
        else:
            retention.append(
                RetentionEntry(
                    label=reference.label.text,
                    relationship="reference",
                    detail=(
                        "Use only the audio characteristics explicitly requested by the user; "
                        "do not copy the signal by default."
                    ),
                )
            )
    return tuple(definitions), tuple(retention)


def _summary(request: CompileRequest) -> tuple[tuple[str, ...], str]:
    references = ", ".join(item.label.text for item in request.references)
    task_types = ["reference generation"]
    audio_refs = tuple(
        item for item in request.references if item.label.kind is ReferenceKind.AUDIO
    )
    if audio_refs:
        if any(_audio_policy(item, request) is AudioReferencePolicy.REUSE for item in audio_refs):
            task_types.append("audio reuse")
        else:
            task_types.append("audio reference")
    source = references or "the supplied reference context"
    return (
        tuple(task_types),
        (
            f"The target video follows the unchanged user instruction using {source}. "
            "Identity, wardrobe, video structure, and audio are retained only according to "
            "the explicit policies below, and no extra speech is introduced."
        ),
    )


def compile_context_ir(request: CompileRequest) -> CompilationResult:
    """Compile one validated request without network access or model inference."""

    if not isinstance(request, CompileRequest):
        raise TypeError("request must be a CompileRequest")
    source_payload = {
        "mode": request.mode.value,
        "prompt": request.prompt,
        "duration_ms": request.duration_ms,
        "style_direction": request.style_direction,
        "dialogue": request.dialogue,
        "soundscape": request.soundscape,
        "audio_direction": request.audio_direction,
        "music_policy": request.music_policy.value,
        "music_direction": request.music_direction,
        "references": [item.to_dict() for item in request.references],
        "dialogue_language": request.dialogue_language,
        "source_outfit_policy": request.source_outfit_policy.value,
        "audio_reference_policy": request.audio_reference_policy.value,
        "embedded_video_audio_policy": request.embedded_video_audio_policy.value,
    }

    raw_ir = is_h3_context_ir(request.prompt)
    reserved = _reserved_control_token(request, raw_ir=raw_ir)
    if reserved is not None:
        field_name, token = reserved
        return _blocked(
            code="IR_RESERVED_CONTROL_TOKEN",
            message=(
                f"{field_name} contains reserved H3 control token {token!r}; "
                "control tokens are emitted only by the compiler."
            ),
            source=source_payload,
            embedded_policy=request.embedded_video_audio_policy,
        )

    if raw_ir and request.dialogue.strip():
        return _blocked(
            code="IR_RAW_WITH_SEPARATE_DIALOGUE",
            message=(
                "A precompiled Context-IR prompt cannot also use the separate dialogue field; "
                "place the dialogue inside the raw IR or clear the dialogue field."
            ),
            source=source_payload,
            embedded_policy=request.embedded_video_audio_policy,
        )

    if raw_ir:
        diagnostics = validate_rendered_ir(
            request.prompt,
            mode=request.mode,
            duration_ms=request.duration_ms,
            allowed_reference_labels=tuple(
                item.label.text for item in request.references
            ),
        )
        if has_fatal(diagnostics):
            return CompilationResult(
                ir_text=None,
                status=CompilationStatus.BLOCKED,
                document=None,
                diagnostics=diagnostics,
                auto_adjustments=(),
                provenance=_provenance(
                    source_payload,
                    output=None,
                    embedded_policy=request.embedded_video_audio_policy,
                ),
            )
        return CompilationResult(
            ir_text=request.prompt,
            status=CompilationStatus.RAW_VALIDATED,
            document=None,
            diagnostics=diagnostics,
            auto_adjustments=(),
            provenance=_provenance(
                source_payload,
                output=request.prompt,
                embedded_policy=request.embedded_video_audio_policy,
            ),
        )

    parsed_prompt = split_prompt_into_shots(request.prompt, request.duration_ms)
    parsed_dialogue = parse_dialogue(
        request.dialogue, default_language=request.dialogue_language
    )
    shots, target_adjustments = assign_dialogue_to_shots(
        parsed_prompt.shots, parsed_dialogue.events
    )
    definitions, retention = _definitions_and_retention(
        request,
        parsed_dialogue.events,
    )
    summary_prefix, summary_text = _summary(request)
    music, music_adjustment = _music(request)
    document = ContextIRDocument(
        mode=request.mode,
        duration_ms=request.duration_ms,
        alignment_instruction=_alignment_instruction(request, len(shots)),
        preamble=parsed_prompt.preamble,
        shots=shots,
        references=request.references,
        subject_definitions=definitions,
        retention_entries=retention,
        summary_prefix=summary_prefix,
        summary_text=summary_text,
        style_direction=request.style_direction,
        overall_soundscape=_soundscape(request, has_dialogue=bool(parsed_dialogue.events)),
        non_diegetic_music=music,
        source_outfit_policy=request.source_outfit_policy,
        audio_reference_policy=request.audio_reference_policy,
        embedded_video_audio_policy=request.embedded_video_audio_policy,
        degraded=True,
    )

    adjustments: list[AutoAdjustment] = [
        AutoAdjustment(
            "OFFICIAL_SECTION_SCAFFOLD",
            "The request was wrapped in the official H3 three- or six-section structure.",
        )
    ]
    if parsed_prompt.inferred_timestamps and len(shots) > 1:
        adjustments.append(
            AutoAdjustment(
                "SHOT_TIMES_INFERRED",
                "Cut/Shot markers were assigned evenly spaced cut times within the effective duration.",
            )
        )
    if parsed_dialogue.events:
        adjustments.append(
            AutoAdjustment(
                "DIALOGUE_IMMUTABLY_TAGGED",
                "Each dialogue utterance was preserved and rendered exactly once in an H3 <d> block.",
            )
        )
    for old_target, new_target in target_adjustments:
        adjustments.append(
            AutoAdjustment(
                "DIALOGUE_TARGET_RESOLVED",
                f"Dialogue target {old_target!r} was resolved to Shot {new_target}.",
            )
        )
    dialogue_speaker_labels = _dialogue_speaker_picture_labels(parsed_dialogue.events)
    resolved_picture_roles = tuple(
        (
            item.label.text,
            _picture_role(
                item,
                request,
                dialogue_speaker_labels=dialogue_speaker_labels,
            ),
        )
        for item in request.references
        if item.label.kind is ReferenceKind.PICTURE
    )
    for label, role in resolved_picture_roles:
        adjustments.append(
            AutoAdjustment(
                "PICTURE_ROLE_RESOLVED",
                f"{label} was deterministically classified as {role} from its explicit "
                "prompt/dialogue context.",
            )
        )
    if (
        request.mode is GenerationMode.OMNI
        and request.source_outfit_policy is SourceOutfitPolicy.REPLACE_IF_SPECIFIED
        and any(role == "character" for _, role in resolved_picture_roles)
    ):
        adjustments.append(
            AutoAdjustment(
                "SOURCE_OUTFIT_OVERRIDE_POLICY",
                "Identity/body traits are preserved, while explicitly requested target clothing overrides source clothing.",
            )
        )
    if any(
        item.label.kind is ReferenceKind.AUDIO
        and _audio_policy(item, request) is AudioReferencePolicy.TIMBRE
        for item in request.references
    ):
        adjustments.append(
            AutoAdjustment(
                "AUDIO_TIMBRE_ONLY_POLICY",
                "Reference audio defaults to timbre/delivery only; source words and background audio are not reused.",
            )
        )
    if request.embedded_video_audio_policy is EmbeddedVideoAudioPolicy.IGNORE and any(
        item.label.kind is ReferenceKind.VIDEO for item in request.references
    ):
        adjustments.append(
            AutoAdjustment(
                "EMBEDDED_VIDEO_AUDIO_IGNORED",
                "Reference-video soundtracks are ignored because their use was not explicitly requested.",
            )
        )
    if music_adjustment is not None:
        adjustments.append(music_adjustment)

    diagnostics: list[Diagnostic] = list(validate_document(document))
    diagnostics.extend(
        _dialogue_speaker_role_conflict_diagnostics(
            document,
            request,
            resolved_picture_roles,
        )
    )
    diagnostics.append(
        Diagnostic(
            Severity.NOTE,
            "IR_LOCAL_RULE_COMPILER",
            "Compilation was fully local and deterministic; no AI model inference or network request was used.",
        )
    )
    ir_text = render_context_ir(document)
    diagnostics.extend(
        validate_rendered_ir(
            ir_text,
            mode=request.mode,
            duration_ms=request.duration_ms,
            document=document,
        )
    )
    final_diagnostics = _deduplicate_diagnostics(diagnostics)
    if has_fatal(final_diagnostics):
        return CompilationResult(
            ir_text=None,
            status=CompilationStatus.BLOCKED,
            document=document,
            diagnostics=final_diagnostics,
            auto_adjustments=tuple(adjustments),
            provenance=_provenance(
                source_payload,
                output=None,
                embedded_policy=request.embedded_video_audio_policy,
            ),
        )
    return CompilationResult(
        ir_text=ir_text,
        status=CompilationStatus.DEGRADED,
        document=document,
        diagnostics=final_diagnostics,
        auto_adjustments=tuple(adjustments),
        provenance=_provenance(
            source_payload,
            output=ir_text,
            embedded_policy=request.embedded_video_audio_policy,
        ),
    )


def _enum_value(enum_type: type[Any], value: Any, *, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, enum_type):
        return value
    return enum_type(str(value).strip().lower())


def _duration_ms(request: Mapping[str, Any]) -> int:
    if request.get("duration_ms") is not None:
        value = request["duration_ms"]
        if isinstance(value, bool):
            raise ValueError("duration_ms must be numeric")
        return int(value)
    if request.get("duration_seconds") is not None:
        value = Decimal(str(request["duration_seconds"])) * Decimal(1000)
        return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    frames = request.get("num_frames", request.get("frames"))
    if frames is not None:
        value = Decimal(int(frames)) * Decimal(1000) / Decimal(24)
        return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    raise ValueError("request needs duration_ms, duration_seconds, num_frames, or frames")


def _reference_counts(request: Mapping[str, Any]) -> tuple[int, int, int]:
    references = request.get("references") or ()
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise ValueError("references must be a sequence")
    counts = {"image": 0, "video": 0, "audio": 0}
    for item in references:
        if not isinstance(item, Mapping):
            raise ValueError("each reference must be an object")
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in counts:
            raise ValueError(f"unsupported reference kind {kind!r}")
        counts[kind] += 1
    return counts["image"], counts["video"], counts["audio"]


def _reference_video_audio_availability(
    request: Mapping[str, Any],
) -> tuple[bool | None, ...]:
    values: list[bool | None] = []
    for item in request.get("references") or ():
        if not isinstance(item, Mapping):
            continue
        if str(item.get("kind", "")).strip().lower() != "video":
            continue
        has_audio = item.get("has_audio")
        values.append(has_audio if isinstance(has_audio, bool) else None)
    return tuple(values)


def _raw_ir_ignored_auxiliary_fields(request: Mapping[str, Any]) -> tuple[str, ...]:
    """Return separately authored controls that raw Context-IR cannot merge safely."""

    ignored: list[str] = []
    for key in ("soundscape", "audio_direction", "music_direction", "style_direction"):
        value = request.get(key)
        if isinstance(value, str) and value.strip():
            ignored.append(key)
    style = str(request.get("style", "natural") or "natural").strip().lower()
    if style not in {"", "natural"}:
        ignored.append("style")
    audio_preset = str(request.get("audio_preset", "auto") or "auto").strip().lower()
    if audio_preset not in {"", "auto"}:
        ignored.append("audio_preset")
    music_policy = str(request.get("music_policy", "auto") or "auto").strip().lower()
    if music_policy not in {"", "auto"}:
        ignored.append("music_policy")
    source_outfit_policy = str(
        request.get("source_outfit_policy", "replace_if_specified")
        or "replace_if_specified"
    ).strip().lower()
    if source_outfit_policy not in {"", "replace_if_specified"}:
        ignored.append("source_outfit_policy")
    audio_reference_policy = str(
        request.get("audio_reference_policy", "timbre") or "timbre"
    ).strip().lower()
    if audio_reference_policy not in {"", "timbre"}:
        ignored.append("audio_reference_policy")
    return tuple(ignored)


_VIDEO_AUDIO_SOURCE_RE = re.compile(_VIDEO_AUDIO_SOURCE_PATTERN, re.IGNORECASE)
_NUMBERED_VIDEO_AUDIO_SOURCE_PATTERN = (
    r"(?:<\s*Video\s+[1-9][0-9]*\s*>|"
    r"\bVideo\s+[1-9][0-9]*\b)\s*"
    r"(?:'s\s*)?(?:audio|soundtrack)"
)
_VIDEO_AUDIO_LIST_CONNECTOR_PATTERN = (
    r"(?:\s*,\s*(?:(?:and|&)\s+)?|\s+(?:and|&)\s+)"
)
_VIDEO_AUDIO_GROUP_PATTERN = (
    rf"(?P<sources>{_NUMBERED_VIDEO_AUDIO_SOURCE_PATTERN}(?:"
    rf"{_VIDEO_AUDIO_LIST_CONNECTOR_PATTERN}{_NUMBERED_VIDEO_AUDIO_SOURCE_PATTERN})+)"
)
_VIDEO_AUDIO_GROUP_REUSE_PREFIX_RE = re.compile(
    rf"(?i)(?:reuse|copy|preserve|keep)\s+(?:the\s+)?{_VIDEO_AUDIO_GROUP_PATTERN}"
)
_VIDEO_AUDIO_GROUP_REUSE_SUFFIX_RE = re.compile(
    rf"(?i){_VIDEO_AUDIO_GROUP_PATTERN}\s*(?:must|should)\s+be\s+"
    rf"(?:reused|copied|preserved|kept)"
)
_VIDEO_AUDIO_GROUP_IGNORE_PREFIX_RE = re.compile(
    rf"(?i)(?:do\s+not|don't|never)\s+(?:reuse|copy|preserve|keep)\s+"
    rf"(?:the\s+)?{_VIDEO_AUDIO_GROUP_PATTERN}"
)
_VIDEO_AUDIO_LOCAL_CONNECTOR_RE = re.compile(
    r"(?:,\s*|\b(?:and|while|whereas|but|then)\b)",
    re.IGNORECASE,
)


def _video_audio_policy_evidence(
    clause: str,
) -> tuple[EmbeddedVideoAudioPolicy, bool]:
    if _VIDEO_AUDIO_HARD_IGNORE_RE.search(clause):
        return EmbeddedVideoAudioPolicy.IGNORE, True
    if _VIDEO_AUDIO_NEGATED_REUSE_RE.search(clause):
        if _VIDEO_AUDIO_REFERENCE_RE.search(clause):
            return EmbeddedVideoAudioPolicy.REFERENCE, True
        return EmbeddedVideoAudioPolicy.IGNORE, True
    if _VIDEO_AUDIO_REUSE_RE.search(clause) or _VIDEO_AUDIO_REUSE_PREFIX_RE.search(clause):
        return EmbeddedVideoAudioPolicy.REUSE, True
    if _VIDEO_AUDIO_REFERENCE_RE.search(clause):
        return EmbeddedVideoAudioPolicy.REFERENCE, True
    return EmbeddedVideoAudioPolicy.IGNORE, False


def _video_audio_policy_from_clause(clause: str) -> EmbeddedVideoAudioPolicy:
    return _video_audio_policy_evidence(clause)[0]


def _clause_mentions_video_audio(clause: str, index: int) -> bool:
    return bool(
        re.search(
            rf"(?i)<\s*Video\s+{index}\s*>\s*(?:'s\s*|の)?(?:audio|soundtrack|音声)|"
            rf"\bVideo\s+{index}\b\s*(?:'s\s*)?(?:audio|soundtrack)|"
            rf"(?:audio|soundtrack)\s+(?:from|of)\s+<\s*Video\s+{index}\s*>|"
            rf"動画\s*{index}(?![0-9])\s*(?:の|に含まれる|内の)?音声",
            clause,
        )
    )


def _grouped_video_audio_policy_events(
    text: str,
) -> tuple[tuple[int, EmbeddedVideoAudioPolicy, tuple[int, ...]], ...]:
    def indices(match: re.Match[str]) -> tuple[int, ...]:
        values: list[int] = []
        for item in re.finditer(
            _NUMBERED_VIDEO_AUDIO_SOURCE_PATTERN,
            match.group("sources"),
            re.IGNORECASE,
        ):
            index_match = re.search(r"(?i)Video\s+([1-9][0-9]*)", item.group())
            assert index_match is not None
            values.append(int(index_match.group(1)))
        return tuple(values)

    events: list[tuple[int, EmbeddedVideoAudioPolicy, tuple[int, ...]]] = []
    events.extend(
        (match.start(), EmbeddedVideoAudioPolicy.IGNORE, indices(match))
        for match in _VIDEO_AUDIO_GROUP_IGNORE_PREFIX_RE.finditer(text)
    )
    for pattern in (
        _VIDEO_AUDIO_GROUP_REUSE_PREFIX_RE,
        _VIDEO_AUDIO_GROUP_REUSE_SUFFIX_RE,
    ):
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 12) : match.start()]
            if re.search(r"(?i)(?:do\s+not|don't|never)\s*$", prefix):
                continue
            events.append(
                (match.start(), EmbeddedVideoAudioPolicy.REUSE, indices(match))
            )
    return tuple(sorted(events, key=lambda item: item[0]))


def _localized_video_audio_policies(
    clause: str, *, video_count: int
) -> tuple[tuple[int, EmbeddedVideoAudioPolicy], ...] | None:
    """Bind policies to each explicit video-audio source inside one clause.

    A conjunction may carry independent instructions, for example reference
    ``<Video 1>`` audio but reuse ``<Video 2>`` audio.  Clause-wide matching
    would let the stronger reuse/ignore phrase leak across both tags.  Split
    only between adjacent explicit source mentions, using the last connector
    in their gap so phrases such as "voice and timbre reference" stay local.
    """

    mentions: list[tuple[int, int, int]] = []
    for match in _VIDEO_AUDIO_SOURCE_RE.finditer(clause):
        source = match.group(0)
        for index in range(1, video_count + 1):
            if _clause_mentions_video_audio(source, index):
                mentions.append((match.start(), match.end(), index))
                break
    if len(mentions) < 2:
        return None

    separators: list[tuple[int, int]] = []
    for left, right in zip(mentions, mentions[1:], strict=False):
        gap_start = left[1]
        gap = clause[gap_start : right[0]]
        connectors = tuple(_VIDEO_AUDIO_LOCAL_CONNECTOR_RE.finditer(gap))
        if not connectors:
            return None
        connector = connectors[-1]
        separators.append(
            (gap_start + connector.start(), gap_start + connector.end())
        )

    localized: list[tuple[int, EmbeddedVideoAudioPolicy, bool]] = []
    for position, (_, _, index) in enumerate(mentions):
        start = 0 if position == 0 else separators[position - 1][1]
        end = len(clause) if position == len(mentions) - 1 else separators[position][0]
        policy, explicit = _video_audio_policy_evidence(clause[start:end])
        localized.append((index, policy, explicit))

    explicit_policies = {
        policy for _, policy, explicit in localized if explicit
    }
    if not explicit_policies:
        return ()
    if len(explicit_policies) == 1:
        shared_policy = next(iter(explicit_policies))
        localized = [
            (index, policy if explicit else shared_policy, explicit)
            for index, policy, explicit in localized
        ]
    return tuple((index, policy) for index, policy, _ in localized)


def _resolve_embedded_selection(
    request: Mapping[str, Any], *, video_count: int
) -> tuple[EmbeddedVideoAudioPolicy, tuple[EmbeddedVideoAudioPolicy, ...]]:
    requested = _enum_value(
        EmbeddedVideoAudioPolicy,
        request.get("embedded_video_audio_policy", "auto"),
        default=EmbeddedVideoAudioPolicy.AUTO,
    )
    if video_count == 0:
        return EmbeddedVideoAudioPolicy.IGNORE, ()
    if requested is not EmbeddedVideoAudioPolicy.AUTO:
        return requested, tuple(requested for _ in range(video_count))
    text = "\n".join(
        str(request.get(key, ""))
        for key in ("prompt", "dialogue", "soundscape", "music_direction")
    )
    per_video: list[EmbeddedVideoAudioPolicy | None] = [None] * video_count
    generic_policy = EmbeddedVideoAudioPolicy.IGNORE
    for _, policy, indices in _grouped_video_audio_policy_events(text):
        for index in indices:
            if 1 <= index <= video_count:
                per_video[index - 1] = policy
    for clause in _ROLE_CLAUSE_SEPARATOR_RE.split(text):
        if not _VIDEO_AUDIO_SOURCE_RE.search(clause):
            continue
        localized = _localized_video_audio_policies(
            clause,
            video_count=video_count,
        )
        if localized is not None:
            for index, policy in localized:
                per_video[index - 1] = policy
            continue
        policy, explicit = _video_audio_policy_evidence(clause)
        mentioned = [
            index
            for index in range(1, video_count + 1)
            if _clause_mentions_video_audio(clause, index)
        ]
        if mentioned:
            if explicit:
                for index in mentioned:
                    per_video[index - 1] = policy
        else:
            generic_policy = policy
    resolved = tuple(item or generic_policy for item in per_video)
    if any(item is EmbeddedVideoAudioPolicy.REUSE for item in resolved):
        aggregate = EmbeddedVideoAudioPolicy.REUSE
    elif any(item is EmbeddedVideoAudioPolicy.REFERENCE for item in resolved):
        aggregate = EmbeddedVideoAudioPolicy.REFERENCE
    else:
        aggregate = EmbeddedVideoAudioPolicy.IGNORE
    return aggregate, resolved


def _references_from_mapping(
    request: Mapping[str, Any],
    *,
    embedded_policies: tuple[EmbeddedVideoAudioPolicy, ...],
) -> tuple[ReferenceSpec, ...]:
    explicit = request.get("reference_labels")
    if explicit:
        if isinstance(explicit, (str, bytes)) or not isinstance(explicit, Sequence):
            raise ValueError("reference_labels must be a sequence")
        return tuple(
            ReferenceSpec(
                ReferenceLabel.parse(str(value)),
                ReferenceOrigin.EXPLICIT,
                source_index=index,
            )
            for index, value in enumerate(explicit)
        )

    references = request.get("references") or ()
    pictures: list[ReferenceSpec] = []
    videos: list[ReferenceSpec] = []
    standalone_audio_source_indices: list[int] = []
    for source_index, item in enumerate(references):
        assert isinstance(item, Mapping)
        kind = str(item.get("kind", "")).strip().lower()
        if kind == "image":
            pictures.append(
                ReferenceSpec(
                    ReferenceLabel(ReferenceKind.PICTURE, len(pictures) + 1),
                    ReferenceOrigin.UPLOADED_IMAGE,
                    source_index,
                )
            )
        elif kind == "video":
            videos.append(
                ReferenceSpec(
                    ReferenceLabel(ReferenceKind.VIDEO, len(videos) + 1),
                    ReferenceOrigin.UPLOADED_VIDEO,
                    source_index,
                )
            )
        elif kind == "audio":
            standalone_audio_source_indices.append(source_index)

    audios: list[ReferenceSpec] = []
    for video, embedded_policy in zip(videos, embedded_policies, strict=True):
        if embedded_policy is not EmbeddedVideoAudioPolicy.IGNORE:
            policy = (
                AudioReferencePolicy.REUSE
                if embedded_policy is EmbeddedVideoAudioPolicy.REUSE
                else AudioReferencePolicy.AS_PROMPTED
            )
            audios.append(
                ReferenceSpec(
                    ReferenceLabel(ReferenceKind.AUDIO, len(audios) + 1),
                    ReferenceOrigin.EMBEDDED_VIDEO_AUDIO,
                    video.source_index,
                    audio_policy=policy,
                )
            )
    standalone_policy = _enum_value(
        AudioReferencePolicy,
        request.get("audio_reference_policy", "timbre"),
        default=AudioReferencePolicy.TIMBRE,
    )
    for source_index in standalone_audio_source_indices:
        label = ReferenceLabel(ReferenceKind.AUDIO, len(audios) + 1)
        audios.append(
            ReferenceSpec(
                label,
                ReferenceOrigin.STANDALONE_AUDIO,
                source_index,
                audio_policy=_standalone_audio_policy(
                    label,
                    request,
                    default=standalone_policy,
                ),
            )
        )

    generated = tuple([*pictures, *videos, *audios])
    if generated or "references" in request:
        return generated
    # Compatibility path for direct compiler callers without an upload manifest.
    discovered = discover_reference_labels(
        str(request.get("prompt", "")),
        str(request.get("dialogue", "")),
        str(request.get("soundscape", "")),
        str(request.get("music_direction", "")),
    )
    return tuple(ReferenceSpec(item, ReferenceOrigin.EXPLICIT) for item in discovered)


def compile_request(request: Mapping[str, Any]) -> CompilationResult:
    """Normalize a persisted H3 Studio request and compile it without raising.

    Ordinary recoverable limitations are represented as warnings and automatic
    adjustments. Only malformed/unsafe input produces a fatal, blocked result.
    """

    if not isinstance(request, Mapping):
        return _blocked(
            code="IR_REQUEST_TYPE",
            message="compile_request expects a mapping.",
            source={"type": type(request).__name__},
        )
    source = {
        key: request.get(key)
        for key in (
            "mode",
            "prompt",
            "num_frames",
            "frames",
            "duration_ms",
            "duration_seconds",
            "style",
            "dialogue",
            "dialogue_language",
            "soundscape",
            "music_policy",
            "references",
            "reference_labels",
            "embedded_video_audio_policy",
        )
    }
    try:
        mode = _enum_value(GenerationMode, request.get("mode"), default=None)
        if mode is None:
            raise ValueError("mode is required")
        duration_ms = _duration_ms(request)
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be blank")
        image_count, video_count, audio_count = _reference_counts(request)
        del image_count
        embedded_policy, embedded_policies = _resolve_embedded_selection(
            request,
            video_count=video_count,
        )
        enabled_video_audio_indices = tuple(
            index
            for index, policy in enumerate(embedded_policies)
            if policy is not EmbeddedVideoAudioPolicy.IGNORE
        )
        availability = _reference_video_audio_availability(request)
        silent_requested = tuple(
            index
            for index in enabled_video_audio_indices
            if index < len(availability) and availability[index] is False
        )
        if silent_requested:
            labels = ", ".join(f"<Video {index + 1}>" for index in silent_requested)
            return _blocked(
                code="IR_REFERENCE_VIDEO_HAS_NO_AUDIO",
                message=(
                    f"Requested embedded audio is unavailable because {labels} has no audio "
                    "stream. Attach a video with audio or remove the video-audio instruction."
                ),
                source=source,
                embedded_policy=EmbeddedVideoAudioPolicy.IGNORE,
            )
        unverified_requested = tuple(
            index
            for index in enabled_video_audio_indices
            if index >= len(availability) or availability[index] is None
        )
        if unverified_requested:
            labels = ", ".join(
                f"<Video {index + 1}>" for index in unverified_requested
            )
            return _blocked(
                code="IR_REFERENCE_VIDEO_AUDIO_UNVERIFIED",
                message=(
                    f"Requested embedded audio could not be verified for {labels}. "
                    "Re-upload a readable video with an audio stream or remove the "
                    "video-audio instruction."
                ),
                source=source,
                embedded_policy=EmbeddedVideoAudioPolicy.IGNORE,
            )
        normalized_request = dict(request)
        audio_label_offset = 0
        if (
            audio_count > 0
            and embedded_policy is not EmbeddedVideoAudioPolicy.IGNORE
            and not is_h3_context_ir(prompt)
        ):
            audio_label_offset = len(enabled_video_audio_indices)
            for key in (
                "prompt",
                "dialogue",
                "soundscape",
                "audio_direction",
                "music_direction",
            ):
                value = normalized_request.get(key)
                if isinstance(value, str):
                    normalized_request[key] = _shift_standalone_audio_labels(
                        value,
                        audio_label_offset,
                    )
        references = _references_from_mapping(
            normalized_request,
            embedded_policies=embedded_policies,
        )
        style_key = str(normalized_request.get("style", "natural")).strip().lower()
        style_direction = normalized_request.get("style_direction")
        if style_direction is None:
            style_direction = STYLE_DIRECTIONS.get(style_key, "")
        audio_key = str(normalized_request.get("audio_preset", "auto")).strip().lower()
        audio_direction = normalized_request.get("audio_direction")
        if audio_direction is None:
            audio_direction = AUDIO_DIRECTIONS.get(audio_key, "")
        compile_input = CompileRequest(
            mode=mode,
            prompt=str(normalized_request["prompt"]),
            duration_ms=duration_ms,
            style_direction=str(style_direction or ""),
            dialogue=str(normalized_request.get("dialogue", "") or ""),
            soundscape=str(normalized_request.get("soundscape", "") or ""),
            audio_direction=str(audio_direction or ""),
            music_policy=_enum_value(
                MusicPolicy,
                normalized_request.get("music_policy", "auto"),
                default=MusicPolicy.AUTO,
            ),
            music_direction=str(normalized_request.get("music_direction", "") or ""),
            references=references,
            dialogue_language=str(
                normalized_request.get("dialogue_language", "Japanese") or "Japanese"
            ),
            source_outfit_policy=_enum_value(
                SourceOutfitPolicy,
                normalized_request.get(
                    "source_outfit_policy", "replace_if_specified"
                ),
                default=SourceOutfitPolicy.REPLACE_IF_SPECIFIED,
            ),
            audio_reference_policy=_enum_value(
                AudioReferencePolicy,
                normalized_request.get("audio_reference_policy", "timbre"),
                default=AudioReferencePolicy.TIMBRE,
            ),
            embedded_video_audio_policy=embedded_policy,
        )
    except (TypeError, ValueError, ArithmeticError) as exc:
        return _blocked(
            code="IR_REQUEST_INVALID",
            message=str(exc),
            source=source,
        )
    try:
        result = compile_context_ir(compile_input)
    except (TypeError, ValueError, ArithmeticError) as exc:
        return _blocked(
            code="IR_COMPILE_INPUT_INVALID",
            message=str(exc),
            source=source,
            embedded_policy=compile_input.embedded_video_audio_policy,
        )
    if result.status is CompilationStatus.RAW_VALIDATED:
        ignored_raw_fields = _raw_ir_ignored_auxiliary_fields(request)
        if ignored_raw_fields:
            joined = ", ".join(ignored_raw_fields)
            result = replace(
                result,
                diagnostics=_deduplicate_diagnostics(
                    (
                        *result.diagnostics,
                        Diagnostic(
                            Severity.WARNING,
                            "IR_RAW_AUXILIARY_FIELDS_IGNORED",
                            "Raw Context-IR is preserved byte-for-byte, so these separate "
                            f"controls were not merged: {joined}. Put their intent in the "
                            "corresponding raw IR sections when needed.",
                            "request",
                        ),
                    )
                ),
            )
    if audio_label_offset:
        result = replace(
            result,
            auto_adjustments=(
                AutoAdjustment(
                    "STANDALONE_AUDIO_LABELS_SHIFTED",
                    "Standalone <Audio N> tags were shifted by "
                    f"{audio_label_offset} because explicitly enabled reference-video "
                    "soundtracks are presented to H3 before standalone audio.",
                ),
                *result.auto_adjustments,
            ),
        )
    return replace(
        result,
        embedded_video_audio_indices=enabled_video_audio_indices,
    )


def write_artifacts(result: CompilationResult, job_dir: str | os.PathLike[str]) -> CompilationResult:
    """Atomically write compiler artifacts beneath one explicit job directory."""

    if not isinstance(result, CompilationResult):
        raise TypeError("result must be a CompilationResult")
    root = Path(job_dir).resolve()
    target_dir = root / "context_ir"
    target_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[tuple[str, str]] = [
        (
            "diagnostics.json",
            json.dumps(
                {
                    "status": result.status.value,
                    "auto_adjustments": [item.to_dict() for item in result.auto_adjustments],
                    "diagnostics": [item.to_dict() for item in result.diagnostics],
                },
                ensure_ascii=False,
                indent=2,
            ),
        ),
        (
            "provenance.json",
            json.dumps(result.provenance.to_dict(), ensure_ascii=False, indent=2),
        ),
    ]
    if result.document is not None:
        payloads.append(
            (
                "document.json",
                json.dumps(result.document.to_dict(), ensure_ascii=False, indent=2),
            )
        )
    if result.ir_text is not None:
        payloads.append(("final_ir.txt", result.ir_text))

    records: list[ArtifactRecord] = []
    for filename, content in payloads:
        target = target_dir / filename
        if target.parent.resolve() != target_dir.resolve():
            raise ValueError("artifact path escaped context_ir directory")
        temporary = target.with_name(target.name + ".tmp")
        # Write the exact bytes whose size and digest are recorded below.
        # Text-mode writes translate LF to CRLF on Windows, which otherwise
        # makes the audit metadata disagree with the persisted artifact.
        temporary.write_bytes(content.encode("utf-8"))
        temporary.replace(target)
        records.append(
            ArtifactRecord(
                relative_path=f"context_ir/{filename}",
                sha256=stable_sha256(content),
                bytes=len(content.encode("utf-8")),
            )
        )
    return result.with_artifacts(tuple(records))


__all__ = [
    "AUDIO_DIRECTIONS",
    "STYLE_DIRECTIONS",
    "compile_context_ir",
    "compile_request",
    "write_artifacts",
]
