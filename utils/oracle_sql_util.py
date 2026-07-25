"""Oracle SQL/PLSQL script parsing helpers.

The parser accepts regular SQL terminated by semicolons and SQL*Plus-style
PL/SQL units terminated by a slash on a line by itself.  It intentionally does
not implement SQL*Plus client commands such as SET, SPOOL, or @script.sql.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


class OracleScriptError(ValueError):
    """Raised when an Oracle script cannot be parsed safely."""

    def __init__(self, message: str, line: int):
        super().__init__(f"line {line}: {message}")
        self.line = line
        self.message = message


@dataclass(frozen=True)
class OracleStatement:
    text: str
    start_line: int
    statement_type: str
    is_plsql_unit: bool = False


_PLSQL_START_RE = re.compile(
    r"""
    ^\s*
    (?:
        DECLARE\b
      | BEGIN\b
      | CREATE\s+
        (?:OR\s+REPLACE\s+)?
        (?:(?:NON)?EDITIONABLE\s+)?
        (?:
            PACKAGE(?:\s+BODY)?\b
          | FUNCTION\b
          | PROCEDURE\b
          | TRIGGER\b
          | TYPE(?:\s+BODY)?\b
        )
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_UNSUPPORTED_SQLPLUS_RE = re.compile(
    r"^\s*(?:SET|SPOOL|PROMPT|WHENEVER|HOST|REM(?:ARK)?|CONNECT|COLUMN)\b|^\s*@@?",
    flags=re.IGNORECASE,
)


def _matching_q_delimiter(delimiter: str) -> str:
    return {"[": "]", "{": "}", "(": ")", "<": ">"}.get(delimiter, delimiter)


def _significant_prefix(text: str, limit: int = 512) -> str:
    """Return leading SQL with comments removed for statement classification."""
    out: list[str] = []
    i = 0
    in_line_comment = False
    in_block_comment = False
    length = len(text)
    while i < length and len(out) < limit:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < length else ""
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(" ")
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                out.append(" ")
                i += 2
            else:
                i += 1
            continue
        if ch == "-" and nxt == "-":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def statement_type(sql: str) -> str:
    prefix = _significant_prefix(str(sql or "")).lstrip()
    if re.match(r"^(SELECT|WITH)\b", prefix, flags=re.IGNORECASE):
        return "SELECT"
    if re.match(r"^COMMENT\s+ON\b", prefix, flags=re.IGNORECASE):
        return "COMMENT"
    if re.match(r"^(DECLARE|BEGIN|EXEC|EXECUTE)\b", prefix, flags=re.IGNORECASE):
        return "PLSQL"
    for keyword in (
        "INSERT",
        "UPDATE",
        "DELETE",
        "MERGE",
        "CREATE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
    ):
        if re.match(rf"^{keyword}\b", prefix, flags=re.IGNORECASE):
            return keyword
    return "UNKNOWN"


def _is_plsql_start(text: str) -> bool:
    return bool(_PLSQL_START_RE.match(_significant_prefix(text)))


def _validate_statement_start(text: str, line: int) -> None:
    prefix = _significant_prefix(text).lstrip()
    if prefix and _UNSUPPORTED_SQLPLUS_RE.match(prefix):
        token = prefix.split(None, 1)[0]
        raise OracleScriptError(
            f"unsupported SQL*Plus client command: {token}", line
        )


def parse_oracle_script(script: str) -> list[OracleStatement]:
    """Split an Oracle script without splitting semicolons inside PL/SQL."""
    source = str(script or "")
    if not source.strip():
        return []

    statements: list[OracleStatement] = []
    buffer: list[str] = []
    start_line = 1
    line_no = 1
    i = 0
    length = len(source)
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False
    q_quote_end: str | None = None
    plsql_mode = False

    def buffer_text() -> str:
        return "".join(buffer)

    def maybe_choose_mode() -> None:
        nonlocal plsql_mode
        if not plsql_mode and _is_plsql_start(buffer_text()):
            plsql_mode = True

    def finish_statement() -> None:
        nonlocal buffer, start_line, plsql_mode
        text = buffer_text().strip()
        if text:
            _validate_statement_start(text, start_line)
            statements.append(
                OracleStatement(
                    text=text,
                    start_line=start_line,
                    statement_type=statement_type(text),
                    is_plsql_unit=plsql_mode,
                )
            )
        buffer = []
        plsql_mode = False

    while i < length:
        # SQL*Plus "/" terminator is meaningful only outside literals/comments.
        if (
            (i == 0 or source[i - 1] == "\n")
            and not in_single
            and not in_double
            and not in_block_comment
            and q_quote_end is None
        ):
            line_end = source.find("\n", i)
            if line_end < 0:
                line_end = length
            current_line = source[i:line_end]
            if current_line.strip() == "/":
                if not plsql_mode:
                    raise OracleScriptError(
                        "unexpected '/' terminator outside a PL/SQL unit", line_no
                    )
                finish_statement()
                i = line_end + (1 if line_end < length else 0)
                line_no += 1 if line_end < length else 0
                start_line = line_no
                in_line_comment = False
                continue

        ch = source[i]
        nxt = source[i + 1] if i + 1 < length else ""

        if in_line_comment:
            buffer.append(ch)
            if ch == "\n":
                in_line_comment = False
                line_no += 1
            i += 1
            continue

        if in_block_comment:
            buffer.append(ch)
            if ch == "*" and nxt == "/":
                buffer.append(nxt)
                in_block_comment = False
                i += 2
            else:
                if ch == "\n":
                    line_no += 1
                i += 1
            continue

        if q_quote_end is not None:
            buffer.append(ch)
            if ch == q_quote_end and nxt == "'":
                buffer.append(nxt)
                q_quote_end = None
                i += 2
            else:
                if ch == "\n":
                    line_no += 1
                i += 1
            continue

        if in_single:
            buffer.append(ch)
            if ch == "'" and nxt == "'":
                buffer.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single = False
            if ch == "\n":
                line_no += 1
            i += 1
            continue

        if in_double:
            buffer.append(ch)
            if ch == '"' and nxt == '"':
                buffer.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_double = False
            if ch == "\n":
                line_no += 1
            i += 1
            continue

        if ch == "-" and nxt == "-":
            buffer.extend((ch, nxt))
            in_line_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            buffer.extend((ch, nxt))
            in_block_comment = True
            i += 2
            continue

        if ch in ("q", "Q") and nxt == "'" and i + 2 < length:
            delimiter = source[i + 2]
            buffer.extend((ch, nxt, delimiter))
            q_quote_end = _matching_q_delimiter(delimiter)
            i += 3
            continue

        if ch == "'":
            buffer.append(ch)
            in_single = True
            i += 1
            continue

        if ch == '"':
            buffer.append(ch)
            in_double = True
            i += 1
            continue

        buffer.append(ch)
        maybe_choose_mode()

        if ch == ";" and not plsql_mode:
            buffer.pop()  # the driver must not receive the SQL terminator
            finish_statement()
            start_line = line_no
        elif ch == "\n":
            line_no += 1
            if not buffer_text().strip():
                start_line = line_no
        i += 1

    trailing = buffer_text().strip()
    if trailing:
        maybe_choose_mode()
        if plsql_mode:
            raise OracleScriptError(
                "PL/SQL unit must end with '/' on a line by itself", start_line
            )
        finish_statement()
    return statements


def split_oracle_script(script: str) -> list[str]:
    return [statement.text for statement in parse_oracle_script(script)]


def is_single_select(script: str) -> bool:
    try:
        statements = parse_oracle_script(script)
    except OracleScriptError:
        return False
    return len(statements) == 1 and statements[0].statement_type == "SELECT"


_PROGRAM_DDL_RE = re.compile(
    r"""
    ^\s*CREATE\s+(?:OR\s+REPLACE\s+)?
    (?:(?:NON)?EDITIONABLE\s+)?
    (?P<type>PACKAGE\s+BODY|PACKAGE|FUNCTION|PROCEDURE|TRIGGER|TYPE\s+BODY|TYPE)
    \s+(?:(?P<owner>"?[A-Za-z][A-Za-z0-9_$#]*"?)\.)?
    (?P<name>"?[A-Za-z][A-Za-z0-9_$#]*"?)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def created_program(sql: str) -> tuple[str, str] | None:
    match = _PROGRAM_DDL_RE.match(_significant_prefix(str(sql or "")))
    if not match:
        return None
    object_type = re.sub(r"\s+", " ", match.group("type").upper())
    object_name = match.group("name").strip('"').upper()
    return object_type, object_name
