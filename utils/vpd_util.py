"""Oracle VPD configuration, authentication, and session context helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hmac
import os
import re
from typing import Iterator


DEFAULT_DATA_SCHEMA = "ADMIN"
RUNTIME_USERNAME = "SQL_ASSIST_RUNTIME"
CONTEXT_NAMESPACE = "SQL_ASSIST_CTX"
CONTEXT_ATTRIBUTE = "LOGIN_USER"
CONTEXT_PACKAGE = f"{DEFAULT_DATA_SCHEMA}.SQL_ASSIST_CTX_PKG"

# DBMS_SESSION.SET_IDENTIFIER accepts at most 64 bytes.  The configured login
# names are ASCII-only, so limiting them to 64 characters also guarantees the
# CLIENT_IDENTIFIER byte limit.
_LOGIN_USER_RE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9_$#]{0,63}|[0-9]{1,64})$")


class VpdConfigurationError(ValueError):
    """Raised when the VPD-related environment is invalid or incomplete."""


class VpdAccessError(PermissionError):
    """Raised when a user is not allowed to perform an operation."""


@dataclass(frozen=True)
class OracleConnectionParts:
    username: str
    password: str
    dsn: str


def parse_oracle_connection_string(value: str) -> OracleConnectionParts:
    """Parse user/password@dsn while allowing '/' and '@' in the password."""
    raw = str(value or "")
    slash = raw.find("/")
    at = raw.rfind("@")
    if slash <= 0 or at <= slash + 1 or at >= len(raw) - 1:
        raise VpdConfigurationError(
            "Oracle接続文字列は user/password@dsn 形式で設定してください"
        )
    return OracleConnectionParts(
        username=raw[:slash],
        password=raw[slash + 1 : at],
        dsn=raw[at + 1 :],
    )


def normalize_vpd_login_users(value: str | None = None) -> tuple[str, ...]:
    raw = os.environ.get("ORACLE_VPD_LOGIN_USERS", "") if value is None else value
    users: list[str] = []
    seen: set[str] = set()
    for item in str(raw or "").split(","):
        username = item.strip()
        if not username:
            continue
        if not _LOGIN_USER_RE.fullmatch(username):
            raise VpdConfigurationError(
                "VPDログインユーザー名は、英字で始まる64文字以内の"
                "Oracle識別子形式、または1～64桁の数字のみにしてください: "
                f"{username!r}"
            )
        key = username.casefold()
        if key == "admin":
            raise VpdConfigurationError(
                "admin は ORACLE_VPD_LOGIN_USERS に指定できません"
            )
        if key not in seen:
            seen.add(key)
            users.append(username)
    return tuple(users)


def validate_vpd_configuration() -> tuple[str, ...]:
    users = normalize_vpd_login_users()
    if not users:
        return users
    if not os.environ.get("APP_VPD_SHARED_PASSWORD"):
        raise VpdConfigurationError(
            "VPDログインユーザーを使う場合は APP_VPD_SHARED_PASSWORD が必要です"
        )
    parts = parse_oracle_connection_string(
        os.environ.get("ORACLE_VPD_RUNTIME_CONNECTION_STRING", "")
    )
    if parts.username.casefold() != RUNTIME_USERNAME.casefold():
        raise VpdConfigurationError(
            f"VPD実行接続のユーザーは {RUNTIME_USERNAME} 固定です"
        )
    return users


def configured_admin_username() -> str:
    try:
        return parse_oracle_connection_string(
            os.environ.get("ORACLE_26AI_CONNECTION_STRING", "")
        ).username
    except VpdConfigurationError:
        return "ADMIN"


def user_role(username: str | None) -> str | None:
    supplied = str(username or "")
    if supplied.casefold() == configured_admin_username().casefold():
        return "admin"
    try:
        if supplied.casefold() in {
            name.casefold() for name in normalize_vpd_login_users()
        }:
            return "vpd"
    except VpdConfigurationError:
        return None
    return None


def authenticate(username: str, password: str) -> bool:
    role = user_role(username)
    if role == "admin":
        expected = os.environ.get("APP_ADMIN_PASSWORD", "")
        # Backward-compatible migration path for existing local .env files.
        # Terraform deployments always set APP_ADMIN_PASSWORD explicitly.
        if not expected:
            try:
                expected = parse_oracle_connection_string(
                    os.environ.get("ORACLE_26AI_CONNECTION_STRING", "")
                ).password
            except VpdConfigurationError:
                expected = ""
    elif role == "vpd":
        try:
            validate_vpd_configuration()
        except VpdConfigurationError:
            return False
        expected = os.environ.get("APP_VPD_SHARED_PASSWORD", "")
    else:
        return False
    return bool(expected) and expected != "TODO" and hmac.compare_digest(
        str(password or "").encode("utf-8"), expected.encode("utf-8")
    )


def request_username(request) -> str:
    return str(getattr(request, "username", "") or "")


def require_admin(request) -> str:
    username = request_username(request)
    if user_role(username) != "admin":
        raise VpdAccessError("この操作はADMINユーザーのみ実行できます")
    return username


def require_vpd_user(username: str) -> str:
    if user_role(username) != "vpd":
        raise VpdAccessError("VPD実行が許可されていないユーザーです")
    # Preserve the spelling configured by the administrator.
    for configured in validate_vpd_configuration():
        if configured.casefold() == username.casefold():
            return configured
    raise VpdAccessError("VPD実行が許可されていないユーザーです")


def password_risk_warnings() -> list[str]:
    """Return non-blocking warnings without exposing any password."""
    warnings: list[str] = []
    shared = os.environ.get("APP_VPD_SHARED_PASSWORD", "")
    if not shared:
        return warnings
    admin_web = os.environ.get("APP_ADMIN_PASSWORD", "")
    if admin_web and hmac.compare_digest(shared, admin_web):
        warnings.append("VPD共有パスワードがADMIN Webログインパスワードと同一です")
    try:
        admin_db = parse_oracle_connection_string(
            os.environ.get("ORACLE_26AI_CONNECTION_STRING", "")
        ).password
        if hmac.compare_digest(shared, admin_db):
            warnings.append("VPD共有パスワードがADMINデータベースパスワードと同一です")
    except VpdConfigurationError:
        pass
    try:
        runtime = parse_oracle_connection_string(
            os.environ.get("ORACLE_VPD_RUNTIME_CONNECTION_STRING", "")
        ).password
        if hmac.compare_digest(shared, runtime):
            warnings.append("VPD共有パスワードが実行ユーザーのDBパスワードと同一です")
    except VpdConfigurationError:
        pass
    return warnings


@contextmanager
def vpd_runtime_connection(pool, login_user: str) -> Iterator[object]:
    """Acquire a runtime connection with a verified, session-local context."""
    canonical_user = require_vpd_user(login_user)
    conn = pool.acquire()
    discard = False
    context_set = False
    context_verified = False
    body_failed = False
    cleanup_error: Exception | None = None
    try:
        conn.current_schema = DEFAULT_DATA_SCHEMA
        with conn.cursor() as cursor:
            cursor.callproc(
                f"{CONTEXT_PACKAGE}.SET_LOGIN_USER", [canonical_user]
            )
            context_set = True
            cursor.execute(
                """
                SELECT SYS_CONTEXT(:namespace, :attribute),
                       SYS_CONTEXT('USERENV', 'CLIENT_IDENTIFIER'),
                       SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
                FROM DUAL
                """,
                namespace=CONTEXT_NAMESPACE,
                attribute=CONTEXT_ATTRIBUTE,
            )
            actual_context, actual_client_id, actual_schema = cursor.fetchone()
            expected = canonical_user.casefold()
            if (
                str(actual_context or "").casefold() != expected
                or str(actual_client_id or "").casefold() != expected
                or str(actual_schema or "").casefold()
                != DEFAULT_DATA_SCHEMA.casefold()
            ):
                raise VpdAccessError(
                    "Application Context、CLIENT_IDENTIFIER、または"
                    "CURRENT_SCHEMAの検証に失敗したため実行を拒否しました"
                )
            context_verified = True
        yield conn
    except BaseException:
        body_failed = True
        if not context_verified:
            discard = True
        raise
    finally:
        try:
            with conn.cursor() as cursor:
                cursor.callproc(f"{CONTEXT_PACKAGE}.CLEAR_LOGIN_USER")
                if context_set:
                    cursor.execute(
                        """
                        SELECT SYS_CONTEXT(:namespace, :attribute),
                               SYS_CONTEXT('USERENV', 'CLIENT_IDENTIFIER')
                        FROM DUAL
                        """,
                        namespace=CONTEXT_NAMESPACE,
                        attribute=CONTEXT_ATTRIBUTE,
                    )
                    remaining_context, remaining_client_id = cursor.fetchone()
                    if (
                        remaining_context is not None
                        or remaining_client_id is not None
                    ):
                        raise RuntimeError(
                            "Application ContextまたはCLIENT_IDENTIFIERを"
                            "消去できませんでした"
                        )
        except Exception as exc:
            cleanup_error = exc
            discard = True
        if discard:
            try:
                pool.drop(conn)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
        else:
            conn.close()
        if cleanup_error is not None and not body_failed:
            raise VpdAccessError(
                "Application Contextの消去に失敗したため接続を破棄しました"
            ) from cleanup_error
