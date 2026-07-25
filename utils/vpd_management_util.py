"""ADMIN-only Gradio controls for Oracle VPD lifecycle management."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass

import gradio as gr
import pandas as pd

from utils.oracle_sql_util import created_program, parse_oracle_script
from utils.query_util import execute_select_sql
from utils.vpd_util import (
    CONTEXT_PACKAGE,
    RUNTIME_USERNAME,
    VpdConfigurationError,
    normalize_vpd_login_users,
    parse_oracle_connection_string,
    password_risk_warnings,
    require_admin,
    validate_vpd_configuration,
)

logger = logging.getLogger(__name__)


ACCOUNT_SQL_PREVIEW = """CREATE USER SQL_ASSIST_RUNTIME IDENTIFIED BY "<configured-secret>";
GRANT CREATE SESSION TO SQL_ASSIST_RUNTIME;
GRANT EXECUTE ON ADMIN.SQL_ASSIST_CTX_PKG TO SQL_ASSIST_RUNTIME;"""

RULE_REGISTRY_TABLE = "SQL_ASSIST_VPD_RULES"
ACCESS_NONE = "NONE"
ACCESS_FULL = "FULL"
ACCESS_ROW_MATCH = "ROW_MATCH"
ACCESS_RELATION_MATCH = "RELATION_MATCH"
ACCESS_MODE_LABELS = {
    ACCESS_NONE: "参照を許可しない",
    ACCESS_FULL: "すべての行の参照を許可",
    ACCESS_ROW_MATCH: "ログインユーザーの行のみ",
    ACCESS_RELATION_MATCH: "関連テーブルで判定",
}
ROW_MATCH_DESCRIPTION = (
    "ログインユーザー名と一致する行だけ参照できます。"
)
RELATION_MATCH_DESCRIPTION = (
    "関連テーブルでログインユーザーに一致する行を探し、その関連キーと"
    "一致する対象行だけ参照できます。"
)
ALLOWLIST_CONFIRM_LABEL = (
    "未登録オブジェクトが参照不可になることを確認しました"
)
DELETE_CONFIRM_LABEL = "削除対象を選択してください"
DELETE_BUTTON_LABEL = "アクセスルールを削除"
DELETE_EMPTY_HINT = "削除可能な管理ルール／未登録Policyはありません"
DELETE_TARGET_MANAGED = "managed_rule"
DELETE_TARGET_UNMANAGED = "unmanaged_policy"

RULE_REGISTRY_DDL = f"""CREATE TABLE ADMIN.{RULE_REGISTRY_TABLE} (
  OBJECT_NAME   VARCHAR2(128) PRIMARY KEY,
  OBJECT_TYPE   VARCHAR2(5) NOT NULL,
  ACCESS_MODE   VARCHAR2(16) NOT NULL,
  MATCH_COLUMN  VARCHAR2(128),
  RELATION_OBJECT        VARCHAR2(128),
  RELATION_TARGET_COLUMN VARCHAR2(128),
  RELATION_OBJECT_COLUMN VARCHAR2(128),
  RELATION_MATCH_COLUMN  VARCHAR2(128),
  POLICY_NAME   VARCHAR2(128),
  FUNCTION_NAME VARCHAR2(128),
  STATE         VARCHAR2(16) DEFAULT 'STAGED' NOT NULL,
  UPDATED_AT    TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
  UPDATED_BY    VARCHAR2(128) NOT NULL,
  CONSTRAINT SQL_ASSIST_VPD_RULE_MODE_CK
    CHECK (ACCESS_MODE IN ('FULL', 'ROW_MATCH', 'RELATION_MATCH')),
  CONSTRAINT SQL_ASSIST_VPD_RULE_STATE_CK
    CHECK (STATE IN ('STAGED', 'ACTIVE', 'ERROR')),
  CONSTRAINT SQL_ASSIST_VPD_RULE_COL_CK
    CHECK (
      (
        ACCESS_MODE = 'FULL'
        AND MATCH_COLUMN IS NULL
        AND RELATION_OBJECT IS NULL
        AND RELATION_TARGET_COLUMN IS NULL
        AND RELATION_OBJECT_COLUMN IS NULL
        AND RELATION_MATCH_COLUMN IS NULL
      )
      OR
      (
        ACCESS_MODE = 'ROW_MATCH'
        AND MATCH_COLUMN IS NOT NULL
        AND RELATION_OBJECT IS NULL
        AND RELATION_TARGET_COLUMN IS NULL
        AND RELATION_OBJECT_COLUMN IS NULL
        AND RELATION_MATCH_COLUMN IS NULL
      )
      OR
      (
        ACCESS_MODE = 'RELATION_MATCH'
        AND MATCH_COLUMN IS NULL
        AND RELATION_OBJECT IS NOT NULL
        AND RELATION_TARGET_COLUMN IS NOT NULL
        AND RELATION_OBJECT_COLUMN IS NOT NULL
        AND RELATION_MATCH_COLUMN IS NOT NULL
      )
    )
)"""

RULE_REGISTRY_UPGRADE_PREVIEW = f"""-- 既存台帳の場合、システムは不足列だけを追加し、
-- 既存ルールを保持したままCHECK制約を更新します。
ALTER TABLE ADMIN.{RULE_REGISTRY_TABLE} ADD (
  RELATION_OBJECT        VARCHAR2(128),
  RELATION_TARGET_COLUMN VARCHAR2(128),
  RELATION_OBJECT_COLUMN VARCHAR2(128),
  RELATION_MATCH_COLUMN  VARCHAR2(128)
);
-- SQL_ASSIST_VPD_RULE_MODE_CK / SQL_ASSIST_VPD_RULE_COL_CK は
-- RELATION_MATCH対応の定義に置き換えます。"""

CONTEXT_INSTALL_SCRIPT = """CREATE OR REPLACE PACKAGE ADMIN.SQL_ASSIST_CTX_PKG AUTHID DEFINER AS
  PROCEDURE SET_LOGIN_USER(P_LOGIN_USER IN VARCHAR2);
  PROCEDURE CLEAR_LOGIN_USER;
END SQL_ASSIST_CTX_PKG;
/

CREATE OR REPLACE PACKAGE BODY ADMIN.SQL_ASSIST_CTX_PKG AS
  PROCEDURE ASSERT_RUNTIME_CALLER AS
  BEGIN
    IF SYS_CONTEXT('USERENV', 'SESSION_USER') <> 'SQL_ASSIST_RUNTIME' THEN
      RAISE_APPLICATION_ERROR(-20000, 'Runtime session is required');
    END IF;
  END ASSERT_RUNTIME_CALLER;

  PROCEDURE SET_LOGIN_USER(P_LOGIN_USER IN VARCHAR2) AS
    L_LOGIN_USER VARCHAR2(128);
  BEGIN
    ASSERT_RUNTIME_CALLER;
    L_LOGIN_USER := TRIM(P_LOGIN_USER);
    IF L_LOGIN_USER IS NULL THEN
      RAISE_APPLICATION_ERROR(-20001, 'LOGIN_USER is required');
    END IF;
    IF LENGTHB(L_LOGIN_USER) > 64 THEN
      RAISE_APPLICATION_ERROR(-20002, 'LOGIN_USER exceeds 64 bytes');
    END IF;
    DBMS_SESSION.SET_CONTEXT('SQL_ASSIST_CTX', 'LOGIN_USER', L_LOGIN_USER);
    DBMS_SESSION.SET_IDENTIFIER(L_LOGIN_USER);
  END SET_LOGIN_USER;

  PROCEDURE CLEAR_LOGIN_USER AS
  BEGIN
    ASSERT_RUNTIME_CALLER;
    DBMS_SESSION.CLEAR_CONTEXT('SQL_ASSIST_CTX', NULL, 'LOGIN_USER');
    DBMS_SESSION.CLEAR_IDENTIFIER;
  END CLEAR_LOGIN_USER;
END SQL_ASSIST_CTX_PKG;
/

CREATE OR REPLACE CONTEXT SQL_ASSIST_CTX
  USING ADMIN.SQL_ASSIST_CTX_PKG;"""

FOUNDATION_SQL_PREVIEW = (
    CONTEXT_INSTALL_SCRIPT
    + "\n\n"
    + RULE_REGISTRY_DDL
    + ";\n"
    + "\n"
    + RULE_REGISTRY_UPGRADE_PREVIEW
    + "\n"
)

_MANAGED_NAME_RE = re.compile(r"^SQL_ASSIST_[A-Z0-9_$#]{1,116}$")
_GENERATED_FUNCTION_RE = re.compile(
    r"^SQL_ASSIST_VPD_[A-F0-9]{12}_FN$"
)
_OBJECT_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_$#]{0,127}$")
_COLUMN_NAME_RE = _OBJECT_NAME_RE
_CHARACTER_TYPES = frozenset({"CHAR", "NCHAR", "VARCHAR2", "NVARCHAR2"})
_NUMBER_TYPE = "NUMBER"
_FILTERED_ACCESS_MODES = frozenset(
    {ACCESS_ROW_MATCH, ACCESS_RELATION_MATCH}
)
_REGISTRY_RELATION_COLUMNS = (
    "RELATION_OBJECT",
    "RELATION_TARGET_COLUMN",
    "RELATION_OBJECT_COLUMN",
    "RELATION_MATCH_COLUMN",
)


@dataclass(frozen=True)
class ManagedRule:
    """Named representation of a row in the managed-rule registry."""

    object_name: str
    object_type: str
    access_mode: str
    match_column: str | None
    relation_object: str | None
    relation_target_column: str | None
    relation_object_column: str | None
    relation_match_column: str | None
    policy_name: str | None
    function_name: str | None
    state: str

    @classmethod
    def from_row(cls, row) -> ManagedRule:
        values = tuple(row)
        if len(values) == 3:
            values = (
                values[0],
                values[1],
                values[2],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "ACTIVE",
            )
        if len(values) == 7:
            values = (
                values[0],
                values[1],
                values[2],
                values[3],
                None,
                None,
                None,
                None,
                values[4],
                values[5],
                values[6],
            )
        if len(values) != 11:
            raise ValueError("アクセスルール台帳の列構成が不正です")
        return cls(
            object_name=str(values[0]),
            object_type=str(values[1]),
            access_mode=str(values[2]),
            match_column=str(values[3]) if values[3] else None,
            relation_object=str(values[4]) if values[4] else None,
            relation_target_column=str(values[5]) if values[5] else None,
            relation_object_column=str(values[6]) if values[6] else None,
            relation_match_column=str(values[7]) if values[7] else None,
            policy_name=str(values[8]) if values[8] else None,
            function_name=str(values[9]) if values[9] else None,
            state=str(values[10]),
        )


def _managed_rule(value) -> ManagedRule | None:
    if value is None or isinstance(value, ManagedRule):
        return value
    return ManagedRule.from_row(value)


@dataclass(frozen=True)
class DeleteTarget:
    """Server-validated identity for one destructive VPD operation."""

    kind: str
    object_name: str
    policy_group: str | None = None
    policy_name: str | None = None


@dataclass(frozen=True)
class RelationSpec:
    """Validated metadata needed to build a one-hop relation predicate."""

    relation_object: str
    target_column: str
    target_data_type: str
    object_column: str
    object_data_type: str
    match_column: str
    match_data_type: str


class UnsupportedOracleReleaseError(RuntimeError):
    """Raised when required Oracle 26ai capabilities are unavailable."""


def _runtime_parts():
    parts = parse_oracle_connection_string(
        os.environ.get("ORACLE_VPD_RUNTIME_CONNECTION_STRING", "")
    )
    if parts.username.casefold() != RUNTIME_USERNAME.casefold():
        raise VpdConfigurationError(
            f"実行ユーザーは {RUNTIME_USERNAME} 固定です"
        )
    return parts


def _oracle_release(cursor) -> tuple[str, str, str]:
    cursor.execute(
        """
        SELECT product, version, version_full
        FROM product_component_version
        WHERE version_full IS NOT NULL
          AND (
            UPPER(product) LIKE 'ORACLE AI DATABASE%'
            OR UPPER(product) LIKE 'ORACLE DATABASE%'
          )
        FETCH FIRST 1 ROW ONLY
        """
    )
    row = cursor.fetchone()
    if not row:
        return "", "", ""
    return (
        str(row[0] or ""),
        str(row[1] or ""),
        str(row[2] or ""),
    )


def _is_26ai_release(
    product: str,
    version: str,
    version_full: str,
) -> bool:
    """Recognize 26ai's marketing name and its 23.26+ release numbering."""
    if "26AI" in str(product or "").upper():
        return True
    numbers = [
        int(value)
        for value in re.findall(r"\d+", str(version_full or version or ""))
    ]
    if not numbers:
        return False
    if numbers[0] >= 26:
        return True
    return numbers[0] == 23 and len(numbers) > 1 and numbers[1] >= 26


def _release_display(release: tuple[str, str, str]) -> str:
    product, version, version_full = release
    release_number = version_full or version or "unknown"
    return f"{product or 'Oracle Database'} ({release_number})"


def _require_26ai(cursor) -> str:
    release = _oracle_release(cursor)
    if not _is_26ai_release(*release):
        raise UnsupportedOracleReleaseError(
            f"{_release_display(release)} を検出しました。ON SCHEMA権限には"
            "Oracle AI Database 26aiが必要です。データベース全体の"
            "SELECT ANY TABLEにはフォールバックしません。"
        )
    # This view is available starting with 26ai and is the capability used by
    # the runtime privilege audit.  Probe it here so branding/version parsing
    # can never silently enable an unsupported database.
    cursor.execute(
        "SELECT privilege, schema FROM session_schema_privs WHERE 1 = 0"
    )
    return _release_display(release)


def _has_system_privilege(cursor, privilege: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM session_privs
        WHERE privilege = :privilege
        """,
        privilege=privilege,
    )
    return int(cursor.fetchone()[0]) > 0


def _has_schema_privilege(
    cursor,
    privilege: str,
    schema: str,
) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM session_schema_privs
        WHERE privilege = :privilege
          AND schema = :schema
        """,
        privilege=privilege,
        schema=schema,
    )
    return int(cursor.fetchone()[0]) > 0


def _can_grant_schema_privileges(cursor) -> bool:
    return any(
        _has_system_privilege(cursor, privilege)
        for privilege in ("GRANT ANY SCHEMA PRIVILEGE", "GRANT ANY PRIVILEGE")
    )


def _can_administer_admin_rls(cursor) -> bool:
    privilege = "ADMINISTER ROW LEVEL SECURITY POLICY"
    return _has_system_privilege(cursor, privilege) or _has_schema_privilege(
        cursor,
        privilege,
        "ADMIN",
    )


def _can_execute_package(cursor, owner: str, package_name: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM all_tab_privs_recd
        WHERE owner = :owner
          AND table_name = :package_name
          AND privilege = 'EXECUTE'
        """,
        owner=owner,
        package_name=package_name,
    )
    return int(cursor.fetchone()[0]) > 0


def _require_schema_grant_capability(cursor) -> None:
    if not _can_grant_schema_privileges(cursor):
        raise RuntimeError(
            "ADMINセッションにGRANT ANY SCHEMA PRIVILEGEまたは"
            "GRANT ANY PRIVILEGEがありません"
        )


def _require_rls_admin_capability(cursor) -> None:
    if not _can_administer_admin_rls(cursor):
        raise RuntimeError(
            "ADMINセッションにADMINISTER ROW LEVEL SECURITY POLICY"
            "（システム権限またはADMIN schema権限）がありません"
        )
    if not _can_execute_package(cursor, "SYS", "DBMS_RLS"):
        raise RuntimeError(
            "ADMINセッションにSYS.DBMS_RLSのEXECUTE権限がありません"
        )


def _unexpected_runtime_object_privileges(
    privileges: set[tuple[str, str, str, str]],
) -> list[str]:
    def expected(grant: tuple[str, str, str, str]) -> bool:
        grantee, owner, object_name, privilege = grant
        if grant == (
            RUNTIME_USERNAME,
            "ADMIN",
            "SQL_ASSIST_CTX_PKG",
            "EXECUTE",
        ):
            return True
        return (
            grantee == RUNTIME_USERNAME
            and owner == "ADMIN"
            and object_name != RULE_REGISTRY_TABLE
            and privilege in {"READ", "SELECT"}
        )

    return sorted(
        f"{privilege} ON {owner}.{object_name} ({grantee})"
        for grantee, owner, object_name, privilege in privileges
        if not expected((grantee, owner, object_name, privilege))
    )


def _compile_errors(cursor, object_type: str, object_name: str) -> list[str]:
    cursor.execute(
        """
        SELECT line, position, text
        FROM user_errors
        WHERE name = :name AND type = :object_type
        ORDER BY sequence
        """,
        name=object_name.upper(),
        object_type=object_type.upper(),
    )
    return [
        f"line {line}, position {position}: {text}"
        for line, position, text in cursor.fetchall()
    ]


def _require_valid_policy_function(cursor, function_name: str) -> None:
    cursor.execute(
        """
        SELECT status
        FROM user_objects
        WHERE object_name = :function_name
          AND object_type = 'FUNCTION'
        """,
        function_name=function_name,
    )
    rows = cursor.fetchall()
    if len(rows) != 1:
        raise ValueError(
            f"Policy Function {function_name}がADMIN schemaに存在しません"
        )
    errors = _compile_errors(cursor, "FUNCTION", function_name)
    if str(rows[0][0] or "").upper() != "VALID" or errors:
        detail = "\n".join(errors) if errors else "STATUS=INVALID"
        raise RuntimeError(
            "Policy Functionにコンパイルエラーがあるため追加できません:\n"
            + detail
        )


def _managed_name(value: str, label: str) -> str:
    name = str(value or "").strip().upper()
    if not _MANAGED_NAME_RE.fullmatch(name):
        raise ValueError(f"{label}はSQL_ASSIST_で始まるOracle識別子にしてください")
    return name


def _object_name(value: str) -> str:
    name = str(value or "").strip().upper()
    if not _OBJECT_NAME_RE.fullmatch(name):
        raise ValueError("オブジェクト名が不正です")
    return name


def _column_name(value: str) -> str:
    name = str(value or "").strip().upper()
    if not _COLUMN_NAME_RE.fullmatch(name):
        raise ValueError("照合列名が不正です")
    return name


def _quoted_identifier(value: str, *, column: bool = False) -> str:
    name = _column_name(value) if column else _object_name(value)
    return f'"{name}"'


def _rule_names(object_name: str) -> tuple[str, str]:
    object_name = _object_name(object_name)
    digest = hashlib.sha256(object_name.encode("ascii")).hexdigest()[:12].upper()
    return (
        f"SQL_ASSIST_VPD_{digest}_POL",
        f"SQL_ASSIST_VPD_{digest}_FN",
    )


def _match_data_type(value: str) -> str:
    data_type = str(value or "").strip().upper()
    if data_type not in _CHARACTER_TYPES | {_NUMBER_TYPE}:
        raise ValueError("照合列のデータ型が不正です")
    return data_type


def _match_predicate(
    match_column: str,
    match_data_type: str,
    login_user_expression: str,
    *,
    qualifier: str | None = None,
) -> str:
    quoted_column = _quoted_identifier(match_column, column=True)
    column_expression = (
        f"{_object_name(qualifier)}.{quoted_column}"
        if qualifier
        else quoted_column
    )
    data_type = _match_data_type(match_data_type)
    if data_type == _NUMBER_TYPE:
        return (
            f"{column_expression} = TO_NUMBER({login_user_expression} "
            "DEFAULT NULL ON CONVERSION ERROR)"
        )
    return f"{column_expression} = {login_user_expression}"


def _relation_predicate(spec: RelationSpec, login_user_expression: str) -> str:
    target_column = _quoted_identifier(spec.target_column, column=True)
    relation_object = _quoted_identifier(spec.relation_object)
    object_column = _quoted_identifier(spec.object_column, column=True)
    match_predicate = _match_predicate(
        spec.match_column,
        spec.match_data_type,
        login_user_expression,
        qualifier="R",
    )
    return (
        f"{target_column} IN (SELECT R.{object_column} "
        f"FROM ADMIN.{relation_object} R WHERE {match_predicate})"
    )


def _policy_function_ddl(
    object_name: str,
    match_column: str | None = None,
    match_data_type: str | None = None,
    *,
    access_mode: str = ACCESS_ROW_MATCH,
    relation_spec: RelationSpec | None = None,
) -> str:
    object_name = _object_name(object_name)
    _, function_name = _rule_names(object_name)
    login_user_expression = (
        "SYS_CONTEXT(''SQL_ASSIST_CTX'', ''LOGIN_USER'')"
    )
    if access_mode == ACCESS_ROW_MATCH:
        predicate = _match_predicate(
            _column_name(match_column),
            _match_data_type(match_data_type),
            login_user_expression,
        )
    elif access_mode == ACCESS_RELATION_MATCH and relation_spec is not None:
        predicate = _relation_predicate(
            relation_spec,
            login_user_expression,
        )
    else:
        raise ValueError("Policy Functionのアクセスモードが不正です")
    return f"""CREATE OR REPLACE FUNCTION ADMIN.{function_name}(
  P_SCHEMA_NAME IN VARCHAR2,
  P_OBJECT_NAME IN VARCHAR2
) RETURN VARCHAR2
AUTHID DEFINER
AS
BEGIN
  IF SYS_CONTEXT('SQL_ASSIST_CTX', 'LOGIN_USER') IS NULL THEN
    RETURN '1=0';
  END IF;
  RETURN '{predicate}';
END;"""


def _rule_sql_preview(
    object_name: str,
    access_mode: str,
    match_column: str | None = None,
    match_data_type: str | None = None,
    relation_spec: RelationSpec | None = None,
) -> str:
    object_name = _object_name(object_name)
    quoted_object = _quoted_identifier(object_name)
    access_mode = str(access_mode or "").upper()
    revoke_sql = (
        f"REVOKE READ ON ADMIN.{quoted_object} FROM {RUNTIME_USERNAME};\n"
        f"REVOKE SELECT ON ADMIN.{quoted_object} FROM {RUNTIME_USERNAME};"
    )
    if access_mode == ACCESS_NONE:
        return (
            "-- 実際には現在付与されている権限だけを失効します。\n"
            f"{revoke_sql}"
        )
    if access_mode == ACCESS_FULL:
        return (
            "-- 既存の管理対象VPD Policyがある場合は、READを先に失効して"
            "から削除します。\n"
            f"GRANT READ ON ADMIN.{quoted_object} TO {RUNTIME_USERNAME};"
        )
    if access_mode not in _FILTERED_ACCESS_MODES:
        raise ValueError("アクセスモードが不正です")
    if access_mode == ACCESS_ROW_MATCH:
        match_column = _column_name(match_column)
        match_data_type = _match_data_type(match_data_type)
    elif relation_spec is None:
        raise ValueError("関連条件が不足しています")
    policy_name, function_name = _rule_names(object_name)
    function_ddl = _policy_function_ddl(
        object_name,
        match_column,
        match_data_type,
        access_mode=access_mode,
        relation_spec=relation_spec,
    )
    return f"""{function_ddl}
/

BEGIN
  DBMS_RLS.ADD_POLICY(
    object_schema   => 'ADMIN',
    object_name     => '{object_name}',
    policy_name     => '{policy_name}',
    function_schema => 'ADMIN',
    policy_function => '{function_name}',
    statement_types => 'SELECT',
    enable          => FALSE,
    policy_type     => DBMS_RLS.CONTEXT_SENSITIVE,
    namespace       => 'SQL_ASSIST_CTX',
    attribute       => 'LOGIN_USER'
  );
END;
/

BEGIN
  DBMS_RLS.ENABLE_POLICY(
    object_schema => 'ADMIN',
    object_name   => '{object_name}',
    policy_name   => '{policy_name}',
    enable        => TRUE
  );
END;
/

GRANT READ ON ADMIN.{quoted_object} TO {RUNTIME_USERNAME};"""


def _registry_exists(cursor) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM user_tables
        WHERE table_name = :table_name
        """,
        table_name=RULE_REGISTRY_TABLE,
    )
    return int(cursor.fetchone()[0]) > 0


def _registry_columns(cursor) -> set[str]:
    if not _registry_exists(cursor):
        return set()
    cursor.execute(
        """
        SELECT column_name
        FROM user_tab_columns
        WHERE table_name = :table_name
        """,
        table_name=RULE_REGISTRY_TABLE,
    )
    return {str(row[0]).upper() for row in cursor.fetchall()}


def _registry_constraints_current(cursor) -> bool:
    if not _registry_exists(cursor):
        return False
    cursor.execute(
        """
        SELECT constraint_name, search_condition_vc
        FROM user_constraints
        WHERE table_name = :table_name
          AND constraint_name IN (
            'SQL_ASSIST_VPD_RULE_MODE_CK',
            'SQL_ASSIST_VPD_RULE_COL_CK'
          )
        """,
        table_name=RULE_REGISTRY_TABLE,
    )
    definitions = {
        str(name).upper(): str(condition or "").upper()
        for name, condition in cursor.fetchall()
    }
    return (
        "RELATION_MATCH"
        in definitions.get("SQL_ASSIST_VPD_RULE_MODE_CK", "")
        and "RELATION_OBJECT"
        in definitions.get("SQL_ASSIST_VPD_RULE_COL_CK", "")
    )


def _registry_schema_current(cursor) -> bool:
    columns = _registry_columns(cursor)
    return set(_REGISTRY_RELATION_COLUMNS).issubset(
        columns
    ) and _registry_constraints_current(cursor)


def _require_registry_current(cursor) -> None:
    if not _registry_exists(cursor):
        raise RuntimeError(
            "データアクセスルール台帳が未インストールです。"
            "先に「VPD基盤のインストール/更新」を実行してください"
        )
    if not _registry_schema_current(cursor):
        raise RuntimeError(
            "データアクセスルール台帳の更新が必要です。"
            "先に「VPD基盤のインストール/更新」を実行してください"
        )


def _replace_registry_rule_constraints(cursor) -> None:
    for constraint_name in (
        "SQL_ASSIST_VPD_RULE_MODE_CK",
        "SQL_ASSIST_VPD_RULE_COL_CK",
    ):
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM user_constraints
            WHERE table_name = :table_name
              AND constraint_name = :constraint_name
            """,
            table_name=RULE_REGISTRY_TABLE,
            constraint_name=constraint_name,
        )
        if int(cursor.fetchone()[0]) > 0:
            cursor.execute(
                f"ALTER TABLE ADMIN.{RULE_REGISTRY_TABLE} "
                f"DROP CONSTRAINT {constraint_name}"
            )
    cursor.execute(
        f"""
        ALTER TABLE ADMIN.{RULE_REGISTRY_TABLE}
        ADD CONSTRAINT SQL_ASSIST_VPD_RULE_MODE_CK
        CHECK (ACCESS_MODE IN ('FULL', 'ROW_MATCH', 'RELATION_MATCH'))
        """
    )
    cursor.execute(
        f"""
        ALTER TABLE ADMIN.{RULE_REGISTRY_TABLE}
        ADD CONSTRAINT SQL_ASSIST_VPD_RULE_COL_CK
        CHECK (
          (
            ACCESS_MODE = 'FULL'
            AND MATCH_COLUMN IS NULL
            AND RELATION_OBJECT IS NULL
            AND RELATION_TARGET_COLUMN IS NULL
            AND RELATION_OBJECT_COLUMN IS NULL
            AND RELATION_MATCH_COLUMN IS NULL
          )
          OR
          (
            ACCESS_MODE = 'ROW_MATCH'
            AND MATCH_COLUMN IS NOT NULL
            AND RELATION_OBJECT IS NULL
            AND RELATION_TARGET_COLUMN IS NULL
            AND RELATION_OBJECT_COLUMN IS NULL
            AND RELATION_MATCH_COLUMN IS NULL
          )
          OR
          (
            ACCESS_MODE = 'RELATION_MATCH'
            AND MATCH_COLUMN IS NULL
            AND RELATION_OBJECT IS NOT NULL
            AND RELATION_TARGET_COLUMN IS NOT NULL
            AND RELATION_OBJECT_COLUMN IS NOT NULL
            AND RELATION_MATCH_COLUMN IS NOT NULL
          )
        )
        """
    )


def _ensure_rule_registry(cursor) -> bool:
    if not _registry_exists(cursor):
        cursor.execute(RULE_REGISTRY_DDL)
        return True
    columns = _registry_columns(cursor)
    changed = False
    for column_name in _REGISTRY_RELATION_COLUMNS:
        if column_name in columns:
            continue
        cursor.execute(
            f"ALTER TABLE ADMIN.{RULE_REGISTRY_TABLE} "
            f"ADD ({column_name} VARCHAR2(128))"
        )
        changed = True
    if changed or not _registry_constraints_current(cursor):
        _replace_registry_rule_constraints(cursor)
        changed = True
    return changed


def _registry_select_list(cursor) -> str:
    columns = _registry_columns(cursor)
    relation_projection = [
        (
            column_name.lower()
            if column_name in columns
            else f"CAST(NULL AS VARCHAR2(128)) AS {column_name.lower()}"
        )
        for column_name in _REGISTRY_RELATION_COLUMNS
    ]
    return ", ".join(
        [
            "object_name",
            "object_type",
            "access_mode",
            "match_column",
            *relation_projection,
            "policy_name",
            "function_name",
            "state",
        ]
    )


def _registry_rule(cursor, object_name: str) -> ManagedRule | None:
    object_name = _object_name(object_name)
    if not _registry_exists(cursor):
        return None
    select_list = _registry_select_list(cursor)
    cursor.execute(
        f"""
        SELECT {select_list}
        FROM {RULE_REGISTRY_TABLE}
        WHERE object_name = :object_name
        """,
        object_name=object_name,
    )
    row = cursor.fetchone()
    return ManagedRule.from_row(row) if row else None


def _registry_rules(cursor) -> dict[str, ManagedRule]:
    if not _registry_exists(cursor):
        return {}
    select_list = _registry_select_list(cursor)
    cursor.execute(
        f"""
        SELECT {select_list}
        FROM {RULE_REGISTRY_TABLE}
        ORDER BY object_name
        """
    )
    rules = [ManagedRule.from_row(row) for row in cursor.fetchall()]
    return {rule.object_name: rule for rule in rules}


def _upsert_registry_rule(
    cursor,
    *,
    object_name: str,
    object_type: str,
    access_mode: str,
    match_column: str | None,
    relation_object: str | None = None,
    relation_target_column: str | None = None,
    relation_object_column: str | None = None,
    relation_match_column: str | None = None,
    policy_name: str | None,
    function_name: str | None,
    state: str,
    updated_by: str,
) -> None:
    _ensure_rule_registry(cursor)
    cursor.execute(
        f"""
        MERGE INTO {RULE_REGISTRY_TABLE} target
        USING (
          SELECT :object_name AS object_name FROM dual
        ) source
        ON (target.object_name = source.object_name)
        WHEN MATCHED THEN UPDATE SET
          target.object_type = :object_type,
          target.access_mode = :access_mode,
          target.match_column = :match_column,
          target.relation_object = :relation_object,
          target.relation_target_column = :relation_target_column,
          target.relation_object_column = :relation_object_column,
          target.relation_match_column = :relation_match_column,
          target.policy_name = :policy_name,
          target.function_name = :function_name,
          target.state = :state,
          target.updated_at = SYSTIMESTAMP,
          target.updated_by = :updated_by
        WHEN NOT MATCHED THEN INSERT (
          object_name, object_type, access_mode, match_column, relation_object,
          relation_target_column, relation_object_column,
          relation_match_column,
          policy_name, function_name, state, updated_at, updated_by
        ) VALUES (
          :object_name, :object_type, :access_mode, :match_column,
          :relation_object, :relation_target_column, :relation_object_column,
          :relation_match_column,
          :policy_name, :function_name, :state, SYSTIMESTAMP, :updated_by
        )
        """,
        object_name=object_name,
        object_type=object_type,
        access_mode=access_mode,
        match_column=match_column,
        relation_object=relation_object,
        relation_target_column=relation_target_column,
        relation_object_column=relation_object_column,
        relation_match_column=relation_match_column,
        policy_name=policy_name,
        function_name=function_name,
        state=state,
        updated_by=updated_by,
    )


def _set_registry_state(cursor, object_name: str, state: str) -> None:
    if not _registry_exists(cursor):
        return
    cursor.execute(
        f"""
        UPDATE {RULE_REGISTRY_TABLE}
        SET state = :state, updated_at = SYSTIMESTAMP
        WHERE object_name = :object_name
        """,
        state=state,
        object_name=_object_name(object_name),
    )


def _delete_registry_rule(cursor, object_name: str) -> None:
    if not _registry_exists(cursor):
        return
    cursor.execute(
        f"DELETE FROM {RULE_REGISTRY_TABLE} WHERE object_name = :object_name",
        object_name=_object_name(object_name),
    )


def _business_objects(cursor) -> list[tuple[str, str]]:
    cursor.execute(
        """
        SELECT object_name, object_type
        FROM user_objects
        WHERE object_type IN ('TABLE', 'VIEW')
          AND subobject_name IS NULL
          AND temporary = 'N'
          AND generated = 'N'
          AND secondary = 'N'
          AND oracle_maintained = 'N'
          AND object_name NOT LIKE 'BIN$%'
          AND object_name NOT LIKE 'VECTOR$%'
          AND object_name <> :registry_table
          AND REGEXP_LIKE(object_name, '^[A-Z][A-Z0-9_$#]{0,127}$')
        ORDER BY object_type, object_name
        """,
        registry_table=RULE_REGISTRY_TABLE,
    )
    return [
        (_object_name(object_name), str(object_type).upper())
        for object_name, object_type in cursor.fetchall()
    ]


def _require_business_object(cursor, object_name: str) -> str:
    object_name = _object_name(object_name)
    objects = dict(_business_objects(cursor))
    object_type = objects.get(object_name)
    if object_type not in {"TABLE", "VIEW"}:
        raise ValueError(
            "対象はADMIN schemaの管理可能なテーブルまたはビューにしてください"
        )
    return object_type


def _match_columns(cursor, object_name: str) -> list[tuple[str, str, str]]:
    object_name = _object_name(object_name)
    _require_business_object(cursor, object_name)
    cursor.execute(
        """
        SELECT column_name, data_type, nullable
        FROM user_tab_cols
        WHERE table_name = :object_name
          AND (
            data_type IN ('CHAR', 'NCHAR', 'VARCHAR2', 'NVARCHAR2')
            OR (
              data_type = 'NUMBER'
              AND (data_scale IS NULL OR data_scale = 0)
            )
          )
          AND hidden_column = 'NO'
          AND REGEXP_LIKE(column_name, '^[A-Z][A-Z0-9_$#]{0,127}$')
        ORDER BY column_id
        """,
        object_name=object_name,
    )
    return [
        (_column_name(name), str(data_type).upper(), str(nullable).upper())
        for name, data_type, nullable in cursor.fetchall()
    ]


def _require_match_column(
    cursor,
    object_name: str,
    match_column: str,
) -> tuple[str, str]:
    match_column = _column_name(match_column)
    columns = {
        name: data_type
        for name, data_type, _nullable in _match_columns(cursor, object_name)
    }
    data_type = columns.get(match_column)
    if data_type is None:
        raise ValueError(
            "照合列は対象オブジェクトの文字列型列、または整数型NUMBER列から"
            "選択してください"
        )
    return match_column, _match_data_type(data_type)


def _relation_objects(
    cursor,
    target_object: str | None = None,
) -> list[tuple[str, str]]:
    target = _object_name(target_object) if target_object else None
    return [
        (object_name, object_type)
        for object_name, object_type in _business_objects(cursor)
        if object_type == "TABLE" and object_name != target
    ]


def _require_relation_table(cursor, object_name: str) -> str:
    object_name = _object_name(object_name)
    tables = {name for name, _object_type in _relation_objects(cursor)}
    if object_name not in tables:
        raise ValueError(
            "関連テーブルはADMIN schemaの管理可能な業務テーブルから"
            "選択してください"
        )
    return object_name


def _relation_columns(
    cursor,
    object_name: str,
) -> list[tuple[str, str, str]]:
    object_name = _object_name(object_name)
    _require_business_object(cursor, object_name)
    cursor.execute(
        """
        SELECT column_name, data_type, nullable
        FROM user_tab_cols
        WHERE table_name = :object_name
          AND data_type IN (
            'CHAR', 'NCHAR', 'VARCHAR2', 'NVARCHAR2', 'NUMBER'
          )
          AND hidden_column = 'NO'
          AND REGEXP_LIKE(column_name, '^[A-Z][A-Z0-9_$#]{0,127}$')
        ORDER BY column_id
        """,
        object_name=object_name,
    )
    return [
        (_column_name(name), str(data_type).upper(), str(nullable).upper())
        for name, data_type, nullable in cursor.fetchall()
    ]


def _require_relation_spec(
    cursor,
    target_object: str,
    relation_target_column: str,
    relation_object: str,
    relation_object_column: str,
    relation_match_column: str,
) -> RelationSpec:
    target_object = _object_name(target_object)
    _require_business_object(cursor, target_object)
    relation_object = _require_relation_table(cursor, relation_object)
    if relation_object == target_object:
        raise ValueError("対象自身を関連テーブルには指定できません")

    target_columns = {
        name: data_type
        for name, data_type, _nullable in _relation_columns(
            cursor, target_object
        )
    }
    object_columns = {
        name: data_type
        for name, data_type, _nullable in _relation_columns(
            cursor, relation_object
        )
    }
    target_column = _column_name(relation_target_column)
    object_column = _column_name(relation_object_column)
    target_data_type = target_columns.get(target_column)
    object_data_type = object_columns.get(object_column)
    if target_data_type is None or object_data_type is None:
        raise ValueError(
            "関連列は文字列型またはNUMBER型の表示列から選択してください"
        )
    if target_data_type != object_data_type:
        raise ValueError(
            "対象側と関連テーブル側の関連列は同じデータ型にしてください"
        )
    match_column, match_data_type = _require_match_column(
        cursor,
        relation_object,
        relation_match_column,
    )
    return RelationSpec(
        relation_object=relation_object,
        target_column=target_column,
        target_data_type=target_data_type,
        object_column=object_column,
        object_data_type=object_data_type,
        match_column=match_column,
        match_data_type=match_data_type,
    )


def _relation_login_column_has_index(
    cursor,
    relation_object: str,
    relation_match_column: str,
) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM user_ind_columns columns_
        JOIN user_indexes indexes_
          ON indexes_.index_name = columns_.index_name
         AND indexes_.table_name = columns_.table_name
        WHERE columns_.table_name = :object_name
          AND columns_.column_name = :column_name
          AND columns_.column_position = 1
          AND indexes_.status = 'VALID'
          AND indexes_.visibility = 'VISIBLE'
        """,
        object_name=_object_name(relation_object),
        column_name=_column_name(relation_match_column),
    )
    return int(cursor.fetchone()[0]) > 0


def _require_unambiguous_number_login_users(
    users: tuple[str, ...],
) -> None:
    aliases: dict[str, list[str]] = {}
    for username in users:
        if not (username.isascii() and username.isdecimal()):
            continue
        numeric_key = username.lstrip("0") or "0"
        aliases.setdefault(numeric_key, []).append(username)
    collisions = [names for names in aliases.values() if len(names) > 1]
    if not collisions:
        return
    details = ", ".join("/".join(names) for names in collisions)
    raise ValueError(
        "NUMBER照合列では同じ数値になるログインユーザーを併用できません: "
        f"{details}"
    )


def _runtime_has_schema_read(cursor) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM dba_schema_privs
        WHERE grantee = :username
          AND schema = 'ADMIN'
          AND privilege IN ('READ ANY TABLE', 'SELECT ANY TABLE')
        """,
        username=RUNTIME_USERNAME,
    )
    return int(cursor.fetchone()[0]) > 0


def _runtime_object_read_privileges(
    cursor,
    object_name: str | None = None,
) -> dict[str, set[str]]:
    sql = """
        SELECT table_name, privilege
        FROM dba_tab_privs
        WHERE grantee = :username
          AND owner = 'ADMIN'
          AND privilege IN ('READ', 'SELECT')
    """
    binds = {"username": RUNTIME_USERNAME}
    if object_name is not None:
        sql += " AND table_name = :object_name"
        binds["object_name"] = _object_name(object_name)
    sql += " ORDER BY table_name, privilege"
    cursor.execute(sql, **binds)
    privileges: dict[str, set[str]] = {}
    for name, privilege in cursor.fetchall():
        privileges.setdefault(str(name), set()).add(str(privilege).upper())
    return privileges


def _safe_database_error(exc: Exception, *, runtime: bool = False) -> str:
    """Return an actionable database error without connection details."""
    error_text = str(exc or "")
    match = re.search(r"\b(?:ORA|DPY)-\d+\b", error_text, re.IGNORECASE)
    error_code = match.group(0).upper() if match else ""
    if error_code == "ORA-01017":
        if not runtime:
            return (
                "ORA-01017: ADMINデータベースの認証に失敗しました。"
                "ORACLE_26AI_CONNECTION_STRINGを確認してください"
            )
        return (
            "ORA-01017: 実行ユーザーの認証に失敗しました。"
            "Terraform設定の実行ユーザー認証情報とDBユーザーの"
            "パスワードが一致しているか確認してください"
        )
    if error_code in {"ORA-28000", "ORA-28001"}:
        account = "実行ユーザー" if runtime else "ADMINデータベースユーザー"
        return (
            f"{error_code}: {account}がロック中またはパスワード期限切れです。"
            "DBユーザーの状態を確認してください"
        )
    if error_code in {"ORA-12514", "DPY-6001"}:
        return (
            f"{error_code}: Oracle Listenerに対象サービスが登録されていません"
        )
    target = "VPD実行プール" if runtime else "ADMINデータベース"
    if error_code:
        return f"{target}の確認に失敗しました（{error_code}）"
    return f"{target}の確認に失敗しました。サーバーログを確認してください"


def _direct_runtime_privilege_rows(cursor) -> list[list[str]]:
    cursor.execute(
        """
        SELECT privilege
        FROM dba_sys_privs
        WHERE grantee = :username
        ORDER BY privilege
        """,
        username=RUNTIME_USERNAME,
    )
    direct_privileges = {row[0] for row in cursor.fetchall()}
    cursor.execute(
        """
        SELECT granted_role
        FROM dba_role_privs
        WHERE grantee = :username
        ORDER BY granted_role
        """,
        username=RUNTIME_USERNAME,
    )
    direct_roles = {row[0] for row in cursor.fetchall()}
    cursor.execute(
        """
        SELECT privilege, schema
        FROM dba_schema_privs
        WHERE grantee = :username
        ORDER BY schema, privilege
        """,
        username=RUNTIME_USERNAME,
    )
    schema_privileges = {(row[0], row[1]) for row in cursor.fetchall()}
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM dba_tab_privs
        WHERE grantee = :username
          AND owner = 'ADMIN'
          AND table_name = 'SQL_ASSIST_CTX_PKG'
          AND privilege = 'EXECUTE'
        """,
        username=RUNTIME_USERNAME,
    )
    has_context_execute = int(cursor.fetchone()[0]) > 0
    direct_object_reads = _runtime_object_read_privileges(cursor)
    registered_objects = {
        name
        for name, row in _registry_rules(cursor).items()
        if row.state.upper() == "ACTIVE"
    }
    unregistered_reads = sorted(
        name for name in direct_object_reads if name not in registered_objects
    )

    rows = [
        [
            "実行ユーザー直接権限: CREATE SESSION",
            "OK" if "CREATE SESSION" in direct_privileges else "不足",
            "付与あり" if "CREATE SESSION" in direct_privileges else "付与なし",
        ],
        [
            "実行ユーザー直接権限: ADMIN.SQL_ASSIST_CTX_PKG EXECUTE",
            "OK" if has_context_execute else "不足",
            "付与あり" if has_context_execute else "付与なし",
        ],
    ]
    for privilege in (
        "EXEMPT ACCESS POLICY",
        "READ ANY TABLE",
        "SELECT ANY TABLE",
        "SELECT ANY DICTIONARY",
        "CREATE ANY CONTEXT",
        "EXECUTE ANY PROCEDURE",
    ):
        granted = privilege in direct_privileges
        rows.append(
            [
                f"実行ユーザー直接権限: {privilege}",
                "危険" if granted else "OK",
                "付与あり" if granted else "付与なし",
            ]
        )
    for role in ("DBA", "ADB_DBA"):
        granted = role in direct_roles
        rows.append(
            [
                f"実行ユーザー直接Role: {role}",
                "危険" if granted else "OK",
                "付与あり" if granted else "付与なし",
            ]
        )

    broad_schema_reads = {
        item
        for item in schema_privileges
        if item in {
            ("READ ANY TABLE", "ADMIN"),
            ("SELECT ANY TABLE", "ADMIN"),
        }
    }
    rows.append(
        [
            "実行ユーザー直接権限: ADMIN schema全表読取",
            "移行警告" if broad_schema_reads else "OK",
            (
                "付与あり。テーブル許可リストへの移行が必要です"
                if broad_schema_reads
                else "付与なし"
            ),
        ]
    )
    unexpected_schema_privileges = sorted(
        f"{privilege} ON SCHEMA {schema}"
        for privilege, schema in schema_privileges
        if (privilege, schema) not in broad_schema_reads
    )
    rows.append(
        [
            "実行ユーザーの想定外直接Schema権限",
            "危険" if unexpected_schema_privileges else "OK",
            ", ".join(unexpected_schema_privileges) or "なし",
        ]
    )
    rows.append(
        [
            "実行ユーザーの未登録テーブル権限",
            "危険" if unregistered_reads else "OK",
            ", ".join(unregistered_reads) or "なし",
        ]
    )
    return rows


def _effective_runtime_privilege_rows(cursor) -> list[list[str]]:
    cursor.execute(
        """
        SELECT privilege
        FROM session_privs
        WHERE privilege IN (
          'CREATE SESSION',
          'EXEMPT ACCESS POLICY',
          'READ ANY TABLE',
          'SELECT ANY TABLE',
          'SELECT ANY DICTIONARY',
          'CREATE ANY CONTEXT',
          'EXECUTE ANY PROCEDURE'
        )
        """
    )
    effective = {row[0] for row in cursor.fetchall()}
    cursor.execute(
        """
        SELECT privilege, schema
        FROM session_schema_privs
        ORDER BY schema, privilege
        """
    )
    effective_schema = {(row[0], row[1]) for row in cursor.fetchall()}
    cursor.execute("SELECT role FROM session_roles")
    effective_roles = {row[0] for row in cursor.fetchall()}
    cursor.execute(
        """
        SELECT grantee, owner, table_name, privilege
        FROM all_tab_privs_recd
        WHERE grantee <> 'PUBLIC'
        ORDER BY grantee, owner, table_name, privilege
        """
    )
    effective_object_privileges = {
        (row[0], row[1], row[2], row[3]) for row in cursor.fetchall()
    }

    dangerous_effective = sorted(
        (effective - {"CREATE SESSION"})
        | (effective_roles & {"DBA", "ADB_DBA"})
    )
    broad_schema_reads = {
        item
        for item in effective_schema
        if item in {
            ("READ ANY TABLE", "ADMIN"),
            ("SELECT ANY TABLE", "ADMIN"),
        }
    }
    unexpected_effective_schema = sorted(
        f"{privilege} ON SCHEMA {schema}"
        for privilege, schema in effective_schema
        if (privilege, schema) not in broad_schema_reads
    )
    unexpected_effective_objects = _unexpected_runtime_object_privileges(
        effective_object_privileges
    )
    return [
        [
            "実行セッションの危険な有効権限",
            "危険" if dangerous_effective else "OK",
            ", ".join(dangerous_effective) or "なし",
        ],
        [
            "実行セッションのADMIN schema全表読取",
            "移行警告" if broad_schema_reads else "OK",
            (
                "有効。テーブル許可リストへの移行が必要です"
                if broad_schema_reads
                else "なし"
            ),
        ],
        [
            "実行セッションの想定外Schema権限",
            "危険" if unexpected_effective_schema else "OK",
            ", ".join(unexpected_effective_schema) or "なし",
        ],
        [
            "実行セッションの想定外オブジェクト権限",
            "危険" if unexpected_effective_objects else "OK",
            ", ".join(unexpected_effective_objects) or "なし",
        ],
    ]


def _runtime_pool_status_rows(vpd_pool) -> list[list[str]]:
    try:
        with vpd_pool.acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM DUAL")
                cursor.fetchone()
                rows = [["VPD実行プール", "OK", "接続確認成功"]]
                try:
                    rows.extend(_effective_runtime_privilege_rows(cursor))
                except Exception as exc:
                    rows.append(
                        [
                            "実行セッション権限監査",
                            "エラー",
                            _safe_database_error(exc, runtime=True),
                        ]
                    )
                return rows
    except Exception as exc:
        return [
            [
                "VPD実行プール",
                "エラー",
                _safe_database_error(exc, runtime=True),
            ]
        ]


_CONFIG_ERROR_STATES = frozenset(
    {"エラー", "非対応", "利用不可", "危険"}
)
_CONFIG_WARNING_STATES = frozenset(
    {
        "不足",
        "未作成",
        "未インストール",
        "更新必要",
        "未確認",
        "未設定",
        "移行警告",
        "PoC警告",
        "高リスク警告",
    }
)


def _configuration_summary(rows: list[list[str]]) -> str:
    states = {row[1] for row in rows}
    if states & _CONFIG_ERROR_STATES:
        return "❌ 状態確認で問題が見つかりました"
    if states & _CONFIG_WARNING_STATES:
        return "⚠️ 状態確認完了：対応が必要な項目があります"
    return "✅ 状態確認完了"


def _configuration_status_classes(status: str) -> list[str]:
    classes = ["operation-status"]
    if str(status).startswith("❌"):
        classes.append("operation-status--error")
    elif str(status).startswith("⚠️"):
        classes.append("operation-status--warning")
    elif str(status).startswith("✅"):
        classes.append("operation-status--success")
    elif str(status).startswith("⏳"):
        classes.append("operation-status--loading")
    return classes


def _style_configuration_frame(frame: pd.DataFrame):
    """Apply semantic status colors without changing the table values."""
    if frame.empty or "状態" not in frame.columns:
        return frame

    def style_row(row):
        state = str(row["状態"])
        if state in _CONFIG_ERROR_STATES:
            css = "background-color: #fff3f2; color: #8f1d18;"
        elif state in _CONFIG_WARNING_STATES:
            css = "background-color: #fff8e6; color: #6f4b00;"
        else:
            css = ""
        return [css] * len(row)

    return frame.style.apply(style_row, axis=1).set_properties(
        subset=["状態"],
        **{"font-weight": "600"},
    )


def _configuration_status(pool, vpd_pool, request) -> tuple[str, pd.DataFrame]:
    require_admin(request)
    rows: list[list[str]] = []
    try:
        users = normalize_vpd_login_users()
        rows.append(
            [
                "VPDログインユーザー",
                "OK" if users else "未使用",
                ", ".join(users),
            ]
        )
    except Exception as exc:
        users = ()
        rows.append(["VPDログインユーザー", "エラー", str(exc)])
    if users:
        try:
            validate_vpd_configuration()
            rows.append(["VPD必須設定", "OK", "共有パスワードと実行接続を設定済み"])
        except Exception as exc:
            rows.append(["VPD必須設定", "エラー", str(exc)])

    rows.append(
        [
            "ADMIN Webパスワード",
            "OK" if os.environ.get("APP_ADMIN_PASSWORD") else "移行警告",
            "個別設定済み"
            if os.environ.get("APP_ADMIN_PASSWORD")
            else "未設定のため、ローカル互換モードではADMIN DBパスワードを使用",
        ]
    )
    rows.append(
        [
            "VPD共有パスワード",
            "OK" if os.environ.get("APP_VPD_SHARED_PASSWORD") else "未設定",
            "設定済み" if os.environ.get("APP_VPD_SHARED_PASSWORD") else "",
        ]
    )
    if users:
        rows.append(
            [
                "VPD認証モデル",
                "PoC警告",
                (
                    "VPDログインユーザーは共有パスワードを使用します。"
                    "ユーザー間のなりすましを防ぐ本番認証ではありません"
                ),
            ]
        )

    try:
        parts = _runtime_parts()
        rows.append(["実行接続ユーザー", "OK", parts.username])
        rows.append(["実行接続DSN", "OK", parts.dsn])
    except Exception as exc:
        rows.append(["実行接続", "エラー" if users else "未設定", str(exc)])

    runtime_exists = False
    runtime_account_usable = False
    admin_check_complete = False
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                version = _require_26ai(cursor)
                rows.append(["Oracleバージョン", "OK", version])
                cursor.execute(
                    """
                    SELECT account_status
                    FROM dba_users
                    WHERE username = :username
                    """,
                    username=RUNTIME_USERNAME,
                )
                runtime_row = cursor.fetchone()
                runtime_exists = runtime_row is not None
                if runtime_exists:
                    account_status = str(runtime_row[0] or "UNKNOWN").upper()
                    runtime_account_usable = account_status.startswith("OPEN")
                    rows.append(
                        [
                            "実行ユーザー",
                            "OK" if runtime_account_usable else "利用不可",
                            f"{RUNTIME_USERNAME}: {account_status}",
                        ]
                    )
                else:
                    rows.append(
                        [
                            "実行ユーザー",
                            "未作成",
                            (
                                "「実行ユーザーの作成・確認」を"
                                "実行してください"
                            ),
                        ]
                    )

                cursor.execute(
                    """
                    SELECT object_type, status
                    FROM user_objects
                    WHERE object_name = 'SQL_ASSIST_CTX_PKG'
                      AND object_type IN ('PACKAGE', 'PACKAGE BODY')
                    ORDER BY object_type
                    """
                )
                context_objects = {
                    str(object_type): str(status)
                    for object_type, status in cursor.fetchall()
                }
                context_ready = (
                    context_objects.get("PACKAGE") == "VALID"
                    and context_objects.get("PACKAGE BODY") == "VALID"
                )
                if not context_objects:
                    rows.append(
                        [
                            "Application Contextパッケージ",
                            "未インストール",
                            (
                                "「Context・信頼済みパッケージの"
                                "インストール/更新」を実行してください"
                            ),
                        ]
                    )
                else:
                    context_detail = ", ".join(
                        f"{object_type}={status}"
                        for object_type, status in context_objects.items()
                    )
                    rows.append(
                        [
                            "Application Contextパッケージ",
                            "OK" if context_ready else "不足",
                            context_detail,
                        ]
                    )

                registry_exists = _registry_exists(cursor)
                registry_ready = (
                    registry_exists and _registry_schema_current(cursor)
                )
                registry_state = (
                    "OK"
                    if registry_ready
                    else ("更新必要" if registry_exists else "未インストール")
                )
                rows.append(
                    [
                        "データアクセスルール台帳",
                        registry_state,
                        (
                            f"ADMIN.{RULE_REGISTRY_TABLE}"
                            if registry_ready
                            else "「VPD基盤のインストール/更新」を実行してください"
                        ),
                    ]
                )
                can_grant_schema = _can_grant_schema_privileges(cursor)
                broad_schema_read = (
                    runtime_exists and _runtime_has_schema_read(cursor)
                )
                rows.append(
                    [
                        "Schema全表権限の移行",
                        (
                            "OK"
                            if can_grant_schema or not broad_schema_read
                            else "不足"
                        ),
                        (
                            "GRANT ANY SCHEMA PRIVILEGEまたは"
                            "GRANT ANY PRIVILEGEで失効可能"
                            if broad_schema_read
                            else "ADMIN schema全表読取は付与されていません"
                        ),
                    ]
                )
                can_administer_rls = _can_administer_admin_rls(cursor)
                rows.append(
                    [
                        "ADMIN schemaのRLS管理",
                        "OK" if can_administer_rls else "不足",
                        "ADMINISTER ROW LEVEL SECURITY POLICY",
                    ]
                )
                can_execute_rls = _can_execute_package(
                    cursor,
                    "SYS",
                    "DBMS_RLS",
                )
                rows.append(
                    [
                        "DBMS_RLS実行",
                        "OK" if can_execute_rls else "不足",
                        "EXECUTE ON SYS.DBMS_RLS",
                    ]
                )
                can_create_context = _has_system_privilege(
                    cursor,
                    "CREATE ANY CONTEXT",
                )
                rows.append(
                    [
                        "Application Context作成",
                        "OK" if can_create_context else "不足",
                        "CREATE ANY CONTEXT",
                    ]
                )
                admin_check_complete = True
                if runtime_exists:
                    try:
                        rows.extend(_direct_runtime_privilege_rows(cursor))
                    except Exception as exc:
                        rows.append(
                            [
                                "実行ユーザーの直接権限監査",
                                "エラー",
                                _safe_database_error(exc),
                            ]
                        )
    except UnsupportedOracleReleaseError as exc:
        rows.append(["Oracleバージョン", "非対応", str(exc)])
    except Exception as exc:
        rows.append(
            [
                "ADMINデータベース",
                "エラー",
                _safe_database_error(exc),
            ]
        )

    if not users:
        rows.append(
            [
                "VPD実行プール",
                "未使用",
                "VPDログインユーザーが設定されていません",
            ]
        )
    elif not admin_check_complete:
        rows.append(
            [
                "VPD実行プール",
                "未確認",
                "ADMINデータベースの確認に失敗したためスキップしました",
            ]
        )
    elif not runtime_exists:
        rows.append(
            [
                "VPD実行プール",
                "未作成",
                (
                    f"{RUNTIME_USERNAME}が未作成のため接続確認を"
                    "スキップしました"
                ),
            ]
        )
    elif not runtime_account_usable:
        rows.append(
            [
                "VPD実行プール",
                "利用不可",
                "実行ユーザーのアカウント状態を確認してください",
            ]
        )
    elif vpd_pool is None:
        rows.append(["VPD実行プール", "エラー", "実行プールが未作成です"])
    else:
        rows.extend(_runtime_pool_status_rows(vpd_pool))

    for warning in password_risk_warnings():
        rows.append(["パスワード再利用", "高リスク警告", warning])

    frame = pd.DataFrame(rows, columns=["確認項目", "状態", "詳細"])
    return _configuration_summary(rows), frame


def _configuration_status_stream(pool, vpd_pool, request):
    require_admin(request)
    yield "⏳ 状態確認中...", None
    try:
        yield _configuration_status(pool, vpd_pool, request)
    except Exception as exc:
        detail = _safe_database_error(exc)
        frame = pd.DataFrame(
            [["状態確認", "エラー", detail]],
            columns=["確認項目", "状態", "詳細"],
        )
        yield f"❌ {detail}", frame


def _create_runtime_account(pool, vpd_pool, request) -> str:
    require_admin(request)
    parts = _runtime_parts()
    # A double quote is not accepted because this DDL must never become ambiguous.
    if '"' in parts.password or "\n" in parts.password or "\r" in parts.password:
        raise ValueError("実行ユーザーのパスワードに二重引用符や改行は使用できません")
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            _require_26ai(cursor)
            cursor.execute(
                "SELECT COUNT(*) FROM all_users WHERE username = :username",
                username=RUNTIME_USERNAME,
            )
            exists = int(cursor.fetchone()[0]) > 0
            if not exists:
                cursor.execute(
                    f'CREATE USER {RUNTIME_USERNAME} IDENTIFIED BY "{parts.password}"'
                )
            cursor.execute(f"GRANT CREATE SESSION TO {RUNTIME_USERNAME}")
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM user_objects
                WHERE object_name = 'SQL_ASSIST_CTX_PKG'
                  AND object_type = 'PACKAGE'
                """
            )
            package_exists = int(cursor.fetchone()[0]) > 0
            if package_exists:
                cursor.execute(
                    f"GRANT EXECUTE ON {CONTEXT_PACKAGE} TO {RUNTIME_USERNAME}"
                )
        conn.commit()
    if vpd_pool is not None:
        vpd_pool.reset()
    if exists:
        return (
            "✅ ユーザーは既に存在するためパスワードを変更していません。"
            "CREATE SESSIONを再確認しました。既存のschema全表読取権限は"
            "自動変更していません。許可リスト移行で明示的に失効してください。"
        )
    return (
        "✅ SQL_ASSIST_RUNTIMEを作成し、CREATE SESSIONのみを付与しました。"
        "テーブルREADはデータアクセスルールから付与してください"
    )


def _install_context(pool, request) -> str:
    require_admin(request)
    parsed = parse_oracle_script(CONTEXT_INSTALL_SCRIPT)
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            _require_26ai(cursor)
            if not _has_system_privilege(cursor, "CREATE ANY CONTEXT"):
                raise RuntimeError(
                    "ADMINセッションにCREATE ANY CONTEXTがありません"
                )
            for statement in parsed:
                cursor.execute(statement.text)
                program = created_program(statement.text)
                if program:
                    errors = _compile_errors(cursor, *program)
                    if errors:
                        raise RuntimeError("\n".join(errors))
            _ensure_rule_registry(cursor)
            cursor.execute(
                "SELECT COUNT(*) FROM all_users WHERE username = :username",
                username=RUNTIME_USERNAME,
            )
            runtime_exists = int(cursor.fetchone()[0]) > 0
            if runtime_exists:
                cursor.execute(
                    f"GRANT EXECUTE ON {CONTEXT_PACKAGE} TO {RUNTIME_USERNAME}"
                )
        conn.commit()
    if runtime_exists:
        return (
            "✅ SQL_ASSIST_CTX、信頼済みパッケージ、アクセスルール台帳を"
            "インストールし、実行ユーザーにEXECUTE権限を付与しました"
        )
    return (
        "✅ SQL_ASSIST_CTX、信頼済みパッケージ、アクセスルール台帳を"
        "インストールしました。実行ユーザー作成時にEXECUTE権限を追加します"
    )


def _policy_rows(cursor, object_name: str | None = None) -> list[tuple]:
    sql = """
        SELECT p.object_name, p.policy_name, p.pf_owner,
               NVL(p.package, '-'), p.function,
               p.sel, p.enable, p.policy_type,
               NVL(a.namespace, '-'), NVL(a.attribute, '-'),
               p.policy_group
        FROM all_policies p
        LEFT JOIN all_policy_attributes a
          ON a.object_owner = p.object_owner
         AND a.object_name = p.object_name
         AND a.policy_group = p.policy_group
         AND a.policy_name = p.policy_name
        WHERE p.object_owner = 'ADMIN'
    """
    binds = {}
    if object_name is not None:
        sql += " AND p.object_name = :object_name"
        binds["object_name"] = _object_name(object_name)
    sql += " ORDER BY p.object_name, p.policy_group, p.policy_name"
    cursor.execute(sql, **binds)
    return [tuple(row) for row in cursor.fetchall()]


def _policy_matches_managed_rule(policy: tuple, registered_rule) -> bool:
    """Return whether metadata is the default-group policy owned by the registry."""
    registered_rule = _managed_rule(registered_rule)
    return bool(
        registered_rule
        and registered_rule.policy_name
        and str(policy[10]).upper() == "SYS_DEFAULT"
        and str(policy[1]) == registered_rule.policy_name
    )


def _unmanaged_policy_names(cursor, object_name: str, registered_rule) -> set[str]:
    return {
        str(row[1])
        for row in _policy_rows(cursor, object_name)
        if str(row[5]).upper() == "YES"
        and not _policy_matches_managed_rule(row, registered_rule)
    }


def _require_safe_relation_source(cursor, relation_object: str) -> None:
    relation_object = _require_relation_table(cursor, relation_object)
    registered = _managed_rule(_registry_rule(cursor, relation_object))
    conflicts = _unmanaged_policy_names(
        cursor,
        relation_object,
        registered,
    )
    if conflicts:
        raise RuntimeError(
            f"関連テーブル{relation_object}に未登録SELECT Policyがあります: "
            + ", ".join(sorted(conflicts))
        )
    if registered and registered.access_mode == ACCESS_RELATION_MATCH:
        policy = (
            _policy_row(cursor, relation_object, registered.policy_name)
            if registered.policy_name
            else None
        )
        if policy and str(policy[6]).upper() == "YES":
            raise RuntimeError(
                "関連テーブル自身に関連テーブル判定ルールがあるため、"
                "多段の関連ルールは設定できません"
            )


def _policy_row(cursor, object_name: str, policy_name: str):
    policy_name = _managed_name(policy_name, "Policy名")
    return next(
        (
            row
            for row in _policy_rows(cursor, object_name)
            if str(row[1]) == policy_name
            and str(row[10]).upper() == "SYS_DEFAULT"
        ),
        None,
    )


def _add_disabled_policy(
    cursor,
    object_name: str,
    policy_name: str,
    function_name: str,
) -> None:
    cursor.execute(
        """
        BEGIN
          DBMS_RLS.ADD_POLICY(
            object_schema   => 'ADMIN',
            object_name     => :object_name,
            policy_name     => :policy_name,
            function_schema => 'ADMIN',
            policy_function => :function_name,
            statement_types => 'SELECT',
            enable          => FALSE,
            policy_type     => DBMS_RLS.CONTEXT_SENSITIVE,
            namespace       => 'SQL_ASSIST_CTX',
            attribute       => 'LOGIN_USER'
          );
        END;
        """,
        object_name=_object_name(object_name),
        policy_name=_managed_name(policy_name, "Policy名"),
        function_name=_managed_name(function_name, "Function名"),
    )


def _set_policy_enabled(
    cursor,
    object_name: str,
    policy_name: str,
    enabled: bool,
) -> None:
    cursor.execute(
        """
        BEGIN
          DBMS_RLS.ENABLE_POLICY(
            object_schema => 'ADMIN',
            object_name   => :object_name,
            policy_name   => :policy_name,
            enable        => :enabled
          );
        END;
        """,
        object_name=_object_name(object_name),
        policy_name=_managed_name(policy_name, "Policy名"),
        enabled=enabled,
    )


def _refresh_policy(cursor, object_name: str, policy_name: str) -> None:
    cursor.execute(
        """
        BEGIN
          DBMS_RLS.REFRESH_POLICY(
            object_schema => 'ADMIN',
            object_name   => :object_name,
            policy_name   => :policy_name
          );
        END;
        """,
        object_name=_object_name(object_name),
        policy_name=_managed_name(policy_name, "Policy名"),
    )


def _drop_policy(cursor, object_name: str, policy_name: str) -> None:
    if _policy_row(cursor, object_name, policy_name) is None:
        return
    cursor.execute(
        """
        BEGIN
          DBMS_RLS.DROP_POLICY(
            object_schema => 'ADMIN',
            object_name   => :object_name,
            policy_name   => :policy_name
          );
        END;
        """,
        object_name=_object_name(object_name),
        policy_name=_managed_name(policy_name, "Policy名"),
    )


def _drop_unmanaged_policy(
    cursor,
    object_name: str,
    policy_group: str,
    policy_name: str,
) -> None:
    """Drop one policy selected from current ALL_POLICIES metadata."""
    object_name = _object_name(object_name)
    if policy_group.upper() == "SYS_DEFAULT":
        cursor.execute(
            """
            BEGIN
              DBMS_RLS.DROP_POLICY(
                object_schema => 'ADMIN',
                object_name   => :object_name,
                policy_name   => :policy_name
              );
            END;
            """,
            object_name=object_name,
            policy_name=policy_name,
        )
        return
    cursor.execute(
        """
        BEGIN
          DBMS_RLS.DROP_GROUPED_POLICY(
            object_schema => 'ADMIN',
            object_name   => :object_name,
            policy_group  => :policy_group,
            policy_name   => :policy_name
          );
        END;
        """,
        object_name=object_name,
        policy_group=policy_group,
        policy_name=policy_name,
    )


def _policy_function_reference_count(
    cursor,
    function_owner: str,
    function_name: str,
) -> int:
    """Count policies in every accessible object that use a standalone function."""
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM all_policies
        WHERE pf_owner = :function_owner
          AND package IS NULL
          AND function = :function_name
        """,
        function_owner=function_owner,
        function_name=function_name,
    )
    return int(cursor.fetchone()[0])


def _drop_function(cursor, function_name: str | None) -> None:
    if not function_name:
        return
    function_name = _managed_name(function_name, "Function名")
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM user_objects
        WHERE object_name = :function_name
          AND object_type = 'FUNCTION'
        """,
        function_name=function_name,
    )
    if int(cursor.fetchone()[0]) > 0:
        cursor.execute(f"DROP FUNCTION ADMIN.{function_name}")


def _grant_runtime_read(cursor, object_name: str) -> None:
    object_name = _object_name(object_name)
    quoted_object = _quoted_identifier(object_name)
    cursor.execute(
        f"GRANT READ ON ADMIN.{quoted_object} TO {RUNTIME_USERNAME}"
    )
    privileges = _runtime_object_read_privileges(cursor, object_name).get(
        object_name, set()
    )
    if "SELECT" in privileges:
        cursor.execute(
            f"REVOKE SELECT ON ADMIN.{quoted_object} FROM {RUNTIME_USERNAME}"
        )


def _revoke_runtime_read(cursor, object_name: str) -> None:
    object_name = _object_name(object_name)
    quoted_object = _quoted_identifier(object_name)
    privileges = _runtime_object_read_privileges(cursor, object_name).get(
        object_name, set()
    )
    for privilege in sorted(privileges):
        cursor.execute(
            f"REVOKE {privilege} ON ADMIN.{quoted_object} "
            f"FROM {RUNTIME_USERNAME}"
        )


def _rule_validation_token(
    object_name: str,
    access_mode: str,
    match_column: str | None,
    match_data_type: str | None = None,
    relation_spec: RelationSpec | None = None,
) -> str:
    object_name = _object_name(object_name)
    access_mode = str(access_mode or "").upper()
    if access_mode == ACCESS_ROW_MATCH:
        values = (
            _column_name(match_column),
            _match_data_type(match_data_type),
        )
    elif access_mode == ACCESS_RELATION_MATCH and relation_spec is not None:
        values = (
            relation_spec.relation_object,
            relation_spec.target_column,
            relation_spec.target_data_type,
            relation_spec.object_column,
            relation_spec.object_data_type,
            relation_spec.match_column,
            relation_spec.match_data_type,
        )
    else:
        values = ("-",)
    return hashlib.sha256(
        "|".join((object_name, access_mode, *values)).encode("ascii")
    ).hexdigest()


def _preview_access_rule(
    pool,
    object_name: str,
    access_mode: str,
    match_column: str | None,
    request,
    relation_target_column: str | None = None,
    relation_object: str | None = None,
    relation_object_column: str | None = None,
    relation_match_column: str | None = None,
) -> tuple[str, str, str]:
    require_admin(request)
    object_name = _object_name(object_name)
    access_mode = str(access_mode or "").upper()
    if access_mode not in {
        ACCESS_NONE,
        ACCESS_FULL,
        ACCESS_ROW_MATCH,
        ACCESS_RELATION_MATCH,
    }:
        raise ValueError("アクセスモードを選択してください")
    match_data_type = None
    relation_spec = None
    index_warning = ""
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            _require_26ai(cursor)
            _require_registry_current(cursor)
            object_type = _require_business_object(cursor, object_name)
            registered = _registry_rule(cursor, object_name)
            conflicts = _unmanaged_policy_names(
                cursor, object_name, registered
            )
            if conflicts and access_mode != ACCESS_NONE:
                raise RuntimeError(
                    "未登録Policyが競合しています: "
                    + ", ".join(sorted(conflicts))
                )
            if access_mode == ACCESS_ROW_MATCH:
                match_column, match_data_type = _require_match_column(
                    cursor, object_name, match_column
                )
                if match_data_type == _NUMBER_TYPE:
                    _require_unambiguous_number_login_users(
                        normalize_vpd_login_users()
                    )
            elif access_mode == ACCESS_RELATION_MATCH:
                match_column = None
                relation_spec = _require_relation_spec(
                    cursor,
                    object_name,
                    relation_target_column,
                    relation_object,
                    relation_object_column,
                    relation_match_column,
                )
                _require_safe_relation_source(
                    cursor,
                    relation_spec.relation_object,
                )
                if relation_spec.match_data_type == _NUMBER_TYPE:
                    _require_unambiguous_number_login_users(
                        normalize_vpd_login_users()
                    )
                if not _relation_login_column_has_index(
                    cursor,
                    relation_spec.relation_object,
                    relation_spec.match_column,
                ):
                    index_warning = (
                        " 関連テーブルのログインユーザー照合列を先頭に持つ"
                        "有効な索引がないため、性能を確認してください。"
                    )
            else:
                match_column = None
            broad_access = _runtime_has_schema_read(cursor)
    prefix = "⚠️" if broad_access or index_warning else "✅"
    migration_note = (
        " schema全表読取が残っているため、許可リスト移行が完了するまで"
        f"「{ACCESS_MODE_LABELS[ACCESS_NONE]}」は実効化できません。"
        if broad_access
        else ""
    )
    status = (
        f"{prefix} {object_type} {object_name}: "
        f"{ACCESS_MODE_LABELS[access_mode]} を検証しました。"
        f"{migration_note}{index_warning}"
    )
    return (
        status,
        _rule_sql_preview(
            object_name,
            access_mode,
            match_column,
            match_data_type,
            relation_spec,
        ),
        _rule_validation_token(
            object_name,
            access_mode,
            match_column,
            match_data_type,
            relation_spec,
        ),
    )


def _apply_access_rule(
    pool,
    vpd_pool,
    object_name: str,
    access_mode: str,
    match_column: str | None,
    validation_token: str,
    request,
    relation_target_column: str | None = None,
    relation_object: str | None = None,
    relation_object_column: str | None = None,
    relation_match_column: str | None = None,
) -> str:
    updated_by = require_admin(request)
    object_name = _object_name(object_name)
    access_mode = str(access_mode or "").upper()
    if access_mode not in {
        ACCESS_NONE,
        ACCESS_FULL,
        ACCESS_ROW_MATCH,
        ACCESS_RELATION_MATCH,
    }:
        raise ValueError("アクセスモードが不正です")
    if access_mode in {ACCESS_NONE, ACCESS_FULL}:
        expected_token = _rule_validation_token(
            object_name,
            access_mode,
            match_column,
        )
        if not validation_token or validation_token != expected_token:
            raise ValueError(
                "設定内容を変更したため、SQLプレビューを再実行してください"
            )

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            _require_26ai(cursor)
            _require_rls_admin_capability(cursor)
            _require_registry_current(cursor)
            object_type = _require_business_object(cursor, object_name)
            normalized_column = None
            match_data_type = None
            relation_spec = None
            if access_mode == ACCESS_ROW_MATCH:
                normalized_column, match_data_type = _require_match_column(
                    cursor, object_name, match_column
                )
                users = normalize_vpd_login_users()
                if match_data_type == _NUMBER_TYPE:
                    _require_unambiguous_number_login_users(users)
                expected_token = _rule_validation_token(
                    object_name,
                    access_mode,
                    normalized_column,
                    match_data_type,
                )
                if not validation_token or validation_token != expected_token:
                    raise ValueError(
                        "設定内容を変更したため、SQLプレビューを"
                        "再実行してください"
                    )
            elif access_mode == ACCESS_RELATION_MATCH:
                relation_spec = _require_relation_spec(
                    cursor,
                    object_name,
                    relation_target_column,
                    relation_object,
                    relation_object_column,
                    relation_match_column,
                )
                _require_safe_relation_source(
                    cursor,
                    relation_spec.relation_object,
                )
                users = normalize_vpd_login_users()
                if relation_spec.match_data_type == _NUMBER_TYPE:
                    _require_unambiguous_number_login_users(users)
                expected_token = _rule_validation_token(
                    object_name,
                    access_mode,
                    None,
                    None,
                    relation_spec,
                )
                if not validation_token or validation_token != expected_token:
                    raise ValueError(
                        "設定内容を変更したため、SQLプレビューを"
                        "再実行してください"
                    )
            registered = _managed_rule(_registry_rule(cursor, object_name))
            conflicts = _unmanaged_policy_names(
                cursor, object_name, registered
            )
            if conflicts and access_mode != ACCESS_NONE:
                raise RuntimeError(
                    "未登録Policyが競合しています: "
                    + ", ".join(sorted(conflicts))
                )

            if access_mode == ACCESS_NONE:
                if _runtime_has_schema_read(cursor):
                    raise RuntimeError(
                        "schema全表読取が残っているため、このオブジェクトだけを"
                        "アクセス不可にはできません。先に必要なルールを準備し、"
                        "テーブル許可リスト移行を実行してください"
                    )
                _revoke_runtime_read(cursor, object_name)
                if (
                    registered
                    and registered.access_mode in _FILTERED_ACCESS_MODES
                    and registered.policy_name
                ):
                    _set_policy_enabled(
                        cursor,
                        object_name,
                        registered.policy_name,
                        False,
                    )
                if registered:
                    _set_registry_state(cursor, object_name, "STAGED")
                conn.commit()
                if vpd_pool is not None:
                    vpd_pool.reset()
                return (
                    f"✅ {object_name} を"
                    f"「{ACCESS_MODE_LABELS[ACCESS_NONE]}」に設定しました"
                )

            policy_name = None
            function_name = None
            if access_mode in _FILTERED_ACCESS_MODES:
                policy_name, function_name = _rule_names(object_name)
            staged_policy_name = (
                policy_name
                if policy_name is not None
                else (registered.policy_name if registered else None)
            )
            staged_function_name = (
                function_name
                if function_name is not None
                else (registered.function_name if registered else None)
            )

            try:
                _revoke_runtime_read(cursor, object_name)
                _upsert_registry_rule(
                    cursor,
                    object_name=object_name,
                    object_type=object_type,
                    access_mode=access_mode,
                    match_column=normalized_column,
                    relation_object=(
                        relation_spec.relation_object
                        if relation_spec
                        else None
                    ),
                    relation_target_column=(
                        relation_spec.target_column
                        if relation_spec
                        else None
                    ),
                    relation_object_column=(
                        relation_spec.object_column
                        if relation_spec
                        else None
                    ),
                    relation_match_column=(
                        relation_spec.match_column
                        if relation_spec
                        else None
                    ),
                    policy_name=staged_policy_name,
                    function_name=staged_function_name,
                    state="STAGED",
                    updated_by=updated_by,
                )
                conn.commit()

                if access_mode in _FILTERED_ACCESS_MODES:
                    cursor.execute(
                        _policy_function_ddl(
                            object_name,
                            normalized_column,
                            match_data_type,
                            access_mode=access_mode,
                            relation_spec=relation_spec,
                        )
                    )
                    errors = _compile_errors(
                        cursor, "FUNCTION", function_name
                    )
                    if errors:
                        raise RuntimeError(
                            "Oracleコンパイルエラー:\n" + "\n".join(errors)
                        )
                    existing_policy = _policy_row(
                        cursor, object_name, policy_name
                    )
                    if existing_policy is None:
                        _add_disabled_policy(
                            cursor,
                            object_name,
                            policy_name,
                            function_name,
                        )
                    else:
                        _refresh_policy(cursor, object_name, policy_name)
                    _set_policy_enabled(
                        cursor, object_name, policy_name, True
                    )
                    _grant_runtime_read(cursor, object_name)
                else:
                    if registered and registered.policy_name:
                        _drop_policy(
                            cursor,
                            object_name,
                            registered.policy_name,
                        )
                    if registered and registered.function_name:
                        _drop_function(cursor, registered.function_name)
                    _grant_runtime_read(cursor, object_name)

                _upsert_registry_rule(
                    cursor,
                    object_name=object_name,
                    object_type=object_type,
                    access_mode=access_mode,
                    match_column=normalized_column,
                    relation_object=(
                        relation_spec.relation_object
                        if relation_spec
                        else None
                    ),
                    relation_target_column=(
                        relation_spec.target_column
                        if relation_spec
                        else None
                    ),
                    relation_object_column=(
                        relation_spec.object_column
                        if relation_spec
                        else None
                    ),
                    relation_match_column=(
                        relation_spec.match_column
                        if relation_spec
                        else None
                    ),
                    policy_name=policy_name,
                    function_name=function_name,
                    state="ACTIVE",
                    updated_by=updated_by,
                )
                conn.commit()
            except Exception:
                try:
                    _set_registry_state(cursor, object_name, "ERROR")
                    conn.commit()
                except Exception:
                    pass
                raise

    if vpd_pool is not None:
        vpd_pool.reset()
    return (
        f"✅ {object_name} を「{ACCESS_MODE_LABELS[access_mode]}」に"
        "設定しました"
    )


def _require_registered_rule(cursor, object_name: str):
    object_name = _object_name(object_name)
    registered = _registry_rule(cursor, object_name)
    if registered is None:
        raise ValueError("このオブジェクトには管理対象ルールがありません")
    conflicts = _unmanaged_policy_names(cursor, object_name, registered)
    if conflicts:
        raise RuntimeError(
            "未登録Policyが競合しています: " + ", ".join(sorted(conflicts))
        )
    return _managed_rule(registered)


def _delete_access_rule(
    pool,
    vpd_pool,
    object_name: str,
    confirmed: bool,
    request,
) -> str:
    require_admin(request)
    if not str(object_name or "").strip():
        raise ValueError("削除するアクセスルールを選択してください")
    if not confirmed:
        raise ValueError("削除の影響を確認するチェックが必要です")
    object_name = _object_name(object_name)
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            _require_26ai(cursor)
            _require_rls_admin_capability(cursor)
            if _runtime_has_schema_read(cursor):
                raise RuntimeError(
                    "schema全表読取が残っている間は削除できません。"
                    "先にテーブル許可リストを有効化してください"
                )
            registered = _require_registered_rule(cursor, object_name)
            try:
                _revoke_runtime_read(cursor, object_name)
                _set_registry_state(cursor, object_name, "STAGED")
                conn.commit()
                if registered.policy_name:
                    _drop_policy(
                        cursor,
                        object_name,
                        registered.policy_name,
                    )
                if registered.function_name:
                    _drop_function(cursor, registered.function_name)
                _delete_registry_rule(cursor, object_name)
                conn.commit()
            except Exception:
                try:
                    _set_registry_state(cursor, object_name, "ERROR")
                    conn.commit()
                except Exception:
                    pass
                raise
    if vpd_pool is not None:
        vpd_pool.reset()
    return f"✅ {object_name} のアクセスルールを削除しました"


def _require_unmanaged_select_policy(
    cursor,
    target: DeleteTarget,
) -> tuple:
    """Revalidate that a selected external policy still exists and is unmanaged."""
    _require_business_object(cursor, target.object_name)
    matching = {
        tuple(row)
        for row in _policy_rows(cursor, target.object_name)
        if str(row[1]) == target.policy_name
        and str(row[10]) == target.policy_group
        and str(row[5]).upper() == "YES"
    }
    if not matching:
        raise ValueError(
            "選択した未登録SELECT Policyは既に存在しません。"
            "対象・現在状態を更新してください"
        )
    policy = next(iter(matching))
    registered = _managed_rule(
        _registry_rule(cursor, target.object_name)
    )
    if _policy_matches_managed_rule(policy, registered):
        raise ValueError(
            "選択したPolicyは管理対象に変わりました。"
            "対象・現在状態を更新してください"
        )
    return policy


def _policy_function_display(policy: tuple) -> str:
    owner = str(policy[2])
    package_name = str(policy[3])
    function_name = str(policy[4])
    if package_name == "-":
        return f"{owner}.{function_name}"
    return f"{owner}.{package_name}.{function_name}"


def _orphan_generated_function(
    cursor,
    object_name: str,
    policy: tuple,
) -> str | None:
    """Return a generated standalone function only when no policy still uses it."""
    function_owner = str(policy[2])
    package_name = str(policy[3])
    function_name = str(policy[4])
    expected_function = _rule_names(object_name)[1]
    if (
        function_owner != "ADMIN"
        or package_name != "-"
        or function_name != expected_function
        or not _GENERATED_FUNCTION_RE.fullmatch(function_name)
    ):
        return None
    if (
        _policy_function_reference_count(
            cursor,
            function_owner,
            function_name,
        )
        != 0
    ):
        return None
    return function_name


def _delete_unmanaged_policy(
    pool,
    vpd_pool,
    target: DeleteTarget,
    confirmed: bool,
    request,
) -> str:
    """Fail closed while removing a selected unmanaged SELECT policy."""
    require_admin(request)
    if (
        target.kind != DELETE_TARGET_UNMANAGED
        or target.policy_group is None
        or target.policy_name is None
    ):
        raise ValueError("未登録Policyの削除対象が不正です")
    if not confirmed:
        raise ValueError("削除の影響を確認するチェックが必要です")

    read_revoked = False
    policy_dropped = False
    function_deleted = None
    function_display = ""
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                _require_26ai(cursor)
                _require_rls_admin_capability(cursor)
                if _runtime_has_schema_read(cursor):
                    raise RuntimeError(
                        "schema全表読取が残っている間は削除できません。"
                        "先にテーブル許可リストを有効化してください"
                    )
                policy = _require_unmanaged_select_policy(cursor, target)
                function_display = _policy_function_display(policy)
                _revoke_runtime_read(cursor, target.object_name)
                conn.commit()
                read_revoked = True

                _drop_unmanaged_policy(
                    cursor,
                    target.object_name,
                    target.policy_group,
                    target.policy_name,
                )
                policy_dropped = True
                orphan_function = _orphan_generated_function(
                    cursor,
                    target.object_name,
                    policy,
                )
                if orphan_function:
                    _drop_function(cursor, orphan_function)
                    function_deleted = orphan_function
                conn.commit()
    except Exception as exc:
        detail = _safe_database_error(exc)
        if policy_dropped:
            raise RuntimeError(
                "未登録Policyは削除しましたが、後処理に失敗しました。"
                f"SQL_ASSIST_RUNTIMEのREAD／SELECTは失効済みです: {detail}"
            ) from exc
        if read_revoked:
            raise RuntimeError(
                "未登録Policyの削除に失敗しました。"
                f"SQL_ASSIST_RUNTIMEのREAD／SELECTは失効済みです: {detail}"
            ) from exc
        raise
    finally:
        if vpd_pool is not None and (read_revoked or policy_dropped):
            vpd_pool.reset()

    function_result = (
        f"孤児Function {function_deleted} も削除しました"
        if function_deleted
        else f"関連Function {function_display} は保持しました"
    )
    return (
        f"✅ {target.object_name} の未登録Policy "
        f"{target.policy_name}（{target.policy_group}）を削除しました。"
        f"{function_result}"
    )


def _delete_target(
    pool,
    vpd_pool,
    target_value: str,
    confirmed: bool,
    request,
) -> str:
    """Dispatch a typed deletion request after decoding its opaque UI value."""
    target = _decode_delete_target(target_value)
    if target.kind == DELETE_TARGET_MANAGED:
        return _delete_access_rule(
            pool,
            vpd_pool,
            target.object_name,
            confirmed,
            request,
        )
    return _delete_unmanaged_policy(
        pool,
        vpd_pool,
        target,
        confirmed,
        request,
    )


def _activate_table_allowlist(
    pool,
    vpd_pool,
    confirmed: bool,
    request,
) -> str:
    require_admin(request)
    if not confirmed:
        raise ValueError("移行の影響を確認するチェックが必要です")
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            _require_26ai(cursor)
            _require_schema_grant_capability(cursor)
            _require_rls_admin_capability(cursor)
            if not _runtime_has_schema_read(cursor):
                return "✅ テーブル許可リストは既に有効です"
            rules = {
                name: row
                for name, row in _registry_rules(cursor).items()
                if row.state.upper() == "ACTIVE"
            }
            if not rules:
                raise RuntimeError(
                    "ACTIVEなアクセスルールがありません。"
                    "少なくとも1件のルールを適用してください"
                )
            for object_name, registered in rules.items():
                _require_business_object(cursor, object_name)
                conflicts = _unmanaged_policy_names(
                    cursor, object_name, registered
                )
                if conflicts:
                    raise RuntimeError(
                        f"{object_name}に未登録Policyがあります: "
                        + ", ".join(sorted(conflicts))
                    )
                if registered.access_mode in _FILTERED_ACCESS_MODES:
                    _require_valid_policy_function(
                        cursor,
                        registered.function_name,
                    )
                    policy = _policy_row(
                        cursor,
                        object_name,
                        registered.policy_name,
                    )
                    if policy is None or str(policy[6]).upper() != "YES":
                        raise RuntimeError(
                            f"{object_name}の管理対象Policyが有効ではありません"
                        )
                _grant_runtime_read(cursor, object_name)

            cursor.execute(
                """
                SELECT privilege
                FROM dba_schema_privs
                WHERE grantee = :username
                  AND schema = 'ADMIN'
                  AND privilege IN ('READ ANY TABLE', 'SELECT ANY TABLE')
                ORDER BY privilege
                """,
                username=RUNTIME_USERNAME,
            )
            for (privilege,) in cursor.fetchall():
                privilege = str(privilege).upper()
                if privilege not in {"READ ANY TABLE", "SELECT ANY TABLE"}:
                    raise RuntimeError("予期しないSchema権限を検出しました")
                cursor.execute(
                    f"REVOKE {privilege} ON SCHEMA ADMIN "
                    f"FROM {RUNTIME_USERNAME}"
                )
        conn.commit()
    if vpd_pool is not None:
        vpd_pool.reset()
    return (
        "✅ ADMIN schema全表読取を失効し、テーブル許可リストを"
        "有効化しました"
    )


def _policy_inventory(pool, request) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_admin(request)
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            _require_26ai(cursor)
            registered = _registry_rules(cursor)
            objects = _business_objects(cursor)
            business_names = {name for name, _object_type in objects}
            policies = [
                row
                for row in _policy_rows(cursor)
                if str(row[0]) in business_names
                and str(row[5]).upper() == "YES"
            ]
            direct_reads = _runtime_object_read_privileges(cursor)
            broad_read = _runtime_has_schema_read(cursor)
            relation_errors: dict[str, str] = {}
            for object_name, rule in registered.items():
                if rule.access_mode != ACCESS_RELATION_MATCH:
                    continue
                try:
                    _require_relation_spec(
                        cursor,
                        object_name,
                        rule.relation_target_column,
                        rule.relation_object,
                        rule.relation_object_column,
                        rule.relation_match_column,
                    )
                    _require_safe_relation_source(
                        cursor,
                        rule.relation_object,
                    )
                except Exception as exc:
                    relation_errors[object_name] = str(exc)

    policies_by_object: dict[str, list[tuple]] = {}
    for policy in policies:
        policies_by_object.setdefault(str(policy[0]), []).append(policy)

    policy_rows: list[list[str]] = []
    for policy in policies:
        object_name = str(policy[0])
        rule = _managed_rule(registered.get(object_name))
        managed = _policy_matches_managed_rule(policy, rule)
        policy_rows.append(
            [
                object_name,
                str(policy[1]),
                str(policy[10]),
                _policy_function_display(policy),
                str(policy[6]),
                str(policy[7]),
                "管理対象" if managed else "参照のみ・競合",
                (
                    "-"
                    if managed
                    else "未登録Policyです。高度な操作から削除できます"
                ),
            ]
        )

    object_rows: list[list[str]] = []
    for object_name, object_type in objects:
        rule = _managed_rule(registered.get(object_name))
        object_policies = policies_by_object.get(object_name, [])
        unmanaged = {
            str(policy[1])
            for policy in object_policies
            if not _policy_matches_managed_rule(policy, rule)
        }
        direct_read = bool(direct_reads.get(object_name))
        condition = "-"
        management_state = "-"
        exception = "-"

        if unmanaged:
            access_label = "競合"
            management_state = f"競合（{len(unmanaged)}件）"
            exception = (
                "未登録Policyがあり、Oracleでは他PolicyとAND結合されます。"
                "高度な操作から個別に削除できます"
            )
        elif rule:
            mode = rule.access_mode
            if mode == ACCESS_ROW_MATCH:
                condition = f"{rule.match_column}=LOGIN_USER"
            elif mode == ACCESS_RELATION_MATCH:
                condition = (
                    f"{rule.relation_target_column} ← "
                    f"{rule.relation_object}.{rule.relation_object_column} / "
                    f"{rule.relation_match_column}=LOGIN_USER"
                )
            management_state = rule.state
            if mode == ACCESS_FULL:
                access_label = (
                    ACCESS_MODE_LABELS[ACCESS_FULL]
                    if broad_read or direct_read
                    else ACCESS_MODE_LABELS[ACCESS_NONE]
                )
            else:
                policy = next(
                    (
                        item
                        for item in object_policies
                        if _policy_matches_managed_rule(item, rule)
                    ),
                    None,
                )
                policy_enabled = bool(
                    policy and str(policy[6]).upper() == "YES"
                )
                if (broad_read or direct_read) and policy_enabled:
                    access_label = ACCESS_MODE_LABELS.get(mode, mode)
                else:
                    access_label = ACCESS_MODE_LABELS[ACCESS_NONE]
                    if management_state == "ACTIVE":
                        management_state = "ドリフト"
                        exception = (
                            "ACTIVE登録と実際のREAD/Policy状態が一致しません"
                        )
            if object_name in relation_errors:
                management_state = "ドリフト"
                exception = relation_errors[object_name]
        elif broad_read:
            access_label = "未管理・全行参照（移行前）"
            management_state = "移行必要"
            exception = "許可リスト移行後はアクセス不可になります"
        elif direct_read:
            access_label = "未登録のREAD付与"
            management_state = "ドリフト"
            exception = "台帳にない直接READ/SELECT権限があります"
        else:
            access_label = ACCESS_MODE_LABELS[ACCESS_NONE]

        object_rows.append(
            [
                object_name,
                object_type,
                access_label,
                condition,
                management_state,
                exception,
            ]
        )

    policy_df = pd.DataFrame(
        policy_rows,
        columns=[
            "オブジェクト",
            "Policy名",
            "Policy Group",
            "Function",
            "有効",
            "Policy Type",
            "操作範囲",
            "例外",
        ],
    )
    object_df = pd.DataFrame(
        object_rows,
        columns=[
            "オブジェクト",
            "種別",
            "アクセス",
            "判定条件",
            "有効状態",
            "例外",
        ],
    )
    return policy_df, object_df


def _safe_status(callback, *args):
    try:
        return callback(*args)
    except Exception as exc:
        return f"❌ {exc}"


def _execution_test_empty_updates(status: str):
    """Return visible feedback while clearing any previous test result."""
    return (
        gr.Markdown(
            value=status,
            visible=True,
            elem_classes=_configuration_status_classes(status),
        ),
        gr.Dataframe(
            value=pd.DataFrame(),
            visible=False,
            label="実行結果",
        ),
        gr.HTML(value="", visible=False),
    )


def _component_update_value(component) -> str:
    """Read a value from Gradio's component and update return shapes."""
    if isinstance(component, dict):
        return str(component.get("value", "") or "").strip()
    value = getattr(component, "value", None)
    if value not in (None, ""):
        return str(value).strip()
    for constructor_args in getattr(component, "_constructor_args", ()):
        if isinstance(constructor_args, dict):
            value = constructor_args.get("value")
            if value not in (None, ""):
                return str(value).strip()
    return ""


def _execute_vpd_test(vpd_pool, username: str, sql: str):
    """Run one VPD SELECT and normalize its components for the management UI."""
    if vpd_pool is None:
        return _execution_test_empty_updates(
            "❌ VPD実行プールが有効ではありません。"
            "VPD設定と実行ユーザーを確認してください"
        )

    try:
        status_component, result_component, style_component = (
            execute_select_sql(
                vpd_pool,
                sql,
                1000,
                login_user=str(username or ""),
            )
        )
    except Exception:
        logger.exception("VPD実行テストに失敗しました")
        return _execution_test_empty_updates(
            "❌ 実行に失敗しました。"
            "VPD実行プールとサーバーログを確認してください"
        )

    status = _component_update_value(status_component)
    if not status:
        logger.error("VPD実行テストから空の状態メッセージが返されました")
        return _execution_test_empty_updates(
            "❌ 実行結果を取得できませんでした。"
            "サーバーログを確認してください"
        )
    return (
        gr.Markdown(
            value=status,
            visible=True,
            elem_classes=_configuration_status_classes(status),
        ),
        result_component,
        style_component,
    )


def _policy_metadata_name(value: str | None, label: str) -> str:
    """Validate a metadata identifier without changing quoted-name casing."""
    name = str(value or "").strip()
    if (
        not name
        or len(name) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise ValueError(f"{label}が不正です")
    return name


def _encode_delete_target(target: DeleteTarget) -> str:
    if target.kind == DELETE_TARGET_MANAGED:
        payload = [target.kind, target.object_name]
    elif target.kind == DELETE_TARGET_UNMANAGED:
        payload = [
            target.kind,
            target.object_name,
            target.policy_group,
            target.policy_name,
        ]
    else:
        raise ValueError("削除対象の種別が不正です")
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _decode_delete_target(value: str | None) -> DeleteTarget:
    if not str(value or "").strip():
        raise ValueError("削除する対象を選択してください")
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("削除対象が不正です") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("削除対象が不正です")
    kind = str(payload[0])
    if kind == DELETE_TARGET_MANAGED and len(payload) == 2:
        return DeleteTarget(
            kind=kind,
            object_name=_object_name(payload[1]),
        )
    if kind == DELETE_TARGET_UNMANAGED and len(payload) == 4:
        return DeleteTarget(
            kind=kind,
            object_name=_object_name(payload[1]),
            policy_group=_policy_metadata_name(payload[2], "Policy Group"),
            policy_name=_policy_metadata_name(payload[3], "Policy名"),
        )
    raise ValueError("削除対象が不正です")


def _delete_target_or_none(value: str | None) -> DeleteTarget | None:
    try:
        return _decode_delete_target(value)
    except ValueError:
        return None


def _delete_target_choices(
    business_objects: list[tuple[str, str]],
    registered: dict[str, ManagedRule],
    policies: list[tuple],
) -> list[tuple[str, str]]:
    """Return managed rules and live unmanaged SELECT policies."""
    choices: list[tuple[str, str]] = []
    seen_values: set[str] = set()
    policy_rows_by_object: dict[str, list[tuple]] = {}
    for policy in policies:
        policy_rows_by_object.setdefault(str(policy[0]), []).append(policy)

    for object_name, object_type in business_objects:
        rule = _managed_rule(registered.get(object_name))
        if rule is not None:
            access_mode = rule.access_mode
            access_label = ACCESS_MODE_LABELS.get(access_mode, access_mode)
            target_value = _encode_delete_target(
                DeleteTarget(
                    kind=DELETE_TARGET_MANAGED,
                    object_name=object_name,
                )
            )
            choices.append(
                (
                    "管理対象ルール · "
                    f"{object_type} · {object_name} · {access_label}",
                    target_value,
                )
            )
            seen_values.add(target_value)

        for policy in policy_rows_by_object.get(object_name, []):
            if str(policy[5]).upper() != "YES":
                continue
            if _policy_matches_managed_rule(policy, rule):
                continue
            policy_group = str(policy[10])
            policy_name = str(policy[1])
            target_value = _encode_delete_target(
                DeleteTarget(
                    kind=DELETE_TARGET_UNMANAGED,
                    object_name=object_name,
                    policy_group=policy_group,
                    policy_name=policy_name,
                )
            )
            if target_value in seen_values:
                continue
            choices.append(
                (
                    "未登録Policy · "
                    f"{object_type} · {object_name} · "
                    f"{policy_group} · {policy_name}",
                    target_value,
                )
            )
            seen_values.add(target_value)
    return choices


def _delete_control_labels(target_value: str | None) -> tuple[str, str]:
    """Return target-bound confirmation and destructive-button labels."""
    target = _delete_target_or_none(target_value)
    if target is None:
        return DELETE_CONFIRM_LABEL, DELETE_BUTTON_LABEL
    if target.kind == DELETE_TARGET_UNMANAGED:
        return (
            f"{target.object_name} の未登録Policy "
            f"{target.policy_name}（{target.policy_group}）、"
            "SQL_ASSIST_RUNTIMEのREAD／SELECT、および安全条件を満たす"
            "孤児Functionを削除することを確認しました",
            f"{target.object_name} の未登録Policy "
            f"{target.policy_name} を削除",
        )
    return (
        f"{target.object_name} のREAD権限、管理対象Policy／Function、"
        "台帳情報を削除することを確認しました",
        f"{target.object_name} のアクセスルールを削除",
    )


def _management_section_visibility(
    broad_read: bool,
    has_deletable_rules: bool,
) -> tuple[bool, bool]:
    """Return migration and managed-rule action visibility."""
    _ = has_deletable_rules
    return bool(broad_read), True


def build_vpd_management_tab(pool, vpd_pool):
    """Build the VPD management content inside an ADMIN-only parent tab."""
    gr.Markdown(
        "ℹ️ 実行ユーザーは `SQL_ASSIST_RUNTIME` 固定です。\n\n"
        "ℹ️ `READ` がオブジェクトへの入口、VPDが表示行の絞り込みです。"
        "オブジェクトの許可リストは全VPDログインユーザー共通で、"
        "ユーザー間の差は行データだけです。本画面はSELECTだけを管理し、"
        "列のマスキングは行いません。\n\n"
        "⚠️ **内部PoC向け:** VPDログインユーザーは共有パスワードです。"
        "ユーザー名の相互なりすましを防ぐ本番の認証境界ではありません。",
        elem_classes="tab-intro",
    )

    with gr.Accordion(label="1. 状態確認", open=True):
        config_btn = gr.Button("状態確認", variant="primary")
        with gr.Row():
            config_status = gr.Markdown(
                visible=False,
                elem_classes=["operation-status"],
            )
        with gr.Row():
            config_df = gr.Dataframe(
                value=pd.DataFrame(columns=["確認項目", "状態", "詳細"]),
                label="設定・バージョン・実行プール・権限の状態",
                interactive=False,
                wrap=True,
                visible=False,
                max_height=300,
                column_widths=["28%", "16%", "56%"],
                elem_classes=["vpd-status-table"],
            )

    with gr.Accordion(label="2. 実行ユーザー", open=True):
        with gr.Row(elem_classes=["vpd-form-row"]):
            with gr.Column(scale=1):
                gr.Markdown(
                    "プリセットSQL",
                    elem_classes="input-label",
                )
            with gr.Column(scale=5):
                gr.Code(
                    value=ACCOUNT_SQL_PREVIEW,
                    language="sql",
                    lines=6,
                    max_lines=8,
                    label="プリセットSQL",
                    show_label=False,
                    container=False,
                    interactive=False,
                    elem_classes=["vpd-form-control"],
                )
        gr.Markdown(
            "ℹ️ プレビューとログには実際のパスワードを表示しません。"
        )
        create_account_btn = gr.Button(
            "実行ユーザーの作成・確認", variant="primary"
        )
        account_status = gr.Markdown(
            visible=False,
            elem_classes=["operation-status"],
        )

    with gr.Accordion(label="3. VPD基盤", open=True):
        with gr.Row(elem_classes=["vpd-form-row"]):
            with gr.Column(scale=1):
                gr.Markdown(
                    "インストールSQL",
                    elem_classes="input-label",
                )
            with gr.Column(scale=5):
                gr.Code(
                    value=FOUNDATION_SQL_PREVIEW,
                    language="sql",
                    lines=12,
                    max_lines=18,
                    label="インストールSQL",
                    show_label=False,
                    container=False,
                    interactive=False,
                    elem_classes=["vpd-form-control"],
                )
        context_btn = gr.Button(
            "VPD基盤のインストール/更新",
            variant="primary",
        )
        context_status = gr.Markdown(
            visible=False,
            elem_classes=["operation-status"],
        )

    with gr.Accordion(label="4. データアクセスルール", open=True):
        gr.Markdown(
            "対象とアクセス方法を選ぶと、Function名・Policy名・SQLは"
            "システムが生成します。SQLプレビューは確認専用です。",
            elem_classes=["vpd-rule-help"],
        )
        inventory_btn = gr.Button(
            "対象・現在状態を更新",
            variant="secondary",
        )
        inventory_status = gr.Markdown(
            visible=False,
            elem_classes=["operation-status"],
        )

        with gr.Accordion(label="アクセス状態・競合", open=False):
            object_df = gr.Dataframe(
                label="業務オブジェクトのアクセス状態",
                interactive=False,
                wrap=True,
                visible=False,
                max_height=300,
                elem_classes=["vpd-inventory-table"],
            )
            policy_df = gr.Dataframe(
                label="既存SELECT Policy（未登録は参照のみ・競合）",
                interactive=False,
                wrap=True,
                visible=False,
                max_height=300,
                elem_classes=["vpd-inventory-table"],
            )

        with gr.Accordion(label="ルールを設定", open=True):
            with gr.Row(elem_classes=["vpd-form-row"]):
                with gr.Column(scale=5):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown(
                                "1. 業務テーブル / ビュー",
                                elem_classes="input-label",
                            )
                            gr.Markdown(
                                "ℹ️ 内部表、生成オブジェクト、"
                                "ベクトル索引の補助表は表示しません。",
                                elem_classes=["vpd-form-hint"],
                            )
                        with gr.Column(scale=5):
                            rule_object = gr.Dropdown(
                                label="1. 業務テーブル / ビュー",
                                choices=[],
                                show_label=False,
                                container=False,
                                elem_classes=["vpd-form-control"],
                            )
                with gr.Column(scale=5):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown(
                                "2. アクセス方法",
                                elem_classes="input-label",
                            )
                            gr.Markdown(
                                "ℹ️ 参照を許可しない：データを参照できません。"
                                "すべての行の参照を許可：すべてのデータを"
                                "参照できます。ログインユーザーの行のみ："
                                + ROW_MATCH_DESCRIPTION
                                + " 関連テーブルで判定："
                                + RELATION_MATCH_DESCRIPTION,
                                elem_classes=["vpd-form-hint"],
                            )
                        with gr.Column(scale=5):
                            access_mode = gr.Radio(
                                choices=[
                                    (
                                        ACCESS_MODE_LABELS[ACCESS_NONE],
                                        ACCESS_NONE,
                                    ),
                                    (
                                        ACCESS_MODE_LABELS[ACCESS_FULL],
                                        ACCESS_FULL,
                                    ),
                                    (
                                        ACCESS_MODE_LABELS[
                                            ACCESS_ROW_MATCH
                                        ],
                                        ACCESS_ROW_MATCH,
                                    ),
                                    (
                                        ACCESS_MODE_LABELS[
                                            ACCESS_RELATION_MATCH
                                        ],
                                        ACCESS_RELATION_MATCH,
                                    ),
                                ],
                                value=ACCESS_NONE,
                                label="2. アクセス方法",
                                show_label=False,
                                container=False,
                                elem_classes=[
                                    "vpd-form-control",
                                    "vpd-access-mode",
                                ],
                            )
            with gr.Column(
                visible=False,
                elem_classes=["vpd-dynamic-field"],
            ) as match_column_section:
                with gr.Row(elem_classes=["vpd-form-row"]):
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "3. ログインユーザー照合列",
                            elem_classes="input-label",
                        )
                    with gr.Column(scale=5):
                        match_column = gr.Dropdown(
                            label="3. ログインユーザー照合列",
                            choices=[],
                            show_label=False,
                            container=False,
                            elem_classes=["vpd-form-control"],
                        )
                        gr.Markdown(
                            "ℹ️ 文字列列はLOGIN_USERと完全一致、"
                            "整数型NUMBER列は数値として一致する行だけを"
                            "表示します。",
                            elem_classes=["vpd-form-hint"],
                        )
            with gr.Column(
                visible=False,
                elem_classes=["vpd-dynamic-field"],
            ) as relation_section:
                with gr.Row(
                    elem_classes=["vpd-form-row", "vpd-relation-row"]
                ):
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "3. 対象側の関連列",
                            elem_classes="input-label",
                        )
                        relation_target_column = gr.Dropdown(
                            label="3. 対象側の関連列",
                            choices=[],
                            show_label=False,
                            container=False,
                            elem_classes=["vpd-form-control"],
                        )
                        gr.Markdown(
                            "ℹ️ 関連テーブルの列と同じデータ型の列を"
                            "選択します。",
                            elem_classes=["vpd-form-hint"],
                        )
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "4. 判定に使用する関連テーブル",
                            elem_classes="input-label",
                        )
                        relation_object = gr.Dropdown(
                            label="4. 判定に使用する関連テーブル",
                            choices=[],
                            show_label=False,
                            container=False,
                            elem_classes=["vpd-form-control"],
                        )
                        gr.Markdown(
                            "ℹ️ ADMIN schemaの通常の業務テーブルだけを"
                            "表示します。",
                            elem_classes=["vpd-form-hint"],
                        )
                with gr.Row(
                    elem_classes=["vpd-form-row", "vpd-relation-row"]
                ):
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "5. 関連テーブル側の関連列",
                            elem_classes="input-label",
                        )
                        relation_object_column = gr.Dropdown(
                            label="5. 関連テーブル側の関連列",
                            choices=[],
                            show_label=False,
                            container=False,
                            elem_classes=["vpd-form-control"],
                        )
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "6. 関連テーブルのログインユーザー照合列",
                            elem_classes="input-label",
                        )
                        relation_match_column = gr.Dropdown(
                            label=(
                                "6. 関連テーブルの"
                                "ログインユーザー照合列"
                            ),
                            choices=[],
                            show_label=False,
                            container=False,
                            elem_classes=["vpd-form-control"],
                        )
                        gr.Markdown(
                            "ℹ️ 文字列列または整数型NUMBER列を"
                            "LOGIN_USERと照合します。",
                            elem_classes=["vpd-form-hint"],
                        )
            rule_input_status = gr.Markdown(
                visible=False,
                elem_classes=["operation-status"],
            )
            preview_btn = gr.Button(
                "SQLを確認",
                variant="secondary",
            )
            preview_status = gr.Markdown(
                visible=False,
                elem_classes=["operation-status"],
            )
            with gr.Column(
                visible=False,
                elem_classes=["vpd-dynamic-field"],
            ) as rule_sql_section:
                with gr.Row(elem_classes=["vpd-form-row"]):
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "システム生成SQL（確認専用）",
                            elem_classes="input-label",
                        )
                    with gr.Column(scale=5):
                        rule_sql = gr.Code(
                            value="",
                            language="sql",
                            label="システム生成SQL（確認専用）",
                            lines=12,
                            max_lines=22,
                            show_label=False,
                            container=False,
                            interactive=False,
                            elem_classes=["vpd-form-control"],
                        )
            validation_state = gr.State("")
            apply_rule_btn = gr.Button(
                "このアクセスルールを適用",
                variant="primary",
                interactive=False,
            )
            apply_status = gr.Markdown(
                visible=False,
                elem_classes=["operation-status"],
            )
            gr.Markdown(
                "ℹ️ 対象・アクセス方法・判定条件を変更すると確認結果は"
                "無効になります。再確認が完了するまで適用ボタンは使えません。"
            )

        with gr.Accordion(
            label="高度な操作（ルール削除）",
            open=False,
            visible=True,
            elem_classes=["vpd-danger-zone"],
        ):
            gr.Markdown(
                "削除する管理ルールまたは未登録SELECT Policyを"
                "選択してください。削除前にSQL_ASSIST_RUNTIMEの"
                "READ／SELECTを失効し、安全側に倒します。"
            )
            with gr.Row(elem_classes=["vpd-form-row"]):
                with gr.Column(scale=1):
                    gr.Markdown(
                        "削除対象",
                        elem_classes="input-label",
                    )
                with gr.Column(scale=5):
                    delete_rule_object = gr.Dropdown(
                        label="削除対象（管理ルール／未登録Policy）",
                        choices=[],
                        value=None,
                        interactive=False,
                        show_label=False,
                        container=False,
                        elem_classes=["vpd-form-control"],
                    )
                    delete_target_hint = gr.Markdown(
                        f"ℹ️ {DELETE_EMPTY_HINT}",
                        elem_classes=["vpd-form-hint"],
                    )
            with gr.Row(elem_classes=["vpd-form-row"]):
                with gr.Column(scale=1):
                    gr.Markdown(
                        "削除影響の確認",
                        elem_classes="input-label",
                    )
                with gr.Column(scale=5):
                    delete_confirm = gr.Checkbox(
                        label=DELETE_CONFIRM_LABEL,
                        value=False,
                        interactive=False,
                        show_label=False,
                        container=False,
                        elem_classes=["vpd-form-control"],
                    )
            delete_rule_btn = gr.Button(
                "アクセスルールを削除",
                variant="stop",
                interactive=False,
                elem_classes=["vpd-delete-action"],
            )
            delete_status = gr.Markdown(
                visible=False,
                elem_classes=["operation-status"],
            )

        with gr.Accordion(
            label="テーブル許可リストへの移行（初回のみ）",
            open=False,
            visible=False,
            elem_classes=["vpd-danger-zone"],
        ) as allowlist_section:
            gr.Markdown(
                "⚠️ 移行では、ACTIVEなルールを検証して個別READを準備した後、"
                "最後にADMIN schema全表読取を失効します。完了後、未登録"
                "オブジェクトは参照できません。"
            )
            with gr.Row(elem_classes=["vpd-form-row"]):
                with gr.Column(scale=1):
                    gr.Markdown(
                        "移行内容の確認",
                        elem_classes="input-label",
                    )
                with gr.Column(scale=5):
                    allowlist_confirm = gr.Checkbox(
                        label=ALLOWLIST_CONFIRM_LABEL,
                        value=False,
                        show_label=False,
                        container=False,
                        elem_classes=["vpd-form-control"],
                    )
            allowlist_btn = gr.Button(
                "テーブル許可リストへ移行",
                variant="stop",
                interactive=False,
            )
            allowlist_status = gr.Markdown(
                visible=False,
                elem_classes=["operation-status"],
            )

    with gr.Accordion(label="5. 実行テスト", open=True):
        users = list(normalize_vpd_login_users())
        with gr.Row(elem_classes=["vpd-form-row"]):
            with gr.Column(scale=1):
                gr.Markdown(
                    "VPDログインユーザー",
                    elem_classes="input-label",
                )
            with gr.Column(scale=5):
                test_user = gr.Dropdown(
                    choices=users,
                    value=users[0] if users else None,
                    label="VPDログインユーザー",
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                )
        with gr.Row(elem_classes=["vpd-form-row"]):
            with gr.Column(scale=1):
                gr.Markdown(
                    "SELECTテスト",
                    elem_classes="input-label",
                )
            with gr.Column(scale=5):
                test_sql = gr.Code(
                    value=(
                        "SELECT SYS_CONTEXT('SQL_ASSIST_CTX', "
                        "'LOGIN_USER') AS LOGIN_USER FROM DUAL;"
                    ),
                    language="sql",
                    lines=8,
                    max_lines=15,
                    label="SELECTテスト",
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                )
        test_btn = gr.Button(
            "VPD実行プールでテスト", variant="primary"
        )
        test_status = gr.Markdown(
            visible=False,
            elem_classes=["operation-status"],
        )
        test_df = gr.Dataframe(
            label="実行結果",
            interactive=False,
            wrap=True,
            visible=False,
        )
        test_style = gr.HTML(visible=False)

    def on_config(request: gr.Request):
        for status, frame in _configuration_status_stream(
            pool,
            vpd_pool,
            request,
        ):
            display_frame = (
                _style_configuration_frame(frame)
                if frame is not None
                else pd.DataFrame()
            )
            yield (
                gr.Markdown(
                    value=status,
                    visible=True,
                    elem_classes=_configuration_status_classes(status),
                ),
                gr.Dataframe(
                    value=display_frame,
                    visible=frame is not None,
                    column_widths=["28%", "16%", "56%"],
                    elem_classes=["vpd-status-table"],
                ),
            )

    def on_account(request: gr.Request):
        status = _safe_status(
            _create_runtime_account, pool, vpd_pool, request
        )
        return gr.Markdown(
            value=status,
            visible=True,
            elem_classes=_configuration_status_classes(status),
        )

    def on_context(request: gr.Request):
        status = _safe_status(_install_context, pool, request)
        return gr.Markdown(
            value=status,
            visible=True,
            elem_classes=_configuration_status_classes(status),
        )

    def _inventory_updates(
        selected_object,
        selected_delete_object,
        request: gr.Request,
    ):
        try:
            policies, objects = _policy_inventory(pool, request)
            with pool.acquire() as conn:
                with conn.cursor() as cursor:
                    business_objects = _business_objects(cursor)
                    registered = _registry_rules(cursor)
                    policy_metadata = _policy_rows(cursor)
                    broad_read = _runtime_has_schema_read(cursor)
            object_choices = [
                (f"{object_type} · {object_name}", object_name)
                for object_name, object_type in business_objects
            ]
            business_names = {
                object_name for object_name, _object_type in business_objects
            }
            selected_name = str(selected_object or "").strip().upper()
            if selected_name not in business_names:
                selected_name = None
            delete_choices = _delete_target_choices(
                business_objects,
                registered,
                policy_metadata,
            )
            delete_values = {
                value for _label, value in delete_choices
            }
            selected_delete_name = str(selected_delete_object or "").strip()
            if selected_delete_name not in delete_values:
                selected_delete_name = None
            confirm_label, delete_button_label = _delete_control_labels(
                selected_delete_name
            )
            show_migration, _show_advanced = _management_section_visibility(
                broad_read,
                bool(delete_choices),
            )
            status_prefix = "⚠️" if broad_read else "✅"
            status_detail = (
                " schema全表読取が残っているため、初回移行が必要です。"
                if broad_read
                else ""
            )
            status_class = (
                "operation-status--warning"
                if broad_read
                else "operation-status--success"
            )
            return (
                gr.Dropdown(
                    choices=object_choices,
                    value=selected_name,
                    label="1. 業務テーブル / ビュー",
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Accordion(open=False, visible=show_migration),
                gr.Checkbox(
                    value=False,
                    label=ALLOWLIST_CONFIRM_LABEL,
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Button(interactive=False),
                gr.Dropdown(
                    choices=delete_choices,
                    value=selected_delete_name,
                    label="削除対象（管理ルール／未登録Policy）",
                    interactive=bool(delete_choices),
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Markdown(
                    value=(
                        "ℹ️ 管理対象ルールと未登録SELECT Policyを"
                        "データベースの状態から表示します。"
                        if delete_choices
                        else f"ℹ️ {DELETE_EMPTY_HINT}"
                    ),
                    elem_classes=["vpd-form-hint"],
                ),
                gr.Checkbox(
                    value=False,
                    label=confirm_label,
                    interactive=selected_delete_name is not None,
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Button(
                    value=delete_button_label,
                    interactive=False,
                ),
                gr.Dataframe(value=objects, visible=True),
                gr.Dataframe(value=policies, visible=True),
                gr.Markdown(
                    value=(
                        f"{status_prefix} "
                        f"{len(business_objects)}件の業務オブジェクトと"
                        f"{len(registered)}件の管理ルールを読み込みました"
                        f"{status_detail}"
                    ),
                    visible=True,
                    elem_classes=[
                        "operation-status",
                        status_class,
                    ],
                ),
            )
        except Exception as exc:
            return (
                gr.Dropdown(
                    choices=[],
                    value=None,
                    label="1. 業務テーブル / ビュー",
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Accordion(open=False, visible=False),
                gr.Checkbox(
                    value=False,
                    label=ALLOWLIST_CONFIRM_LABEL,
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Button(interactive=False),
                gr.Dropdown(
                    choices=[],
                    value=None,
                    label="削除対象（管理ルール／未登録Policy）",
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Markdown(
                    value="ℹ️ 削除対象を読み込めませんでした",
                    elem_classes=["vpd-form-hint"],
                ),
                gr.Checkbox(
                    value=False,
                    label=DELETE_CONFIRM_LABEL,
                    interactive=False,
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Button(
                    value=DELETE_BUTTON_LABEL,
                    interactive=False,
                ),
                gr.Dataframe(value=pd.DataFrame(), visible=False),
                gr.Dataframe(value=pd.DataFrame(), visible=False),
                gr.Markdown(
                    value=f"❌ {exc}",
                    visible=True,
                    elem_classes=[
                        "operation-status",
                        "operation-status--error",
                    ],
                ),
            )

    def on_inventory(
        selected_object,
        selected_delete_object,
        request: gr.Request,
    ):
        return (
            *_inventory_updates(
                selected_object,
                selected_delete_object,
                request,
            ),
            hidden_operation_status(),
            hidden_operation_status(),
            hidden_operation_status(),
        )

    def on_operation_refresh(
        selected_object,
        selected_delete_object,
        request: gr.Request,
    ):
        return _inventory_updates(
            selected_object,
            selected_delete_object,
            request,
        )[:-1]

    def on_allowlist_refresh(
        selected_object,
        selected_delete_object,
        request: gr.Request,
    ):
        updates = list(
            _inventory_updates(
                selected_object,
                selected_delete_object,
                request,
            )[:-1]
        )
        updates[1] = gr.Accordion(open=True, visible=True)
        return tuple(updates)

    def metadata_choices(rows):
        return [
            (
                f"{name} · {data_type}"
                f"{' · NULL可' if nullable == 'Y' else ''}",
                name,
            )
            for name, data_type, nullable in rows
        ]

    def empty_dropdown(label):
        return gr.Dropdown(
            choices=[],
            value=None,
            label=label,
            show_label=False,
            container=False,
            elem_classes=["vpd-form-control"],
        )

    def hidden_operation_status():
        return gr.Markdown(
            visible=False,
            elem_classes=["operation-status"],
        )

    def empty_rule_editor(status: str | None = None):
        return (
            gr.Radio(
                value=ACCESS_NONE,
                label="2. アクセス方法",
                show_label=False,
                container=False,
                elem_classes=["vpd-form-control", "vpd-access-mode"],
            ),
            gr.Column(visible=False),
            empty_dropdown("3. ログインユーザー照合列"),
            gr.Column(visible=False),
            empty_dropdown("3. 対象側の関連列"),
            empty_dropdown("4. 判定に使用する関連テーブル"),
            empty_dropdown("5. 関連テーブル側の関連列"),
            empty_dropdown(
                "6. 関連テーブルのログインユーザー照合列"
            ),
            gr.Markdown(
                value=status or "",
                visible=bool(status),
                elem_classes=_configuration_status_classes(status or ""),
            ),
            hidden_operation_status(),
            gr.Column(visible=False),
            gr.Code(
                value="",
                language="sql",
                label="システム生成SQL（確認専用）",
                show_label=False,
                container=False,
                interactive=False,
                elem_classes=["vpd-form-control"],
            ),
            "",
            gr.Button(interactive=False),
        )

    def on_object_change(object_name, request: gr.Request):
        if not object_name:
            return empty_rule_editor()
        try:
            require_admin(request)
            with pool.acquire() as conn:
                with conn.cursor() as cursor:
                    object_type = _require_business_object(
                        cursor, object_name
                    )
                    match_columns = _match_columns(cursor, object_name)
                    target_columns = _relation_columns(cursor, object_name)
                    relation_tables = _relation_objects(cursor, object_name)
                    registered = _managed_rule(
                        _registry_rule(cursor, object_name)
                    )
                    broad_read = _runtime_has_schema_read(cursor)
                    registry_current = _registry_schema_current(cursor)
                    conflicts = _unmanaged_policy_names(
                        cursor, object_name, registered
                    )
                    mode = (
                        registered.access_mode
                        if registered
                        else ACCESS_NONE
                    )
                    match_value = (
                        registered.match_column if registered else None
                    )
                    relation_target_value = (
                        registered.relation_target_column
                        if registered
                        else None
                    )
                    relation_object_value = (
                        registered.relation_object
                        if registered
                        else None
                    )
                    relation_object_value = (
                        relation_object_value
                        if relation_object_value
                        in {name for name, _type in relation_tables}
                        else None
                    )
                    source_columns = []
                    source_match_columns = []
                    if relation_object_value:
                        target_types = {
                            name: data_type
                            for name, data_type, _nullable in target_columns
                        }
                        target_type = target_types.get(
                            relation_target_value
                        )
                        source_columns = [
                            row
                            for row in _relation_columns(
                                cursor, relation_object_value
                            )
                            if target_type and row[1] == target_type
                        ]
                        source_match_columns = _match_columns(
                            cursor, relation_object_value
                        )
            match_choices = metadata_choices(match_columns)
            target_choices = metadata_choices(target_columns)
            relation_choices = [
                (f"TABLE · {name}", name)
                for name, _object_type in relation_tables
            ]
            source_choices = metadata_choices(source_columns)
            source_match_choices = metadata_choices(source_match_columns)
            source_column_values = {value for _label, value in source_choices}
            source_match_values = {
                value for _label, value in source_match_choices
            }
            relation_object_column_value = (
                registered.relation_object_column
                if registered
                and registered.relation_object_column in source_column_values
                else None
            )
            relation_match_column_value = (
                registered.relation_match_column
                if registered
                and registered.relation_match_column in source_match_values
                else None
            )
            notes = [f"{object_type} {object_name} を選択しました。"]
            if not registry_current:
                notes.append(
                    "VPD基盤のインストール/更新が必要です。"
                )
            if broad_read:
                notes.append("schema全表読取が残っている移行中です。")
            if conflicts:
                notes.append(
                    "未登録Policyと競合: " + ", ".join(sorted(conflicts))
                )
            status = (
                "⚠️ " + " ".join(notes)
                if broad_read or conflicts or not registry_current
                else "ℹ️ " + " ".join(notes)
            )
            return (
                gr.Radio(
                    value=mode,
                    label="2. アクセス方法",
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control", "vpd-access-mode"],
                ),
                gr.Column(visible=mode == ACCESS_ROW_MATCH),
                gr.Dropdown(
                    choices=match_choices,
                    value=match_value,
                    label="3. ログインユーザー照合列",
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Column(visible=mode == ACCESS_RELATION_MATCH),
                gr.Dropdown(
                    choices=target_choices,
                    value=relation_target_value,
                    label="3. 対象側の関連列",
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Dropdown(
                    choices=relation_choices,
                    value=relation_object_value,
                    label="4. 判定に使用する関連テーブル",
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Dropdown(
                    choices=source_choices,
                    value=relation_object_column_value,
                    label="5. 関連テーブル側の関連列",
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Dropdown(
                    choices=source_match_choices,
                    value=relation_match_column_value,
                    label=(
                        "6. 関連テーブルのログインユーザー照合列"
                    ),
                    show_label=False,
                    container=False,
                    elem_classes=["vpd-form-control"],
                ),
                gr.Markdown(
                    value=status,
                    visible=True,
                    elem_classes=_configuration_status_classes(status),
                ),
                hidden_operation_status(),
                gr.Column(visible=False),
                gr.Code(
                    value="",
                    language="sql",
                    label="システム生成SQL（確認専用）",
                    show_label=False,
                    container=False,
                    interactive=False,
                    elem_classes=["vpd-form-control"],
                ),
                "",
                gr.Button(interactive=False),
            )
        except Exception as exc:
            return empty_rule_editor(f"❌ {exc}")

    def invalidate_preview(mode):
        return (
            gr.Column(visible=mode == ACCESS_ROW_MATCH),
            gr.Column(visible=mode == ACCESS_RELATION_MATCH),
            hidden_operation_status(),
            gr.Column(visible=False),
            gr.Code(
                value="",
                language="sql",
                label="システム生成SQL（確認専用）",
                show_label=False,
                container=False,
                interactive=False,
                elem_classes=["vpd-form-control"],
            ),
            "",
            gr.Button(interactive=False),
        )

    def invalidate_column_preview(*_values):
        return (
            hidden_operation_status(),
            gr.Column(visible=False),
            gr.Code(
                value="",
                language="sql",
                label="システム生成SQL（確認専用）",
                show_label=False,
                container=False,
                interactive=False,
                elem_classes=["vpd-form-control"],
            ),
            "",
            gr.Button(interactive=False),
        )

    def compatible_relation_columns(
        target_object,
        target_column,
        source_object,
        request: gr.Request,
    ):
        try:
            require_admin(request)
            if not target_object or not target_column or not source_object:
                return []
            with pool.acquire() as conn:
                with conn.cursor() as cursor:
                    target_types = {
                        name: data_type
                        for name, data_type, _nullable in _relation_columns(
                            cursor, target_object
                        )
                    }
                    target_type = target_types.get(
                        _column_name(target_column)
                    )
                    _require_relation_table(cursor, source_object)
                    return [
                        row
                        for row in _relation_columns(cursor, source_object)
                        if target_type and row[1] == target_type
                    ]
        except Exception:
            logger.exception("関連列候補の読み込みに失敗しました")
            return []

    def on_relation_target_change(
        target_object,
        target_column,
        source_object,
        request: gr.Request,
    ):
        choices = metadata_choices(
            compatible_relation_columns(
                target_object,
                target_column,
                source_object,
                request,
            )
        )
        return (
            gr.Dropdown(
                choices=choices,
                value=None,
                label="5. 関連テーブル側の関連列",
                show_label=False,
                container=False,
                elem_classes=["vpd-form-control"],
            ),
            *invalidate_column_preview(),
        )

    def on_relation_object_change(
        target_object,
        target_column,
        source_object,
        request: gr.Request,
    ):
        relation_columns = compatible_relation_columns(
            target_object,
            target_column,
            source_object,
            request,
        )
        match_columns = []
        if source_object:
            try:
                require_admin(request)
                with pool.acquire() as conn:
                    with conn.cursor() as cursor:
                        _require_relation_table(cursor, source_object)
                        match_columns = _match_columns(
                            cursor, source_object
                        )
            except Exception:
                logger.exception(
                    "関連テーブルの照合列候補の読み込みに失敗しました"
                )
        return (
            gr.Dropdown(
                choices=metadata_choices(relation_columns),
                value=None,
                label="5. 関連テーブル側の関連列",
                show_label=False,
                container=False,
                elem_classes=["vpd-form-control"],
            ),
            gr.Dropdown(
                choices=metadata_choices(match_columns),
                value=None,
                label="6. 関連テーブルのログインユーザー照合列",
                show_label=False,
                container=False,
                elem_classes=["vpd-form-control"],
            ),
            *invalidate_column_preview(),
        )

    def on_preview(
        object_name,
        mode,
        column,
        relation_target,
        relation_table,
        relation_column,
        relation_login_column,
        request: gr.Request,
    ):
        try:
            status, sql, token = _preview_access_rule(
                pool,
                object_name,
                mode,
                column,
                request,
                relation_target,
                relation_table,
                relation_column,
                relation_login_column,
            )
            status_class = (
                "operation-status--warning"
                if status.startswith("⚠️")
                else "operation-status--success"
            )
            return (
                gr.Markdown(
                    value=status,
                    visible=True,
                    elem_classes=["operation-status", status_class],
                ),
                gr.Column(visible=True),
                gr.Code(
                    value=sql,
                    language="sql",
                    label="システム生成SQL（確認専用）",
                    show_label=False,
                    container=False,
                    interactive=False,
                    elem_classes=["vpd-form-control"],
                ),
                token,
                gr.Button(interactive=True),
                hidden_operation_status(),
            )
        except Exception as exc:
            return (
                gr.Markdown(
                    value=f"❌ {exc}",
                    visible=True,
                    elem_classes=[
                        "operation-status",
                        "operation-status--error",
                    ],
                ),
                gr.Column(visible=False),
                gr.Code(
                    value="",
                    language="sql",
                    label="システム生成SQL（確認専用）",
                    show_label=False,
                    container=False,
                    interactive=False,
                    elem_classes=["vpd-form-control"],
                ),
                "",
                gr.Button(interactive=False),
                hidden_operation_status(),
            )

    def on_apply(
        object_name,
        mode,
        column,
        relation_target,
        relation_table,
        relation_column,
        relation_login_column,
        token,
        request: gr.Request,
    ):
        status = _safe_status(
            _apply_access_rule,
            pool,
            vpd_pool,
            object_name,
            mode,
            column,
            token,
            request,
            relation_target,
            relation_table,
            relation_column,
            relation_login_column,
        )
        return (
            gr.Markdown(
                value=status,
                visible=True,
                elem_classes=_configuration_status_classes(status),
            ),
            "",
            gr.Button(interactive=False),
        )

    def on_delete(object_name, confirmed, request: gr.Request):
        status = _safe_status(
            _delete_target,
            pool,
            vpd_pool,
            object_name,
            confirmed,
            request,
        )
        return gr.Markdown(
            value=status,
            visible=True,
            elem_classes=_configuration_status_classes(status),
        )

    def on_delete_target_change(object_name):
        confirm_label, delete_button_label = _delete_control_labels(
            object_name
        )
        has_target = _delete_target_or_none(object_name) is not None
        return (
            gr.Checkbox(
                value=False,
                label=confirm_label,
                interactive=has_target,
                show_label=False,
                container=False,
                elem_classes=["vpd-form-control"],
            ),
            gr.Button(
                value=delete_button_label,
                interactive=False,
            ),
            hidden_operation_status(),
        )

    def on_delete_confirm_change(object_name, confirmed):
        _confirm_label, delete_button_label = _delete_control_labels(
            object_name
        )
        return (
            gr.Button(
                value=delete_button_label,
                interactive=bool(
                    _delete_target_or_none(object_name) is not None
                    and confirmed
                ),
            ),
            hidden_operation_status(),
        )

    def on_allowlist_confirm_change(confirmed):
        return (
            gr.Button(interactive=bool(confirmed)),
            hidden_operation_status(),
        )

    def on_allowlist(confirmed, request: gr.Request):
        status = _safe_status(
            _activate_table_allowlist,
            pool,
            vpd_pool,
            confirmed,
            request,
        )
        return gr.Markdown(
            value=status,
            visible=True,
            elem_classes=_configuration_status_classes(status),
        )

    def on_test(username, sql, request: gr.Request):
        try:
            require_admin(request)
        except Exception:
            yield _execution_test_empty_updates(
                "❌ この操作はADMINユーザーのみ実行できます"
            )
            return
        yield _execution_test_empty_updates("⏳ 実行中...")
        yield _execute_vpd_test(vpd_pool, username, sql)

    config_btn.click(on_config, outputs=[config_status, config_df])
    create_account_btn.click(on_account, outputs=[account_status])
    context_btn.click(on_context, outputs=[context_status])
    inventory_refresh_outputs = [
        rule_object,
        allowlist_section,
        allowlist_confirm,
        allowlist_btn,
        delete_rule_object,
        delete_target_hint,
        delete_confirm,
        delete_rule_btn,
        object_df,
        policy_df,
        inventory_status,
    ]
    inventory_outputs = [
        *inventory_refresh_outputs,
        apply_status,
        delete_status,
        allowlist_status,
    ]
    operation_refresh_outputs = inventory_refresh_outputs[:-1]
    inventory_btn.click(
        on_inventory,
        inputs=[rule_object, delete_rule_object],
        outputs=inventory_outputs,
    )
    rule_object.change(
        on_object_change,
        inputs=[rule_object],
        outputs=[
            access_mode,
            match_column_section,
            match_column,
            relation_section,
            relation_target_column,
            relation_object,
            relation_object_column,
            relation_match_column,
            rule_input_status,
            preview_status,
            rule_sql_section,
            rule_sql,
            validation_state,
            apply_rule_btn,
        ],
    )
    access_mode.change(
        invalidate_preview,
        inputs=[access_mode],
        outputs=[
            match_column_section,
            relation_section,
            preview_status,
            rule_sql_section,
            rule_sql,
            validation_state,
            apply_rule_btn,
        ],
    )
    match_column.change(
        invalidate_column_preview,
        outputs=[
            preview_status,
            rule_sql_section,
            rule_sql,
            validation_state,
            apply_rule_btn,
        ],
    )
    relation_target_column.change(
        on_relation_target_change,
        inputs=[
            rule_object,
            relation_target_column,
            relation_object,
        ],
        outputs=[
            relation_object_column,
            preview_status,
            rule_sql_section,
            rule_sql,
            validation_state,
            apply_rule_btn,
        ],
    )
    relation_object.change(
        on_relation_object_change,
        inputs=[
            rule_object,
            relation_target_column,
            relation_object,
        ],
        outputs=[
            relation_object_column,
            relation_match_column,
            preview_status,
            rule_sql_section,
            rule_sql,
            validation_state,
            apply_rule_btn,
        ],
    )
    relation_object_column.change(
        invalidate_column_preview,
        outputs=[
            preview_status,
            rule_sql_section,
            rule_sql,
            validation_state,
            apply_rule_btn,
        ],
    )
    relation_match_column.change(
        invalidate_column_preview,
        outputs=[
            preview_status,
            rule_sql_section,
            rule_sql,
            validation_state,
            apply_rule_btn,
        ],
    )
    for rule_input in (
        rule_object,
        access_mode,
        match_column,
        relation_target_column,
        relation_object,
        relation_object_column,
        relation_match_column,
    ):
        rule_input.input(
            hidden_operation_status,
            outputs=[apply_status],
        )
    preview_btn.click(
        on_preview,
        inputs=[
            rule_object,
            access_mode,
            match_column,
            relation_target_column,
            relation_object,
            relation_object_column,
            relation_match_column,
        ],
        outputs=[
            preview_status,
            rule_sql_section,
            rule_sql,
            validation_state,
            apply_rule_btn,
            apply_status,
        ],
    )
    apply_event = apply_rule_btn.click(
        on_apply,
        inputs=[
            rule_object,
            access_mode,
            match_column,
            relation_target_column,
            relation_object,
            relation_object_column,
            relation_match_column,
            validation_state,
        ],
        outputs=[apply_status, validation_state, apply_rule_btn],
    )
    apply_event.then(
        on_operation_refresh,
        inputs=[rule_object, delete_rule_object],
        outputs=operation_refresh_outputs,
    )
    delete_rule_object.input(
        on_delete_target_change,
        inputs=[delete_rule_object],
        outputs=[delete_confirm, delete_rule_btn, delete_status],
    )
    delete_confirm.input(
        on_delete_confirm_change,
        inputs=[delete_rule_object, delete_confirm],
        outputs=[delete_rule_btn, delete_status],
    )
    delete_event = delete_rule_btn.click(
        on_delete,
        inputs=[delete_rule_object, delete_confirm],
        outputs=[delete_status],
    )
    delete_event.then(
        on_operation_refresh,
        inputs=[rule_object, delete_rule_object],
        outputs=operation_refresh_outputs,
    )
    allowlist_confirm.input(
        on_allowlist_confirm_change,
        inputs=[allowlist_confirm],
        outputs=[allowlist_btn, allowlist_status],
    )
    allowlist_event = allowlist_btn.click(
        on_allowlist,
        inputs=[allowlist_confirm],
        outputs=[allowlist_status],
    )
    allowlist_event.then(
        on_allowlist_refresh,
        inputs=[rule_object, delete_rule_object],
        outputs=operation_refresh_outputs,
    )
    test_btn.click(
        on_test,
        inputs=[test_user, test_sql],
        outputs=[test_status, test_df, test_style],
    )
