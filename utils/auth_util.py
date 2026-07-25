"""認証ユーティリティモジュール.

このモジュールは、OCI、OpenAI、Azure OpenAIなどの
各種サービスの認証情報を設定するための関数を提供します。
"""

from dataclasses import dataclass

from utils.vpd_util import authenticate


@dataclass(frozen=True)
class AccessNavigation:
    """Role-specific tab visibility and initial navigation targets."""

    settings_visible: bool
    management_visible: bool
    query_visible: bool
    selectai_visible: bool
    chat_visible: bool
    developer_features_visible: bool
    user_features_visible: bool
    sql_learning_schema_setup_visible: bool
    sql_learning_select_label: str
    primary_selection: str | None
    selectai_selection: str | None
    user_selection: str | None


def access_navigation_for_role(role: str | None) -> AccessNavigation:
    """Return the safe initial tab state for an authenticated role."""
    is_admin = role == "admin"
    is_allowed = role in {"admin", "vpd"}
    return AccessNavigation(
        settings_visible=is_admin,
        management_visible=is_admin,
        query_visible=is_allowed,
        selectai_visible=is_allowed,
        chat_visible=is_admin,
        developer_features_visible=is_admin,
        user_features_visible=is_allowed,
        sql_learning_schema_setup_visible=is_admin,
        sql_learning_select_label=(
            "2. SELECTの学習（ステップ）"
            if is_admin
            else "1. SELECTの学習（ステップ）"
        ),
        primary_selection=(
            "settings" if is_admin else "selectai" if is_allowed else None
        ),
        selectai_selection=(
            "developer" if is_admin else "user" if is_allowed else None
        ),
        user_selection="basic" if is_allowed else None,
    )


def do_auth(username, password):
    """ADMINまたは設定済みVPDユーザーを認証する.

    Args:
        username: ユーザー名
        password: パスワード

    Returns:
        bool: 認証が成功した場合True、失敗した場合False
    """
    return authenticate(username, password)
