"""認証ユーティリティモジュール.

このモジュールは、OCI、OpenAI、Azure OpenAIなどの
各種サービスの認証情報を設定するための関数を提供します。
"""

import os
import re

def do_auth(username, password):
    """データベース接続文字列を使用してユーザー認証を行う.

    Args:
        username: ユーザー名
        password: パスワード

    Returns:
        bool: 認証が成功した場合True、失敗した場合False
    """
    dsn = os.environ.get("ORACLE_26AI_CONNECTION_STRING", "")
    pattern = r"^([^/]+)/([^@]+)@"
    match = re.match(pattern, dsn)

    if match:
        if username.lower() == match.group(1).lower() and password == match.group(2):
            return True
    return False
