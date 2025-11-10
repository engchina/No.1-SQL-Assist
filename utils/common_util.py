"""共通ユーティリティモジュール.

プロジェクト全体で使用される共通の関数を提供します。
"""


def get_dict_value(dictionary, key, default_value=None):
    """辞書から値を安全に取得する.

    Args:
        dictionary (dict): 値を取得する辞書
        key: 辞書で検索するキー
        default_value: キーが見つからない場合に返す値（デフォルト: None）

    Returns:
        キーが存在する場合はその値、存在しない場合はdefault_value
    """
    try:
        return dictionary[key]
    except KeyError:
        return default_value
