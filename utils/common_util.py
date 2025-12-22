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


def remove_comments(sql_str: str) -> str:
    """SQLからコメントを除去する.

    Args:
        sql_str (str): コメントを除去するSQL

    Returns:
        str: コメントが除去されたSQL
    """
    if not sql_str:
        return ""
        
    # 行単位で処理
    lines = sql_str.split('\n')
    result_lines = []
    
    for line in lines:
        # '--'が含まれるか確認
        if '--' in line:
            # '--'の位置を探す
            # ただし、文字列リテラル内の'--'は無視する必要がある
            # 簡易的な実装として、シングルクォートの外にある'--'以降を削除する
            
            in_quote = False
            comment_start = -1
            
            for i, char in enumerate(line):
                if char == "'":
                    in_quote = not in_quote
                elif char == '-' and i + 1 < len(line) and line[i+1] == '-' and not in_quote:
                    comment_start = i
                    break
            
            if comment_start != -1:
                line = line[:comment_start]
                
        result_lines.append(line)
        
    return '\n'.join(result_lines)
