---
trigger: always_on
alwaysApply: true
---

## 1. プロジェクト全体のコーディング規約

**タイプ:** Always Apply

### ドキュメント
- すべてのモジュール、クラス、関数には日本語のdocstringを記述する
- docstringフォーマット: 簡潔な説明 + Args + Returns（該当する場合）
- ファイルヘッダーには必ずモジュールの目的を記述する

### 命名規則
- 関数名: スネークケース（例: `build_selectai_tab`, `get_dict_value`）
- 変数名: スネークケース（例: `pool`, `team_name_input`）
- クラス名: パスカルケース（例: `LazyPool`）
- 定数: 大文字スネークケース（例: `ORACLE_CLIENT_LIB_DIR`）

### コード構造
- ファイルの先頭に必要なimportをすべて記述する
- 標準ライブラリ → サードパーティ → ローカルモジュールの順でimportする
- logging を使用してエラーとデバッグ情報を記録する
- 例外処理は具体的なエラーメッセージを含める

### Gradioコンポーネント
- UIコンポーネント変数名: `<用途>_<タイプ>` の形式（例: `team_name_input`, `execute_btn`）
- ボタンのvariant: 主要アクション="primary"
- エラーメッセージ: "❌ " で開始
- 警告メッセージ: "⚠️ " で開始
- 成功メッセージ: "✅ " で開始

---

## 2. Python ベストプラクティス

**タイプ:** Specific Files (`*.py`)

### エラーハンドリング
- データベース操作は必ずtry-exceptブロックで囲む
- pool.acquire()を使用する際はwith文を使用する
- cursor操作もwith文を使用する
- すべての例外をloggerに記録する

### リソース管理
- データベース接続はwith文で自動的にクローズする
- CLOBデータは `.read()` メソッドで読み取る
- 大きなデータセットは `fetchmany(size=N)` を使用する

### 型ヒント
- 関数の引数と戻り値に型ヒントを追加する（可能な限り）
- Optional型を適切に使用する

### コードの可読性
- 関数は1つの責任を持つように設計する
- マジックナンバーを避け、定数または環境変数を使用する
- 複雑なロジックには説明的なコメントを追加する

---

## 3. ユニットテスト生成規約

**タイプ:** Model Decision  
**適用シナリオ:** ユニットテストを生成する、テストコードを書く、テストを追加する

### テストフレームワーク
- pytestを使用する
- テストファイル名: `test_<モジュール名>.py`
- テスト関数名: `test_<機能名>_<シナリオ>`

### テスト構造
- Arrange-Act-Assert パターンを使用する
- モック: データベース接続とOCI APIにはモックを使用する
- pytest.fixtures を活用してテストデータを共有する

### カバレッジ
- 正常系とエラー系の両方をテストする
- 境界値をテストする
- データベース接続エラーのシナリオを含める

### データベース接続のモック例

```python
from unittest.mock import MagicMock, patch
import pytest

@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()
    pool.acquire.return_value.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cursor
    return pool
```

---

## 4. データベース操作の実装規約

**タイプ:** Model Decision  
**適用シナリオ:** データベース操作を実装する、SQLクエリを実行する、Oracle DBに接続する

### 接続管理
- 必ずpool.acquire()をwith文で使用する
- cursor も with文で管理する
- 例外発生時は pool.reset() を検討する

### SQL実行パターン

```python
try:
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT ...", param1=value1)
            rows = cursor.fetchall()
            # 処理
except Exception as e:
    logger.error(f"エラーメッセージ: {e}")
    # エラーハンドリング
```

### CLOB/BLOB処理
- CLOB/BLOB型のデータは `.read()` で読み取る
- 読み取り前に `hasattr(value, 'read')` で確認する

### パラメータバインディング
- SQLインジェクション防止のため、必ず名前付きパラメータを使用する
- 例: `cursor.execute(sql, param_name=value)`

### エラーログ
- すべてのデータベースエラーをloggerに記録する
- エラーメッセージはユーザーフレンドリーな日本語で表示する

---

## 5. Gradio UI 実装規約

**タイプ:** Model Decision  
**適用シナリオ:** Gradio UIを作成する、タブを追加する、UIコンポーネントを実装する

### ファイル構造
- 各タブは `utils/` ディレクトリに `build_<機能名>_tab(pool)` 関数として実装する
- 関数名: `build_<機能名>_tab`
- すべてのタブ関数は `pool` パラメータを受け取る

### UI構造

```python
def build_xxx_tab(pool):
    with gr.Tabs():
        with gr.TabItem(label="タブ名"):
            with gr.Accordion(label="1. 入力", open=True):
                # 入力コンポーネント
            with gr.Accordion(label="2. 結果", open=True):
                # 出力コンポーネント
            # イベントハンドラ
```

### コンポーネント命名
- 入力: `<用途>_input`
- ボタン: `<アクション>_btn`
- 出力: `<用途>_output`, `<用途>_md`, `<用途>_df`

### エラー表示
- gr.Markdown の visible パラメータでエラー表示を制御する
- エラー時: `gr.Markdown(visible=True, value="❌ エラーメッセージ")`
- 成功時: `gr.Markdown(visible=True, value="✅ 成功メッセージ")`

### イベントハンドラ
- 内部関数として定義する（`_<アクション名>` の形式）
- Gradioコンポーネントの更新は `gr.ComponentType(visible=..., value=...)` を返す

---

## 6. セキュリティと環境変数

**タイプ:** Always Apply

### 環境変数
- すべての機密情報は環境変数として管理する
- `.env` ファイルを使用し、`python-dotenv` でロードする
- デフォルト値は `os.environ.get("KEY", "default")` で設定する

### ハードコード禁止
- データベース接続文字列
- APIキー、トークン
- ホスト名、ポート番号（開発用のデフォルト値は可）

### ログ出力
- パスワードやトークンをログに出力しない
- 機密情報をマスクする
- ログレベルを適切に設定する（INFO, WARNING, ERROR）

---

## 7. プロジェクト固有の規約

### Oracle Database 関連
- Oracle Instant Clientの初期化はLinux環境でのみ実行する
- 接続プールは遅延初期化（LazyPool）を使用する
- 接続プールの設定: min=0, max=8, increment=1, timeout=30

### OCI GenAI 統合
- OCI SDK (`oci` パッケージ) を使用する
- GenAI関連の設定は環境変数から読み込む
- エンベッディングとチャット機能を分離する

### Gradio テーマとスタイル
- フォント: Noto Sans JP, Noto Sans SC, Roboto の順で指定する
- カスタムCSSは `utils/css_util.py` で管理する
- テーマ: Gradio Default テーマをベースにする

### ファイル構成
```
No.1-SQL-Assist/
├── main.py              # アプリケーションエントリーポイント
├── utils/               # ユーティリティモジュール
│   ├── auth_util.py     # 認証関連
│   ├── common_util.py   # 共通関数
│   ├── css_util.py      # スタイル定義
│   ├── oci_util.py      # OCI GenAI 統合
│   ├── chat_util.py     # チャット機能
│   ├── query_util.py    # クエリ実行
│   ├── selectai_util.py # SelectAI 機能
│   ├── selectai_agent_util.py # SelectAI Agent
│   └── management_util.py # データベース管理
├── models/              # メタデータとインデックス
├── profiles/            # プロファイル設定
└── requirements.txt    # 依存パッケージ
```

---

## 8. コミットとバージョン管理

### コミットメッセージ
- 日本語または英語で記述する
- 形式: `[タイプ] 簡潔な説明`
- タイプ例: `[追加]`, `[修正]`, `[更新]`, `[削除]`

### ブランチ戦略
- main: 本番用安定版
- develop: 開発用ブランチ
- feature/*: 機能追加用ブランチ

