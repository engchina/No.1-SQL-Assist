"""Management utility module for SQL Assist.

This module provides UI components for database management functions including
Table Management, View Management, and Data Management.
"""

import logging
import traceback
import re
from datetime import datetime
from dateutil import parser as dateutil_parser

import gradio as gr
import pandas as pd
from utils.common_util import remove_comments

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _convert_to_date(value):
    """複数の日付フォーマットに対応した柔軟な日付変換.
    
    Args:
        value: 変換する値（文字列、数値、datetime等）
        
    Returns:
        datetime: 変換された日付オブジェクト、またはNone
    """
    if value is None or pd.isna(value):
        return None
    
    # 既にdatetimeオブジェクトの場合
    if isinstance(value, datetime):
        return value
    
    # 数値の場合（Excelのシリアル日付など）
    if isinstance(value, (int, float)):
        try:
            # Excelのシリアル日付形式（1900-01-01からの日数）
            if 1 <= value <= 2958465:  # 1900-01-01 ～ 9999-12-31
                # Excelの1900年問題を考慮
                base_date = datetime(1899, 12, 30)
                return base_date + pd.Timedelta(days=value)
        except Exception:
            pass
    
    # 文字列の場合
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        
        # マイクロ秒を含む日付フォーマットを正規化（Oracle形式など）
        # 例: 1900/01/01 00:00:00.000000000 → 1900/01/01 00:00:00
        normalized_value = re.sub(r'(\d{4}[/-]\d{1,2}[/-]\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2})\.\d+', r'\1', value)
        
        # 1桁の月/日を0パディング（例: 1900/1/1 → 1900/01/01）
        def _normalize_date_parts(match):
            year = match.group(1)
            sep1 = match.group(2)
            month = match.group(3).zfill(2)
            sep2 = match.group(4)
            day = match.group(5).zfill(2)
            rest = match.group(6) if match.group(6) else ''
            return f"{year}{sep1}{month}{sep2}{day}{rest}"
        
        normalized_value = re.sub(
            r'(\d{4})([/-])(\d{1,2})([/-])(\d{1,2})(.*)',
            _normalize_date_parts,
            normalized_value
        )
        
        # 1桁の時間を0パディング（例: 1900/01/01 0:00 → 1900/01/01 00:00）
        def _normalize_time_parts(match):
            date_part = match.group(1)
            hour = match.group(2).zfill(2)
            minute = match.group(3).zfill(2)
            rest = match.group(4) if match.group(4) else ''
            return f"{date_part} {hour}:{minute}{rest}"
        
        normalized_value = re.sub(
            r'(\d{4}[/-]\d{2}[/-]\d{2})\s+(\d{1,2}):(\d{1,2})(:\d{1,2})?',
            _normalize_time_parts,
            normalized_value
        )
        
        # よくある日付フォーマットパターンを定義
        date_formats = [
            '%Y-%m-%d',           # 2024-12-24
            '%Y/%m/%d',           # 2024/12/24
            '%d-%m-%Y',           # 24-12-2024
            '%d/%m/%Y',           # 24/12/2024
            '%m/%d/%Y',           # 12/24/2024
            '%Y%m%d',             # 20241224
            '%Y-%m-%d %H:%M:%S',  # 2024-12-24 15:30:00
            '%Y/%m/%d %H:%M:%S',  # 2024/12/24 15:30:00
            '%d-%m-%Y %H:%M:%S',  # 24-12-2024 15:30:00
            '%d/%m/%Y %H:%M:%S',  # 24/12/2024 15:30:00
            '%Y-%m-%dT%H:%M:%S',  # 2024-12-24T15:30:00 (ISO)
            '%Y-%m-%d %H:%M',     # 2024-12-24 15:30
            '%Y/%m/%d %H:%M',     # 2024/12/24 15:30
        ]
        
        # 各フォーマットで変換を試行（正規化された値を使用）
        for fmt in date_formats:
            try:
                return datetime.strptime(normalized_value, fmt)
            except ValueError:
                continue
        
        # 上記で失敗した場合、dateutilで柔軟にパース
        try:
            return dateutil_parser.parse(normalized_value, dayfirst=False)
        except Exception:
            pass
    
    # 変換できない場合はNoneを返す（エラーを防ぐため）
    logger.warning(f"日付変換に失敗しました: {value} (type: {type(value)})")
    return None


def get_table_list(pool):
    """Get list of tables for ADMIN user.
    
    Args:
        pool: Oracle database connection pool
        
    Returns:
        pd.DataFrame: DataFrame containing table information
    """
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                sql = """
                SELECT 
                    t.table_name AS "Table Name",
                    NVL(c.comments, ' ') AS "Comments"
                FROM all_tables t
                LEFT JOIN all_tab_comments c ON t.table_name = c.table_name AND t.owner = c.owner
                WHERE t.owner = 'ADMIN' AND t.table_name NOT LIKE '%$%'
                ORDER BY t.table_name
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                if rows:
                    data = []
                    for tn, comment in rows:
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM ADMIN.{tn.upper()}")
                            cnt = cursor.fetchone()[0]
                        except Exception:
                            cnt = 0
                        cm = comment.read() if hasattr(comment, "read") else comment
                        data.append((tn, cnt, cm))
                    df = pd.DataFrame(data, columns=["Table Name", "Rows", "Comments"])
                    logger.info(f"Retrieved {len(df)} tables for ADMIN user")
                    return df
                else:
                    logger.info("No tables found for ADMIN user")
                    return pd.DataFrame(columns=["Table Name", "Rows", "Comments"])
    except Exception as e:
        logger.error(f"Error getting table list: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=["Table Name", "Rows", "Comments"])


def get_table_details(pool, table_name):
    """Get column information for a specific table.
    
    Args:
        pool: Oracle database connection pool
        table_name: Name of the table
        
    Returns:
        tuple: (pd.DataFrame of columns, CREATE TABLE SQL)
    """
    if not table_name:
        return pd.DataFrame(columns=["Column Name", "Data Type", "Nullable", "Comments"]), ""
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Get column information
                col_sql = """
                SELECT 
                    c.column_name AS "Column Name",
                    c.data_type || 
                    CASE 
                        WHEN c.data_type IN ('VARCHAR2', 'CHAR', 'NVARCHAR2', 'NCHAR') 
                        THEN '(' || c.data_length || ')'
                        WHEN c.data_type = 'NUMBER' AND c.data_precision IS NOT NULL
                        THEN '(' || c.data_precision || 
                             CASE WHEN c.data_scale > 0 THEN ',' || c.data_scale ELSE '' END || ')'
                        ELSE ''
                    END AS "Data Type",
                    CASE c.nullable WHEN 'Y' THEN 'YES' ELSE 'NO' END AS "Nullable",
                    NVL(cm.comments, ' ') AS "Comments"
                FROM all_tab_columns c
                LEFT JOIN all_col_comments cm ON c.table_name = cm.table_name 
                    AND c.column_name = cm.column_name AND c.owner = cm.owner
                WHERE c.owner = 'ADMIN' AND c.table_name = :table_name
                ORDER BY c.column_id
                """
                cursor.execute(col_sql, table_name=table_name.upper())
                col_rows = cursor.fetchall()
                
                if col_rows:
                    cleaned_rows = []
                    for r in col_rows:
                        cleaned_rows.append([v.read() if hasattr(v, "read") else v for v in r])
                    columns = [desc[0] for desc in cursor.description]
                    col_df = pd.DataFrame(cleaned_rows, columns=columns)
                else:
                    col_df = pd.DataFrame(columns=["Column Name", "Data Type", "Nullable", "Comments"])
                
                # Get CREATE TABLE statement
                cursor.execute(
                    "SELECT DBMS_METADATA.GET_DDL('TABLE', :table_name, 'ADMIN') FROM DUAL",
                    table_name=table_name.upper()
                )
                ddl_result = cursor.fetchone()
                create_sql = ddl_result[0].read() if ddl_result and ddl_result[0] else ""
                
                # Get table comment
                cursor.execute(
                    """
                    SELECT comments 
                    FROM all_tab_comments 
                    WHERE owner = 'ADMIN' AND table_name = :table_name
                    """,
                    table_name=table_name.upper()
                )
                table_comment_result = cursor.fetchone()
                table_comment_val = table_comment_result[0] if table_comment_result and table_comment_result[0] else None
                table_comment = table_comment_val.read() if hasattr(table_comment_val, "read") else table_comment_val
                
                # Get column comments
                cursor.execute(
                    """
                    SELECT column_name, comments 
                    FROM all_col_comments 
                    WHERE owner = 'ADMIN' AND table_name = :table_name AND comments IS NOT NULL
                    ORDER BY column_name
                    """,
                    table_name=table_name.upper()
                )
                col_comments = cursor.fetchall()
                if col_comments:
                    cleaned_comments = []
                    for cn, cm in col_comments:
                        cm_val = cm.read() if hasattr(cm, "read") else cm
                        cleaned_comments.append((cn, cm_val))
                    col_comments = cleaned_comments
                
                # Append COMMENT statements to the DDL
                if create_sql:
                    create_sql = create_sql.rstrip()
                    if not create_sql.endswith(';'):
                        create_sql += ';'
                    create_sql += '\n'
                    
                    # Add table comment
                    if table_comment:
                        escaped_comment = table_comment.replace("'", "''")
                        create_sql += f"\nCOMMENT ON TABLE {table_name.upper()} IS '{escaped_comment}';\n"
                    
                    # Add column comments
                    if col_comments:
                        for col_name, col_comment in col_comments:
                            escaped_col_comment = col_comment.replace("'", "''")
                            create_sql += f"COMMENT ON COLUMN {table_name.upper()}.{col_name} IS '{escaped_col_comment}';\n"
                
                logger.info(f"Retrieved details for table: {table_name}")
                return col_df, create_sql
                
    except Exception as e:
        logger.error(f"Error getting table details: {e}")
        logger.error(traceback.format_exc())
        logger.error(f"テーブル詳細の取得に失敗しました: {str(e)}")
        return pd.DataFrame(columns=["Column Name", "Data Type", "Nullable", "Comments"]), ""


def drop_table(pool, table_name):
    """Drop a table.
    
    Args:
        pool: Oracle database connection pool
        table_name: Name of the table to drop
        
    Returns:
        str: Result message
    """
    if not table_name:
        logger.error("テーブル名が未指定です")
        return "❌ エラー: テーブル名が指定されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                sql = f"DROP TABLE ADMIN.{table_name.upper()} PURGE"
                cursor.execute(sql)
                conn.commit()
                logger.info(f"Table dropped: {table_name}")
                return f"✅ 成功: テーブル '{table_name}' を削除しました"
    except Exception as e:
        logger.error(f"Error dropping table: {e}")
        logger.error(traceback.format_exc())
        return f"❌ エラー: {str(e)}"


def execute_create_table(pool, create_sql):
    """Execute CREATE TABLE SQL statement(s).
    
    Supports executing multiple SQL statements separated by semicolons,
    including CREATE TABLE, COMMENT ON TABLE, COMMENT ON COLUMN, etc.
    
    Args:
        pool: Oracle database connection pool
        create_sql: Single or multiple SQL statements
        
    Returns:
        str: Result message
    """
    if not create_sql or not create_sql.strip():
        logger.error("CREATE TABLE文が未入力です")
        return "❌ エラー: SQLが入力されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Split SQL statements by semicolon
                sql_statements = [stmt.strip() for stmt in create_sql.split(';') if stmt.strip()]
                
                if not sql_statements:
                    logger.error("CREATE TABLEのSQLが空です")
                    return "❌ エラー: SQLが空です"

                disallowed = []
                for idx, sql_stmt in enumerate(sql_statements, 1):
                    stmt_upper = sql_stmt.strip().upper()
                    is_create_table = stmt_upper.startswith('CREATE TABLE') or bool(re.match(r'^CREATE\s+GLOBAL\s+TEMPORARY\s+TABLE\b', stmt_upper))
                    is_comment = stmt_upper.startswith('COMMENT ON TABLE') or stmt_upper.startswith('COMMENT ON COLUMN')
                    is_drop_table = stmt_upper.startswith('DROP TABLE')
                    if not (is_create_table or is_comment or is_drop_table):
                        disallowed.append((idx, sql_stmt))
                if disallowed:
                    first_idx, first_sql = disallowed[0]
                    error_msg = f"禁止された操作: CREATE TABLE / COMMENT ON / DROP TABLE 以外の文は実行できません。\n文{first_idx}: {first_sql[:100]}..."
                    logger.warning(f"Disallowed statement for table creation: {first_sql[:100]}...")
                    return f"❌ エラー: {error_msg}"
                
                executed_count = 0
                error_messages = []
                
                # Execute each statement
                for idx, sql_stmt in enumerate(sql_statements, 1):
                    try:
                        cursor.execute(sql_stmt)
                        executed_count += 1
                        logger.info(f"Statement {idx}/{len(sql_statements)} executed successfully")
                    except Exception as stmt_error:
                        error_msg = f"文{idx}: {str(stmt_error)}"
                        error_messages.append(error_msg)
                        logger.error(f"Error executing statement {idx}: {stmt_error}")
                        logger.error(f"Failed SQL: {sql_stmt[:100]}...")
                
                # Commit if at least one statement succeeded
                if executed_count > 0:
                    conn.commit()
                    
                # Prepare result message
                if error_messages:
                    result = f"⚠️ 部分的に成功: {executed_count}/{len(sql_statements)}件の文を実行しました\n\nエラー:\n" + "\n".join(error_messages)
                    logger.warning(f"Partial success: {executed_count}/{len(sql_statements)} statements executed")
                    return result
                else:
                    result = f"✅ 成功: {executed_count}件の文をすべて実行しました"
                    logger.info(f"All {executed_count} statements executed successfully")
                    return result
                    
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        logger.error(traceback.format_exc())
        return f"❌ エラー: {str(e)}"


def get_view_list(pool):
    """Get list of views for ADMIN user.
    
    Args:
        pool: Oracle database connection pool
        
    Returns:
        pd.DataFrame: DataFrame containing view information
    """
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Query to get view list with comments
                sql = """
                SELECT 
                    v.view_name AS "View Name",
                    NVL(c.comments, ' ') AS "Comments"
                FROM all_views v
                LEFT JOIN all_tab_comments c ON v.view_name = c.table_name AND v.owner = c.owner
                WHERE v.owner = 'ADMIN' AND v.view_name NOT LIKE '%$%'
                ORDER BY v.view_name
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                if rows:
                    cleaned_rows = []
                    for r in rows:
                        cleaned_rows.append([v.read() if hasattr(v, "read") else v for v in r])
                    columns = [desc[0] for desc in cursor.description]
                    df = pd.DataFrame(cleaned_rows, columns=columns)
                    logger.info(f"Retrieved {len(df)} views for ADMIN user")
                    return df
                else:
                    logger.info("No views found for ADMIN user")
                    return pd.DataFrame(columns=["View Name", "Comments"])
    except Exception as e:
        logger.error(f"Error getting view list: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=["View Name", "Comments"])


def get_view_details(pool, view_name):
    """Get column information for a specific view.
    
    Args:
        pool: Oracle database connection pool
        view_name: Name of the view
        
    Returns:
        tuple: (pd.DataFrame of columns, CREATE VIEW SQL)
    """
    if not view_name:
        return pd.DataFrame(columns=["Column Name", "Data Type", "Nullable", "Comments"]), ""
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Get column information
                col_sql = """
                SELECT 
                    c.column_name AS "Column Name",
                    c.data_type || 
                    CASE 
                        WHEN c.data_type IN ('VARCHAR2', 'CHAR', 'NVARCHAR2', 'NCHAR') 
                        THEN '(' || c.data_length || ')'
                        WHEN c.data_type = 'NUMBER' AND c.data_precision IS NOT NULL
                        THEN '(' || c.data_precision || 
                             CASE WHEN c.data_scale > 0 THEN ',' || c.data_scale ELSE '' END || ')'
                        ELSE ''
                    END AS "Data Type",
                    CASE c.nullable WHEN 'Y' THEN 'YES' ELSE 'NO' END AS "Nullable",
                    NVL(cm.comments, ' ') AS "Comments"
                FROM all_tab_columns c
                LEFT JOIN all_col_comments cm ON c.table_name = cm.table_name 
                    AND c.column_name = cm.column_name AND c.owner = cm.owner
                WHERE c.owner = 'ADMIN' AND c.table_name = :view_name
                ORDER BY c.column_id
                """
                cursor.execute(col_sql, view_name=view_name.upper())
                col_rows = cursor.fetchall()
                
                if col_rows:
                    cleaned_rows = []
                    for r in col_rows:
                        cleaned_rows.append([v.read() if hasattr(v, "read") else v for v in r])
                    columns = [desc[0] for desc in cursor.description]
                    col_df = pd.DataFrame(cleaned_rows, columns=columns)
                else:
                    col_df = pd.DataFrame(columns=["Column Name", "Data Type", "Nullable", "Comments"])
                
                # Check object type first to avoid ORA-31603 if it's not a standard VIEW
                cursor.execute(
                    "SELECT object_type FROM all_objects WHERE owner = 'ADMIN' AND object_name = :name",
                    name=view_name.upper()
                )
                obj_type_row = cursor.fetchone()
                obj_type = obj_type_row[0] if obj_type_row else 'VIEW'
                
                # Get DDL based on object type
                ddl_type = 'VIEW'
                if obj_type == 'MATERIALIZED VIEW':
                    ddl_type = 'MATERIALIZED_VIEW'
                elif obj_type == 'TABLE':
                    ddl_type = 'TABLE'
                
                try:
                    # Get CREATE statement
                    cursor.execute(
                        "SELECT DBMS_METADATA.GET_DDL(:type, :name, 'ADMIN') FROM DUAL",
                        type=ddl_type,
                        name=view_name.upper()
                    )
                    ddl_result = cursor.fetchone()
                    create_sql = ddl_result[0].read() if ddl_result and ddl_result[0] else ""
                except Exception as e:
                    logger.warning(f"Failed to get DDL for {view_name} as {ddl_type}: {e}")
                    create_sql = f"-- DDL取得失敗 (Type: {ddl_type}): {str(e)}"
                
                # Get view comment
                cursor.execute(
                    """
                    SELECT comments 
                    FROM all_tab_comments 
                    WHERE owner = 'ADMIN' AND table_name = :view_name
                    """,
                    view_name=view_name.upper()
                )
                view_comment_result = cursor.fetchone()
                view_comment_val = view_comment_result[0] if view_comment_result and view_comment_result[0] else None
                view_comment = view_comment_val.read() if hasattr(view_comment_val, "read") else view_comment_val
                
                # Get column comments
                cursor.execute(
                    """
                    SELECT column_name, comments 
                    FROM all_col_comments 
                    WHERE owner = 'ADMIN' AND table_name = :view_name AND comments IS NOT NULL
                    ORDER BY column_name
                    """,
                    view_name=view_name.upper()
                )
                col_comments = cursor.fetchall()
                if col_comments:
                    cleaned_comments = []
                    for cn, cm in col_comments:
                        cm_val = cm.read() if hasattr(cm, "read") else cm
                        cleaned_comments.append((cn, cm_val))
                    col_comments = cleaned_comments
                
                # Append COMMENT statements to the DDL
                if create_sql:
                    create_sql = create_sql.rstrip()
                    if not create_sql.endswith(';'):
                        create_sql += ';'
                    create_sql += '\n'
                    
                    if view_comment:
                        escaped_view_comment = view_comment.replace("'", "''")
                        create_sql += f"\nCOMMENT ON TABLE {view_name.upper()} IS '{escaped_view_comment}';\n"
                    
                    # Add column comments
                    if col_comments:
                        for col_name, col_comment in col_comments:
                            escaped_col_comment = col_comment.replace("'", "''")
                            create_sql += f"COMMENT ON COLUMN {view_name.upper()}.{col_name} IS '{escaped_col_comment}';\n"
                
                logger.info(f"Retrieved details for view: {view_name}")
                return col_df, create_sql
                
    except Exception as e:
        logger.error(f"Error getting view details: {e}")
        logger.error(traceback.format_exc())
        logger.error(f"ビュー詳細の取得に失敗しました: {str(e)}")
        return pd.DataFrame(columns=["Column Name", "Data Type", "Nullable", "Comments"]), ""


def get_primary_key_info(pool, object_name):
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                sql = (
                    "SELECT ac.constraint_name, LISTAGG(acc.column_name, ', ') WITHIN GROUP (ORDER BY acc.position) AS columns "
                    "FROM all_constraints ac "
                    "JOIN all_cons_columns acc ON ac.owner = acc.owner AND ac.constraint_name = acc.constraint_name "
                    "WHERE ac.owner = 'ADMIN' AND ac.table_name = :name AND ac.constraint_type = 'P' "
                    "GROUP BY ac.constraint_name"
                )
                cursor.execute(sql, name=object_name.upper())
                rows = cursor.fetchall() or []
                def _clean(v):
                    return v.read() if hasattr(v, "read") else v
                if rows:
                    parts = []
                    for r in rows:
                        cn = _clean(r[0])
                        cols = _clean(r[1])
                        parts.append(f"{cn}: {cols}")
                    return "\n".join(parts)
                return ""
    except Exception:
        return ""


def get_foreign_key_info(pool, object_name):
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                sql = (
                    "SELECT ac.constraint_name, LISTAGG(acc.column_name, ', ') WITHIN GROUP (ORDER BY acc.position) AS src_columns, "
                    "       ac_r.table_name AS ref_table, "
                    "       (SELECT LISTAGG(acc2.column_name, ', ') WITHIN GROUP (ORDER BY acc2.position) "
                    "        FROM all_cons_columns acc2 "
                    "        WHERE acc2.owner = ac_r.owner AND acc2.constraint_name = ac.r_constraint_name) AS ref_columns "
                    "FROM all_constraints ac "
                    "JOIN all_cons_columns acc ON ac.owner = acc.owner AND ac.constraint_name = acc.constraint_name "
                    "JOIN all_constraints ac_r ON ac.owner = ac_r.owner AND ac.r_constraint_name = ac_r.constraint_name "
                    "WHERE ac.owner = 'ADMIN' AND ac.table_name = :name AND ac.constraint_type = 'R' "
                    "GROUP BY ac.constraint_name, ac_r.table_name, ac_r.owner, ac.r_constraint_name"
                )
                cursor.execute(sql, name=object_name.upper())
                rows = cursor.fetchall() or []
                def _clean(v):
                    return v.read() if hasattr(v, "read") else v
                if rows:
                    parts = []
                    for r in rows:
                        cn = _clean(r[0])
                        src = _clean(r[1])
                        rt = _clean(r[2])
                        rc = _clean(r[3])
                        parts.append(f"{cn}: {src} -> {rt}({rc})")
                    return "\n".join(parts)
                return ""
    except Exception:
        return ""


def drop_view(pool, view_name):
    """Drop a view.
    
    Args:
        pool: Oracle database connection pool
        view_name: Name of the view to drop
        
    Returns:
        str: Result message
    """
    if not view_name:
        logger.error("ビュー名が未指定です")
        return "❌ エラー: ビュー名が指定されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                sql = f"DROP VIEW ADMIN.{view_name.upper()}"
                cursor.execute(sql)
                conn.commit()
                logger.info(f"View dropped: {view_name}")
                return f"✅ 成功: ビュー '{view_name}' を削除しました"
    except Exception as e:
        logger.error(f"Error dropping view: {e}")
        logger.error(traceback.format_exc())
        return f"❌ エラー: {str(e)}"


def execute_create_view(pool, create_sql):
    """Execute CREATE VIEW SQL statement(s).
    
    Supports executing multiple SQL statements separated by semicolons,
    including CREATE VIEW, COMMENT ON TABLE, COMMENT ON COLUMN, etc.
    
    Args:
        pool: Oracle database connection pool
        create_sql: Single or multiple SQL statements
        
    Returns:
        str: Result message
    """
    if not create_sql or not create_sql.strip():
        logger.error("CREATE VIEW文が未入力です")
        return "❌ エラー: SQLが入力されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Split SQL statements by semicolon
                sql_statements = [stmt.strip() for stmt in create_sql.split(';') if stmt.strip()]
                
                if not sql_statements:
                    logger.error("CREATE VIEWのSQLが空です")
                    return "❌ エラー: SQLが空です"

                disallowed = []
                for idx, sql_stmt in enumerate(sql_statements, 1):
                    stmt_upper = sql_stmt.strip().upper()
                    is_create_view = (
                        stmt_upper.startswith('CREATE VIEW')
                        or bool(re.match(r'^CREATE\s+OR\s+REPLACE\s+VIEW\b', stmt_upper))
                        or bool(re.match(r'^CREATE\s+OR\s+REPLACE\s+FORCE\s+(?:EDITIONABLE\s+)?VIEW\b', stmt_upper))
                        or bool(re.match(r'^CREATE\s+OR\s+REPLACE\s+EDITIONABLE\s+VIEW\b', stmt_upper))
                    )
                    is_comment = stmt_upper.startswith('COMMENT ON TABLE') or stmt_upper.startswith('COMMENT ON COLUMN')
                    is_drop_view = stmt_upper.startswith('DROP VIEW')
                    if not (is_create_view or is_comment or is_drop_view):
                        disallowed.append((idx, sql_stmt))
                if disallowed:
                    first_idx, first_sql = disallowed[0]
                    error_msg = f"禁止された操作: VIEW作成/削除に関係ない文は実行できません。\n文{first_idx}: {first_sql[:100]}..."
                    logger.warning(f"Disallowed statement for view creation: {first_sql[:100]}...")
                    return f"❌ エラー: {error_msg}"
                
                executed_count = 0
                error_messages = []
                
                # Execute each statement
                for idx, sql_stmt in enumerate(sql_statements, 1):
                    try:
                        cursor.execute(sql_stmt)
                        executed_count += 1
                        logger.info(f"Statement {idx}/{len(sql_statements)} executed successfully")
                    except Exception as stmt_error:
                        error_msg = f"文{idx}: {str(stmt_error)}"
                        error_messages.append(error_msg)
                        logger.error(f"Error executing statement {idx}: {stmt_error}")
                        logger.error(f"Failed SQL: {sql_stmt[:100]}...")
                
                # Commit if at least one statement succeeded
                if executed_count > 0:
                    conn.commit()
                    
                # Prepare result message
                if error_messages:
                    result = f"⚠️ 部分的に成功: {executed_count}/{len(sql_statements)}件の文を実行しました\n\nエラー:\n" + "\n".join(error_messages)
                    logger.warning(f"Partial success: {executed_count}/{len(sql_statements)} statements executed")
                    return result
                else:
                    result = f"✅ 成功: {executed_count}件の文をすべて実行しました"
                    logger.info(f"All {executed_count} statements executed successfully")
                    return result
                    
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        logger.error(traceback.format_exc())
        return f"❌ エラー: {str(e)}"


def get_table_list_for_data(pool):
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                sql = """
                SELECT name
                FROM (
                    SELECT table_name AS name, 0 AS obj_type
                    FROM all_tables 
                    WHERE owner = 'ADMIN' AND table_name NOT LIKE '%$%'
                    UNION ALL
                    SELECT view_name AS name, 1 AS obj_type
                    FROM all_views
                    WHERE owner = 'ADMIN' AND view_name NOT LIKE '%$%'
                )
                GROUP BY name
                ORDER BY MIN(obj_type), name
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                names = [row[0] for row in rows] if rows else []
                logger.info(f"Retrieved {len(names)} tables and views for data management")
                return names
    except Exception as e:
        logger.error(f"Error getting table/view list for data: {e}")
        logger.error(traceback.format_exc())
        return []


def get_table_list_for_upload(pool):
    """Get list of table names for CSV upload dropdown, excluding views and names containing '$'.
    
    Args:
        pool: Oracle database connection pool
        
    Returns:
        list: List of table names eligible for upload
    """
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                sql = """
                SELECT table_name FROM all_tables 
                WHERE owner = 'ADMIN' 
                  AND table_name NOT LIKE '%$%'
                  AND table_name NOT LIKE 'DR$%'
                  AND table_name NOT LIKE 'VECTOR$%'
                ORDER BY 1
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                names = [row[0] for row in rows] if rows else []
                logger.info(f"Retrieved {len(names)} uploadable tables for ADMIN user")
                return names
    except Exception as e:
        logger.error(f"Error getting uploadable table list: {e}")
        logger.error(traceback.format_exc())
        return []


def display_table_data(pool, table_name, limit, where_clause=""):
    """Display data from selected table or view.
    
    Args:
        pool: Oracle database connection pool
        table_name: Name of the table or view
        limit: Maximum number of rows to fetch
        where_clause: Optional WHERE clause for filtering
        
    Returns:
        pd.DataFrame: DataFrame containing table/view data
    """
    if not table_name:
        logger.error("テーブルまたはビューが未選択です")
        return pd.DataFrame()
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Build SQL query
                where_part = f" WHERE {where_clause}" if where_clause and where_clause.strip() else ""
                sql = f"SELECT * FROM ADMIN.{table_name.upper()}{where_part} FETCH FIRST {int(limit)} ROWS ONLY"
                
                logger.info(f"Executing query: {sql}")
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                if rows:
                    cleaned_rows = []
                    for r in rows:
                        cleaned_rows.append([v.read() if hasattr(v, "read") else v for v in r])
                    columns = [desc[0] for desc in cursor.description]
                    df = pd.DataFrame(cleaned_rows, columns=columns)
                    logger.info(f"Retrieved {len(df)} rows from {table_name}")
                    return df
                else:
                    logger.info(f"No data found in {table_name}")
                    return pd.DataFrame()
                    
    except Exception as e:
        logger.error(f"Error displaying table/view data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def upload_csv_data(pool, file, table_name, upload_mode):
    """Upload CSV data to selected table.
    
    Args:
        pool: Oracle database connection pool
        file: Uploaded CSV file
        table_name: Target table name
        upload_mode: Upload mode (INSERT, TRUNCATE, UPDATE)
        
    Returns:
        tuple: (preview_df, result_message)
    """
    if not file:
        logger.error("CSVファイルが未選択です")
        return pd.DataFrame(), "❌ エラー: ファイルが選択されていません"
    
    if not table_name:
        logger.error("アップロード先テーブルが未選択です")
        return pd.DataFrame(), "❌ エラー: テーブルが選択されていません"
    
    try:
        # Read CSV file
        df = pd.read_csv(file.name)
        logger.info(f"CSV file loaded: {len(df)} rows, {len(df.columns)} columns")
        
        if df.empty:
            logger.error("CSVファイルが空です")
            return df, "❌ エラー: CSVファイルにデータがありません"
        
        # Get preview (first 10 rows)
        preview_df = df.head(10)
        
        # Execute upload based on mode
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Get table columns with data types
                cursor.execute(
                    """SELECT column_name, data_type 
                       FROM all_tab_columns 
                       WHERE owner = 'ADMIN' AND table_name = :table_name 
                       ORDER BY column_id""",
                    table_name=table_name.upper()
                )
                table_columns_info = cursor.fetchall()
                table_columns = [row[0] for row in table_columns_info]
                column_types = {row[0]: row[1] for row in table_columns_info}
                
                if not table_columns:
                    return preview_df, f"❌ エラー: テーブル '{table_name}' が見つかりません"
                
                # Match CSV columns to table columns (case-insensitive)
                csv_columns = df.columns.tolist()
                column_mapping = {}
                for csv_col in csv_columns:
                    for tbl_col in table_columns:
                        if csv_col.upper() == tbl_col.upper():
                            column_mapping[csv_col] = tbl_col
                            break
                
                if not column_mapping:
                    return preview_df, "❌ エラー: CSVの列名がテーブルの列名と一致しません"
                
                # Truncate if mode is TRUNCATE
                if upload_mode == "TRUNCATE & INSERT":
                    cursor.execute(f"TRUNCATE TABLE ADMIN.{table_name.upper()}")
                    logger.info(f"Table {table_name} truncated")
                
                # Prepare INSERT statement
                mapped_columns = list(column_mapping.values())
                placeholders = ", ".join([f":{i+1}" for i in range(len(mapped_columns))])
                insert_sql = f"INSERT INTO ADMIN.{table_name.upper()} ({', '.join(mapped_columns)}) VALUES ({placeholders})"
                
                # Insert data
                success_count = 0
                error_count = 0
                error_messages = []
                
                for idx, row in df.iterrows():
                    try:
                        values = []
                        for csv_col in column_mapping.keys():
                            if csv_col not in column_mapping:
                                values.append(None)
                                continue
                            
                            value = row[csv_col]
                            # Convert NaN to None
                            if pd.isna(value):
                                values.append(None)
                                continue
                            
                            # Get target column name and type
                            target_col = column_mapping[csv_col]
                            col_type = column_types.get(target_col, '')
                            
                            # Convert date/timestamp values flexibly
                            if col_type == 'DATE' or col_type.startswith('TIMESTAMP'):
                                converted_value = _convert_to_date(value)
                                values.append(converted_value)
                            else:
                                values.append(value)
                        
                        cursor.execute(insert_sql, values)
                        success_count += 1
                    except Exception as row_error:
                        error_count += 1
                        if error_count <= 5:  # Show first 5 errors
                            # 詳細なエラー情報を提供
                            error_detail = f"行{idx+1}"
                            error_str = str(row_error)
                            
                            # 日付フォーマットエラーの場合、該当する列と値を表示
                            if 'ORA-01861' in error_str or 'ORA-01843' in error_str or 'format string' in error_str.lower():
                                date_cols_info = []
                                for csv_col in column_mapping.keys():
                                    target_col = column_mapping[csv_col]
                                    col_type = column_types.get(target_col, '')
                                    if col_type == 'DATE' or col_type.startswith('TIMESTAMP'):
                                        val = row[csv_col]
                                        date_cols_info.append(f"{target_col}={val}")
                                if date_cols_info:
                                    error_detail += f" [日付列: {', '.join(date_cols_info)}]"
                            
                            error_messages.append(f"{error_detail}: {error_str[:150]}")
                
                # Commit transaction
                conn.commit()
                
                # Prepare result message
                result = f"✅ 成功: {success_count}件のデータを挿入しました"
                if error_count > 0:
                    # 日付エラーかどうかを判定
                    has_date_error = any('ORA-01861' in msg or 'format string' in msg.lower() for msg in error_messages)
                    
                    result += f"\n\n⚠️ 警告: {error_count}件のエラーが発生しました\n" + "\n".join(error_messages)
                    if error_count > 5:
                        result += f"\n... 他 {error_count - 5} 件のエラー"
                    
                    # 日付エラーの場合はヒントを追加
                    if has_date_error:
                        result += "\n\n💡 ヒント: 日付フォーマットエラーが発生しています。以下の対応をお試しください:\n"
                        result += "  1. CSVの日付列を 'YYYY-MM-DD' 形式（例: 2024-12-24）に変換\n"
                        result += "  2. Excelで開いている場合、セルの書式を「文字列」に設定して保存\n"
                        result += "  3. タイムスタンプの場合は 'YYYY-MM-DD HH:MM:SS' 形式を使用"
                
                logger.info(f"CSV upload completed: {success_count} success, {error_count} errors")
                return preview_df, result
                
    except Exception as e:
        logger.error(f"Error uploading CSV: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(), f"❌ エラー: {str(e)}"


def execute_data_sql(pool, sql_statements):
    """Execute multiple DML/DDL SQL statements.
    
    Supports executing multiple SQL statements separated by semicolons,
    including INSERT, UPDATE, DELETE, MERGE, etc.
    SELECT statements are prohibited for security reasons.
    
    Args:
        pool: Oracle database connection pool
        sql_statements: Single or multiple SQL statements
        
    Returns:
        str: Result message
    """
    if not sql_statements or not sql_statements.strip():
        logger.error("データSQLが未入力です")
        return "❌ エラー: SQLが入力されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Split SQL statements by semicolon
                statements = [stmt.strip() for stmt in sql_statements.split(';') if stmt.strip()]
                
                if not statements:
                    logger.error("データSQLの文が空です")
                    return "❌ エラー: SQLが空です"
                
                disallowed = []
                for idx, sql_stmt in enumerate(statements, 1):
                    stmt_upper = sql_stmt.strip().upper()
                    is_insert = stmt_upper.startswith('INSERT')
                    is_update = stmt_upper.startswith('UPDATE')
                    is_delete = stmt_upper.startswith('DELETE')
                    is_merge = stmt_upper.startswith('MERGE')
                    is_truncate = stmt_upper.startswith('TRUNCATE')
                    if not (is_insert or is_update or is_delete or is_merge or is_truncate):
                        disallowed.append((idx, sql_stmt))
                if disallowed:
                    first_idx, first_sql = disallowed[0]
                    error_msg = f"禁止された操作: INSERT, UPDATE, DELETE, MERGE, TRUNCATE 以外の文は実行できません。\n文{first_idx}: {first_sql[:100]}..."
                    logger.warning(f"Disallowed statement for data SQL: {first_sql[:100]}...")
                    return f"❌ エラー: {error_msg}"
                
                executed_count = 0
                error_messages = []
                affected_rows = []
                
                # Execute each statement
                for idx, sql_stmt in enumerate(statements, 1):
                    try:
                        cursor.execute(sql_stmt)
                        rows_affected = cursor.rowcount
                        affected_rows.append(rows_affected)
                        executed_count += 1
                        logger.info(f"Statement {idx}/{len(statements)} executed successfully, {rows_affected} rows affected")
                    except Exception as stmt_error:
                        error_msg = f"文{idx}: {str(stmt_error)}"
                        error_messages.append(error_msg)
                        logger.error(f"Error executing statement {idx}: {stmt_error}")
                        logger.error(f"Failed SQL: {sql_stmt[:100]}...")
                
                # Commit if at least one statement succeeded
                if executed_count > 0:
                    conn.commit()
                    
                # Prepare result message
                if error_messages:
                    total_rows = sum(affected_rows)
                    result = f"⚠️ 部分的に成功: {executed_count}/{len(statements)}件の文を実行しました（{total_rows}行に影響）\n\nエラー:\n" + "\n".join(error_messages)
                    logger.warning(f"Partial success: {executed_count}/{len(statements)} statements executed")
                    return result
                else:
                    total_rows = sum(affected_rows)
                    result = f"✅ 成功: {executed_count}件の文をすべて実行しました（{total_rows}行に影響）"
                    logger.info(f"All {executed_count} statements executed successfully")
                    return result
                    
    except Exception as e:
        logger.error(f"Error executing data SQL: {e}")
        logger.error(traceback.format_exc())
        return f"❌ エラー: {str(e)}"


def execute_comment_sql(pool, sql_statements):
    """Execute COMMENT ON SQL statement(s).
    
    Supports executing multiple COMMENT statements separated by semicolons.
    COMMENT ON TABLE / COLUMN / MATERIALIZED VIEW / VIEW を許可。
    
    Args:
        pool: Oracle database connection pool
        sql_statements: Single or multiple SQL statements
        
    Returns:
        str: Result message
    """
    if not sql_statements or not sql_statements.strip():
        logger.error("COMMENT文が未入力です")
        return "❌ エラー: SQLが入力されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                statements = [stmt.strip() for stmt in sql_statements.split(';') if stmt.strip()]
                if not statements:
                    logger.error("COMMENTのSQLが空です")
                    return "❌ エラー: SQLが空です"
                disallowed = []
                for idx, sql_stmt in enumerate(statements, 1):
                    stmt_upper = sql_stmt.strip().upper()
                    is_comment_table = stmt_upper.startswith('COMMENT ON TABLE')
                    is_comment_column = stmt_upper.startswith('COMMENT ON COLUMN')
                    is_comment_mview = stmt_upper.startswith('COMMENT ON MATERIALIZED VIEW')
                    is_comment_view = stmt_upper.startswith('COMMENT ON VIEW')
                    if not (is_comment_table or is_comment_column or is_comment_mview or is_comment_view):
                        disallowed.append((idx, sql_stmt))
                if disallowed:
                    first_idx, first_sql = disallowed[0]
                    error_msg = f"禁止された操作: COMMENT ON TABLE/COLUMN/MATERIALIZED VIEW/VIEW 以外の文は実行できません。\n文{first_idx}: {first_sql[:100]}..."
                    logger.warning(f"Disallowed statement for comments: {first_sql[:100]}...")
                    return f"❌ エラー: {error_msg}"
                executed_count = 0
                error_messages = []
                for idx, sql_stmt in enumerate(statements, 1):
                    try:
                        cursor.execute(sql_stmt)
                        executed_count += 1
                        logger.info(f"Statement {idx}/{len(statements)} executed successfully")
                    except Exception as stmt_error:
                        error_msg = f"文{idx}: {str(stmt_error)}"
                        error_messages.append(error_msg)
                        logger.error(f"Error executing statement {idx}: {stmt_error}")
                        logger.error(f"Failed SQL: {sql_stmt[:100]}...")
                if executed_count > 0:
                    conn.commit()
                if error_messages:
                    result = f"⚠️ 部分的に成功: {executed_count}/{len(statements)}件の文を実行しました\n\nエラー:\n" + "\n".join(error_messages)
                    logger.warning(f"Partial success: {executed_count}/{len(statements)} statements executed")
                    return result
                else:
                    result = f"✅ 成功: {executed_count}件の文をすべて実行しました"
                    logger.info(f"All {executed_count} statements executed successfully")
                    return result
    except Exception as e:
        logger.error(f"Error executing comment SQL: {e}")
        logger.error(traceback.format_exc())
        return f"❌ エラー: {str(e)}"

def execute_annotation_sql(pool, sql_statements):
    if not sql_statements or not str(sql_statements).strip():
        logger.error("アノテーション文が未入力です")
        return "❌ エラー: SQLが入力されていません"
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                statements = [stmt.strip() for stmt in str(sql_statements).split(';') if stmt.strip()]
                if not statements:
                    logger.error("アノテーションのSQLが空です")
                    return "❌ エラー: SQLが空です"
                disallowed = []
                for idx, sql_stmt in enumerate(statements, 1):
                    up = sql_stmt.strip().upper()
                    has_annotations = ' ANNOTATIONS ' in re.sub(r"\s+", ' ', up)
                    is_alter_table = up.startswith('ALTER TABLE ')
                    is_alter_view = up.startswith('ALTER VIEW ')
                    norm = re.sub(r"\s+", ' ', sql_stmt.strip())
                    if has_annotations and is_alter_table:
                        if ' MODIFY ' in norm.upper():
                            m = re.search(r"MODIFY\s*\(([^\)]*)\)", norm, flags=re.IGNORECASE)
                            if m:
                                inner = m.group(1)
                                if ' ANNOTATIONS ' not in re.sub(r"\s+", ' ', inner.upper()):
                                    disallowed.append((idx, sql_stmt))
                                    continue
                        elif ' ANNOTATIONS ' in norm.upper():
                            pass
                        else:
                            disallowed.append((idx, sql_stmt))
                            continue
                    elif has_annotations and is_alter_view:
                        if ' MODIFY ' in norm.upper():
                            disallowed.append((idx, sql_stmt))
                            continue
                        if ' ANNOTATIONS ' in norm.upper():
                            pass
                        else:
                            disallowed.append((idx, sql_stmt))
                            continue
                    if not (has_annotations and (is_alter_table or is_alter_view)):
                        disallowed.append((idx, sql_stmt))
                if disallowed:
                    first_idx, first_sql = disallowed[0]
                    msg = f"禁止された操作: 無効なアノテーション文は実行できません。許可: ALTER TABLE MODIFY ... ANNOTATIONS / ALTER TABLE ANNOTATIONS / ALTER VIEW ANNOTATIONS\n文{first_idx}: {first_sql[:100]}..."
                    logger.warning(f"Disallowed statement for annotations: {first_sql[:100]}...")
                    return f"❌ エラー: {msg}"
                executed_count = 0
                errors = []
                for idx, sql_stmt in enumerate(statements, 1):
                    try:
                        cursor.execute(sql_stmt)
                        executed_count += 1
                        logger.info(f"Statement {idx}/{len(statements)} executed successfully")
                    except Exception as e:
                        em = f"文{idx}: {str(e)}"
                        errors.append(em)
                        logger.error(f"Error executing statement {idx}: {e}")
                        logger.error(f"Failed SQL: {sql_stmt[:100]}...")
                if executed_count > 0:
                    conn.commit()
                if errors:
                    result = f"⚠️ 部分的に成功: {executed_count}/{len(statements)}件の文を実行しました\n\nエラー:\n" + "\n".join(errors)
                    logger.warning(f"Partial success: {executed_count}/{len(statements)} statements executed")
                    return result
                else:
                    result = f"✅ 成功: {executed_count}件の文をすべて実行しました"
                    logger.info(f"All {executed_count} statements executed successfully")
                    return result
    except Exception as e:
        logger.error(f"Error executing annotation SQL: {e}")
        logger.error(traceback.format_exc())
        return f"❌ エラー: {str(e)}"


def build_management_tab(pool):
    """Build the Management Function tab with three sub-functions.
    
    Args:
        pool: Oracle database connection pool
    """
    with gr.Tabs():
        # Table Management Tab
        with gr.TabItem(label="テーブルの管理"):
            # Feature 1: Table List
            with gr.Accordion(label="1. テーブル一覧", open=True):
                table_refresh_btn = gr.Button("テーブル一覧を取得（時間がかかる場合があります）", variant="primary")
                table_refresh_status = gr.Markdown(visible=False)
                table_list_df = gr.Dataframe(
                    label="テーブル一覧（件数: 0）",
                    interactive=False,
                    wrap=True,
                    value=pd.DataFrame(columns=["Table Name", "Rows", "Comments"]),
                    visible=False,
                    max_height=300,
                )
            
            # Feature 2: Table Details and Drop
            with gr.Accordion(label="2. テーブル詳細と削除", open=True):
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("選択されたテーブル名*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                selected_table_name = gr.Textbox(
                                    show_label=False,
                                    interactive=False,
                                    container=False,
                                )
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                table_drop_btn = gr.Button("選択したテーブルを削除", variant="stop")
                with gr.Row():
                    table_drop_result = gr.Markdown(visible=False)
                
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("### CREATE TABLE SQL")
                        table_ddl_text = gr.Textbox(
                            label="DDL",
                            show_label=False,
                            lines=15,
                            max_lines=15,
                            interactive=False,
                            show_copy_button=True,
                            autoscroll=False,
                            container=False,
                        )
                    with gr.Column(scale=5):
                        gr.Markdown("### 列情報")
                        table_columns_df = gr.Dataframe(
                            label="列情報",
                            show_label=False,
                            interactive=False,
                            wrap=True,
                            visible=False,
                        )
            
            # Feature 3: Create Table
            with gr.Accordion(label="3. テーブル作成", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("ℹ️ 複数の文をセミコロンで区切って入力可能")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("SQLファイル", elem_classes="input-label")
                    with gr.Column(scale=5):
                        table_sql_file_input = gr.File(
                            show_label=False,
                            file_types=[".sql", ".txt"],
                            type="filepath",
                        )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("CREATE TABLE SQL*", elem_classes="input-label")
                    with gr.Column(scale=5):
                        create_table_sql = gr.Textbox(
                            show_label=False,
                            placeholder="CREATE TABLE文を入力してください\n例:\nCREATE TABLE test_table (\n  id NUMBER PRIMARY KEY,\n  name VARCHAR2(100)\n);\n\nCOMMENT ON TABLE test_table IS 'テストテーブル';\nCOMMENT ON COLUMN test_table.id IS 'ID';\nCOMMENT ON COLUMN test_table.name IS '名称';",
                            lines=10,
                            max_lines=15,
                            show_copy_button=True,
                            container=False,
                            autoscroll=False,
                        )
                
                with gr.Row():
                    with gr.Column():
                        clear_sql_btn = gr.Button("クリア", variant="secondary")
                    with gr.Column():
                        create_table_btn = gr.Button("テーブルを作成", variant="primary")
                
                with gr.Row():
                    create_table_result = gr.Markdown(visible=False)

                with gr.Accordion(label="AI分析と処理", open=True):
                    with gr.Row():
                        with gr.Column(scale=5):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("モデル*", elem_classes="input-label")
                                with gr.Column(scale=5):
                                    table_ai_model_input = gr.Dropdown(
                                        show_label=False,
                                        choices=[
                                            "xai.grok-code-fast-1",
                                            "xai.grok-3",
                                            "xai.grok-3-fast",
                                            "xai.grok-4",
                                            "xai.grok-4-fast-non-reasoning",
                                            "meta.llama-4-scout-17b-16e-instruct",
                                            "gpt-4o",
                                            "gpt-5.1",
                                        ],
                                        value="xai.grok-code-fast-1",
                                        interactive=True,
                                        container=False,
                                    )
                        with gr.Column(scale=5):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    table_ai_analyze_btn = gr.Button("AI分析", variant="primary")                       
                    with gr.Row():
                        table_ai_status_md = gr.Markdown(visible=False)
                    with gr.Row():
                        table_ai_result_md = gr.Markdown(visible=False)
            
            # Event handlers
            def on_table_select(evt: gr.SelectData, current_df):
                """Handle table row selection.
                
                Always extracts the table name from the first column (Table Name),
                regardless of which column in the row was clicked.
                
                Args:
                    evt: Gradio SelectData event
                    current_df: Current dataframe value
                """
                try:
                    row_index = evt.index[0]
                    logger.info(f"Row clicked: {row_index}, Value clicked: {evt.value}")
                    
                    if row_index >= 0 and current_df is not None:
                        logger.info(f"Dataframe type: {type(current_df)}")
                        
                        # Convert to DataFrame if it's a dict (Gradio format)
                        if isinstance(current_df, dict):
                            logger.info(f"Dict keys: {current_df.keys()}")
                            # Gradio returns dict with 'data' and 'headers' keys
                            if 'data' in current_df:
                                data = current_df['data']
                                headers = current_df.get('headers', [])
                                logger.info(f"Data length: {len(data)}, Headers: {headers}")
                                current_df = pd.DataFrame(data, columns=headers)
                            else:
                                # Try direct conversion with orientation
                                try:
                                    current_df = pd.DataFrame.from_dict(current_df, orient='tight')
                                except Exception as e:
                                    logger.warning(f"Failed to convert with orient='tight': {e}")
                                    current_df = pd.DataFrame(current_df)
                        
                        logger.info(f"DataFrame shape: {current_df.shape}")
                        logger.info(f"DataFrame columns: {list(current_df.columns)}")
                        
                        if len(current_df) > row_index:
                            # Always get the table name from the first column (index 0)
                            table_name = str(current_df.iloc[row_index, 0])
                            logger.info(f"Table selected from row {row_index}: {table_name}")
                            col_df, ddl = get_table_details(pool, table_name)
                            if isinstance(col_df, pd.DataFrame) and not col_df.empty:
                                return table_name, gr.Dataframe(visible=True, value=col_df), ddl
                            else:
                                return table_name, gr.Dataframe(visible=False, value=pd.DataFrame()), ddl
                        else:
                            logger.warning(f"Row index {row_index} out of bounds, dataframe has {len(current_df)} rows")
                except Exception as e:
                    logger.error(f"Error in on_table_select: {e}")
                    logger.error(traceback.format_exc())
                
                return "", gr.Dataframe(visible=False, value=pd.DataFrame()), ""
            
            def refresh_table_list():
                try:
                    logger.info("テーブル一覧を取得ボタンがクリックされました")
                    yield gr.Markdown(value="⏳ テーブル一覧を取得中...", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Table Name", "Rows", "Comments"]), label="テーブル一覧（件数: 0）")
                    df = get_table_list(pool)
                    cnt = len(df) if isinstance(df, pd.DataFrame) else 0
                    status_text = "✅ 取得完了（データなし）" if cnt == 0 else "✅ 取得完了"
                    yield gr.Markdown(value=status_text, visible=True), gr.Dataframe(value=df, visible=True, label=f"テーブル一覧（件数: {cnt}）")
                except Exception as e:
                    yield gr.Markdown(value=f"❌ 取得に失敗しました: {str(e)}", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Table Name", "Rows", "Comments"]), label="テーブル一覧（件数: 0）")
            
            def drop_selected_table(table_name):
                """Drop the selected table and refresh list."""
                yield (
                    gr.Markdown(visible=True, value="⏳ テーブルを削除中..."),
                    gr.Textbox(value=str(table_name or ""), autoscroll=False),
                    gr.Dataframe(visible=False, value=pd.DataFrame()),
                    gr.Textbox(value="", autoscroll=False),
                )
                result = drop_table(pool, table_name)
                status_md = gr.Markdown(visible=True, value=result)
                yield (
                    status_md,
                    gr.Textbox(value="", autoscroll=False),
                    gr.Dataframe(value=pd.DataFrame()),
                    gr.Textbox(value="", autoscroll=False),
                )
            
            def execute_create(sql):
                """Execute CREATE TABLE and refresh list."""
                sql_no_comment = remove_comments(sql)
                yield gr.Markdown(visible=True, value="⏳ テーブル作成を実行中...")
                result = execute_create_table(pool, sql_no_comment)
                status_md = gr.Markdown(visible=True, value=result)
                yield status_md
            
            def clear_sql():
                """Clear the SQL input."""
                return ""
            
            def load_table_sql_file(file_path):
                """
                SQLファイルを読み込み、テキストボックスに表示する.
                
                Args:
                    file_path: アップロードされたファイルのパス
                
                Returns:
                    ファイルの内容(文字列)
                """
                if not file_path:
                    return ""
                
                try:
                    # 複数のエンコーディングで試行
                    encodings = ['utf-8', 'shift_jis', 'cp932', 'latin1', 'euc-jp']
                    content = None
                    
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            logger.info(f"SQLファイルを{encoding}で読み込みました: {file_path}")
                            break
                        except (UnicodeDecodeError, UnicodeError):
                            continue
                    
                    if content is None:
                        logger.error(f"SQLファイルの読み込みに失敗しました: {file_path}")
                        return "❌ エラー: ファイルの読み込みに失敗しました。エンコーディングを確認してください。"
                    
                    return content
                    
                except Exception as e:
                    logger.error(f"SQLファイルの読み込み中にエラーが発生しました: {e}")
                    logger.error(traceback.format_exc())
                    return f"❌ エラー: {str(e)}"
            
            # Wire up events
            table_refresh_btn.click(
                fn=refresh_table_list,
                outputs=[table_refresh_status, table_list_df]
            )
            
            table_list_df.select(
                fn=on_table_select,
                inputs=[table_list_df],
                outputs=[selected_table_name, table_columns_df, table_ddl_text]
            )
            
            table_drop_btn.click(
                fn=drop_selected_table,
                inputs=[selected_table_name],
                outputs=[table_drop_result, selected_table_name, table_columns_df, table_ddl_text]
            ).then(
                fn=refresh_table_list,
                outputs=[table_refresh_status, table_list_df]
            )
            
            create_table_btn.click(
                fn=execute_create,
                inputs=[create_table_sql],
                outputs=[create_table_result]
            ).then(
                fn=refresh_table_list,
                outputs=[table_refresh_status, table_list_df]
            )
            
            clear_sql_btn.click(
                fn=clear_sql,
                outputs=[create_table_sql]
            )

            table_sql_file_input.change(
                fn=load_table_sql_file,
                inputs=[table_sql_file_input],
                outputs=[create_table_sql],
            )

            async def _table_ai_analyze_async(model_name, create_sql_text, exec_result_text):
                if not model_name.startswith("gpt-"):
                    from utils.chat_util import get_oci_region, get_compartment_id
                    region = get_oci_region()
                    compartment_id = get_compartment_id()
                    if not region or not compartment_id:
                        return gr.Markdown(visible=True, value="ℹ️ OCI設定が不足しています")
                try:
                    
                    sql_part = str(create_sql_text or "").strip()
                    result_part = str(exec_result_text or "").strip()
                    prompt = (
                        "以下のSQLと実行結果を分析してください。出力は次の3点に限定します。\n"
                        "1) エラー原因（該当する場合）\n"
                        "2) 解決方法（修正案や具体的手順）\n"
                        "3) 簡潔な結論（不要な詳細は省略）\n\n"
                        + ("SQL:\n```sql\n" + sql_part + "\n```\n" if sql_part else "")
                        + ("実行結果:\n" + result_part + "\n" if result_part else "")
                    )
                    
                    messages = [
                        {"role": "system", "content": "あなたはシニアDBエンジニアです。SQLと実行結果の故障診断に特化し、エラー原因と実行可能な修復策のみを簡潔に提示してください。不要な詳細は出力しないでください。"},
                        {"role": "user", "content": prompt},
                    ]

                    if model_name.startswith("gpt-"):
                        from openai import AsyncOpenAI
                        client = AsyncOpenAI()
                        # Use Chat Completions API instead of Responses API to avoid 404 errors
                        resp = await client.chat.completions.create(model=model_name, messages=messages)
                        text = ""
                        if getattr(resp, "choices", None):
                            msg = resp.choices[0].message
                            text = msg.content if hasattr(msg, "content") else ""
                    else:
                        from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                        client = AsyncOciOpenAI(
                            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                            auth=OciUserPrincipalAuth(),
                            compartment_id=compartment_id,
                        )
                        resp = await client.chat.completions.create(model=model_name, messages=messages)
                        text = ""
                        if getattr(resp, "choices", None):
                            msg = resp.choices[0].message
                            text = msg.content if hasattr(msg, "content") else ""
                            
                    return gr.Markdown(visible=True, value=text or "分析結果が空です")
                except Exception as e:
                    return gr.Markdown(visible=True, value=f"❌ エラー: {e}")

            def table_ai_analyze(model_name, create_sql_text, exec_result_text):
                import asyncio
                # 必須入力項目のチェック
                if not model_name or not str(model_name).strip():
                    yield gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Markdown(visible=False)
                    return
                if not create_sql_text or not str(create_sql_text).strip():
                    yield gr.Markdown(visible=True, value="⚠️ CREATE TABLE SQLを入力してください"), gr.Markdown(visible=False)
                    return
                if not exec_result_text or not str(exec_result_text).strip():
                    yield gr.Markdown(visible=True, value="⚠️ 実行結果がありません。先にテーブル作成を実行してください"), gr.Markdown(visible=False)
                    return
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Markdown(visible=False)
                    result_md = loop.run_until_complete(_table_ai_analyze_async(model_name, create_sql_text, exec_result_text))
                    yield gr.Markdown(visible=True, value="✅ 完了"), result_md
                except Exception as e:
                    yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Markdown(visible=False)
                finally:
                    loop.close()

            table_ai_analyze_btn.click(
                fn=table_ai_analyze,
                inputs=[table_ai_model_input, create_table_sql, create_table_result],
                outputs=[table_ai_status_md, table_ai_result_md],
            )
        
        # View Management Tab
        with gr.TabItem(label="ビューの管理"):
            # Feature 1: View List
            with gr.Accordion(label="1. ビュー一覧", open=True):
                view_refresh_btn = gr.Button("ビュー一覧を取得（時間がかかる場合があります）", variant="primary")
                view_refresh_status = gr.Markdown(visible=False)
                view_list_df = gr.Dataframe(
                    label="ビュー一覧（件数: 0）",
                    interactive=False,
                    wrap=True,
                    value=pd.DataFrame(columns=["View Name", "Comments"]),
                    visible=False,
                )
            
            # Feature 2: View Details and Drop
            with gr.Accordion(label="2. ビュー詳細と削除", open=True):
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("選択されたビュー名*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                selected_view_name = gr.Textbox(
                                    show_label=False,
                                    interactive=False,
                                    container=False,
                                    autoscroll=False,
                                )
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                view_drop_btn = gr.Button("選択したビューを削除", variant="stop")
                   
                with gr.Row():
                    view_drop_result = gr.Markdown(visible=False)
                
                with gr.Row():                   
                    with gr.Column():
                        gr.Markdown("### CREATE VIEW SQL")
                        view_ddl_text = gr.Textbox(
                            label="DDL",
                            show_label=False,
                            lines=15,
                            max_lines=15,
                            interactive=False,
                            show_copy_button=True,
                            autoscroll=False,
                            container=False,
                        )
                    with gr.Column():
                        gr.Markdown("### 列情報")
                        view_columns_df = gr.Dataframe(
                            label="列情報",
                            show_label=False,
                            interactive=False,
                            wrap=True,
                            visible=False,
                        )

                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("モデル*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                view_analysis_model_input = gr.Dropdown(
                                    show_label=False,
                                    choices=[
                                        "xai.grok-code-fast-1",
                                        "xai.grok-3",
                                        "xai.grok-3-fast",
                                        "xai.grok-4",
                                        "xai.grok-4-fast-non-reasoning",
                                        "meta.llama-4-scout-17b-16e-instruct",
                                        "gpt-4o",
                                        "gpt-5.1",
                                    ],
                                    value="xai.grok-code-fast-1",
                                    interactive=True,
                                    container=False,
                                )
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                view_ai_extract_btn = gr.Button("AIでJOIN/WHERE条件を抽出", variant="primary")
                with gr.Row():
                    view_ai_extract_status_md = gr.Markdown(visible=False)
                
                with gr.Row():
                    with gr.Column():
                        view_join_text = gr.Textbox(
                            label="結合条件",
                            lines=6,
                            max_lines=15,
                            interactive=False,
                            show_copy_button=True,
                            autoscroll=False,
                        )
                    with gr.Column():
                        view_where_text = gr.Textbox(
                            label="Where条件",
                            lines=6,
                            max_lines=15,
                            interactive=False,
                            show_copy_button=True,
                            autoscroll=False,
                        )
            
            # Feature 3: Create View
            with gr.Accordion(label="3. ビュー作成", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("ℹ️ 複数の文をセミコロンで区切って入力可能", elem_classes="input-label")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("SQLファイル", elem_classes="input-label")
                    with gr.Column(scale=5):
                        view_sql_file_input = gr.File(
                            show_label=False,
                            file_types=[".sql", ".txt"],
                            type="filepath",
                        )
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("CREATE VIEW SQL*", elem_classes="input-label")
                    with gr.Column(scale=5):
                        create_view_sql = gr.Textbox(
                            show_label=False,
                            placeholder="CREATE VIEW文を入力してください\n例:\nCREATE VIEW test_view AS\n\nCOMMENT ON TABLE test_view IS 'テストビュー';\nCOMMENT ON COLUMN test_view.id IS 'ID';\nCOMMENT ON COLUMN test_view.name IS '名称';",
                            lines=10,
                            max_lines=15,
                            show_copy_button=True,
                            container=False,
                            autoscroll=False,
                        )
                
                with gr.Row():
                    with gr.Column():
                        clear_view_sql_btn = gr.Button("クリア", variant="secondary")
                    with gr.Column():
                        create_view_btn = gr.Button("ビューを作成", variant="primary")
                with gr.Row():                
                    create_view_result = gr.Markdown(visible=False)

                with gr.Accordion(label="AI分析と処理", open=True):
                    with gr.Row():
                        with gr.Column(scale=5):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("モデル*", elem_classes="input-label")
                                with gr.Column(scale=5):
                                    view_ai_model_input = gr.Dropdown(
                                        show_label=False,
                                        choices=[
                                            "xai.grok-code-fast-1",
                                            "xai.grok-3",
                                            "xai.grok-3-fast",
                                            "xai.grok-4",
                                            "xai.grok-4-fast-non-reasoning",
                                        "meta.llama-4-scout-17b-16e-instruct",
                                        "gpt-4o",
                                        "gpt-5.1",
                                    ],
                                    value="xai.grok-code-fast-1",
                                        interactive=True,
                                        container=False,
                                    )
                        with gr.Column(scale=5):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    view_ai_analyze_btn = gr.Button("AI分析", variant="primary")
                    with gr.Row():
                        view_ai_analyze_status_md = gr.Markdown(visible=False)
                    with gr.Row():
                        view_ai_result_md = gr.Markdown(visible=False)
            
            # Event handlers
            def on_view_select(evt: gr.SelectData, current_df):
                """Handle view row selection.
                
                Always extracts the view name from the first column (View Name),
                regardless of which column in the row was clicked.
                
                Args:
                    evt: Gradio SelectData event
                    current_df: Current dataframe value
                """
                try:
                    row_index = evt.index[0]
                    logger.info(f"Row clicked: {row_index}, Value clicked: {evt.value}")
                    
                    if row_index >= 0 and current_df is not None:
                        logger.info(f"Dataframe type: {type(current_df)}")
                        
                        # Convert to DataFrame if it's a dict (Gradio format)
                        if isinstance(current_df, dict):
                            logger.info(f"Dict keys: {current_df.keys()}")
                            # Gradio returns dict with 'data' and 'headers' keys
                            if 'data' in current_df:
                                data = current_df['data']
                                headers = current_df.get('headers', [])
                                logger.info(f"Data length: {len(data)}, Headers: {headers}")
                                current_df = pd.DataFrame(data, columns=headers)
                            else:
                                # Try direct conversion with orientation
                                try:
                                    current_df = pd.DataFrame.from_dict(current_df, orient='tight')
                                except Exception as e:
                                    logger.warning(f"Failed to convert with orient='tight': {e}")
                                    current_df = pd.DataFrame(current_df)
                        
                        logger.info(f"DataFrame shape: {current_df.shape}")
                        logger.info(f"DataFrame columns: {list(current_df.columns)}")
                        
                        if len(current_df) > row_index:
                            # Always get the view name from the first column (index 0)
                            view_name = str(current_df.iloc[row_index, 0])
                            logger.info(f"View selected from row {row_index}: {view_name}")
                            col_df, ddl = get_view_details(pool, view_name)
                            if isinstance(col_df, pd.DataFrame) and not col_df.empty:
                                return view_name, gr.Dataframe(visible=True, value=col_df), ddl
                            else:
                                return view_name, gr.Dataframe(visible=False, value=pd.DataFrame()), ddl
                        else:
                            logger.warning(f"Row index {row_index} out of bounds, dataframe has {len(current_df)} rows")
                except Exception as e:
                    logger.error(f"Error in on_view_select: {e}")
                    logger.error(traceback.format_exc())
                
                return "", gr.Dataframe(visible=False, value=pd.DataFrame()), ""
            
            def refresh_view_list():
                try:
                    logger.info("ビュー一覧を取得ボタンがクリックされました")
                    yield gr.Markdown(value="⏳ ビュー一覧を取得中...", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["View Name", "Comments"]), label="ビュー一覧（件数: 0）")
                    df = get_view_list(pool)
                    cnt = len(df) if isinstance(df, pd.DataFrame) else 0
                    status_text = "✅ 取得完了（データなし）" if cnt == 0 else "✅ 取得完了"
                    yield gr.Markdown(value=status_text, visible=True), gr.Dataframe(value=df, visible=True, label=f"ビュー一覧（件数: {cnt}）")
                except Exception as e:
                    yield gr.Markdown(value=f"❌ 取得に失敗しました: {str(e)}", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["View Name", "Comments"]), label="ビュー一覧（件数: 0）")
            
            def drop_selected_view(view_name):
                """Drop the selected view and refresh list."""
                yield (
                    gr.Markdown(visible=True, value="⏳ ビューを削除中..."),
                    "",
                    pd.DataFrame(),
                    "",
                    "",
                    "",
                )
                result = drop_view(pool, view_name)
                status_md = gr.Markdown(visible=True, value=result)
                yield status_md, "", pd.DataFrame(), "", "", ""
            
            def execute_create_view_handler(sql):
                """Execute CREATE VIEW and refresh list."""
                sql_no_comment = remove_comments(sql)
                yield gr.Markdown(visible=True, value="⏳ ビュー作成を実行中...")
                result = execute_create_view(pool, sql_no_comment)
                status_md = gr.Markdown(visible=True, value=result)
                yield status_md
            
            def clear_view_sql():
                """Clear the SQL input."""
                return ""
            
            def load_view_sql_file(file_path):
                """
                SQLファイルを読み込み、テキストボックスに表示する.
                
                Args:
                    file_path: アップロードされたファイルのパス
                
                Returns:
                    ファイルの内容(文字列)
                """
                if not file_path:
                    return ""
                
                try:
                    # 複数のエンコーディングで試行
                    encodings = ['utf-8', 'shift_jis', 'cp932', 'latin1', 'euc-jp']
                    content = None
                    
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            logger.info(f"SQLファイルを{encoding}で読み込みました: {file_path}")
                            break
                        except (UnicodeDecodeError, UnicodeError):
                            continue
                    
                    if content is None:
                        logger.error(f"SQLファイルの読み込みに失敗しました: {file_path}")
                        return "❌ エラー: ファイルの読み込みに失敗しました。エンコーディングを確認してください。"
                    
                    return content
                    
                except Exception as e:
                    logger.error(f"SQLファイルの読み込み中にエラーが発生しました: {e}")
                    logger.error(traceback.format_exc())
                    return f"❌ エラー: {str(e)}"
            
            # Wire up events
            view_refresh_btn.click(
                fn=refresh_view_list,
                outputs=[view_refresh_status, view_list_df]
            )
            
            view_list_df.select(
                fn=on_view_select,
                inputs=[view_list_df],
                outputs=[selected_view_name, view_columns_df, view_ddl_text]
            )
            
            view_drop_btn.click(
                fn=drop_selected_view,
                inputs=[selected_view_name],
                outputs=[view_drop_result, selected_view_name, 
                        view_columns_df, view_ddl_text, view_join_text, view_where_text]
            ).then(
                fn=refresh_view_list,
                outputs=[view_refresh_status, view_list_df]
            )

            async def _view_join_where_ai_extract_async(model_name, ddl_text):
                if not model_name.startswith("gpt-"):
                    from utils.chat_util import get_oci_region, get_compartment_id
                    region = get_oci_region()
                    compartment_id = get_compartment_id()
                    if not region or not compartment_id:
                        return gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                try:
                    s = str(ddl_text or "").strip()
                    if not s:
                        return gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                    m = re.search(r"\b(SELECT|WITH)\b[\s\S]*", s, flags=re.IGNORECASE)
                    if m:
                        s = m.group(0)
                    
                    prompt = (
                        "Extract ONLY JOIN and WHERE conditions from the SQL query below.\n"
                        "Output in STRICT format (no explanations, no markdown, no extra text):\n\n"
                        "JOIN:\n"
                        "[JOIN_TYPE] alias1(schema.table1).column1 = alias2(schema.table2).column2\n"
                        "[JOIN_TYPE] alias3(schema.table3).column3 = alias4(schema.table4).column4\n\n"
                        "WHERE:\n"
                        "alias(schema.table).column operator value\n\n"
                        "Rules:\n"
                        "- Format: alias(schema.table_name).column or schema.table_name.column (if no alias)\n"
                        "- JOIN_TYPE must be one of: INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN, CROSS JOIN, JOIN\n"
                        "- Include schema name if present (e.g., ADMIN.USER_ROLE)\n"
                        "- One condition per line\n"
                        "- Keep original operators (=, >, <, LIKE, IN, etc.)\n"
                        "- Preserve exact column names and values with quotes\n"
                        "- If no JOIN/WHERE exists, output 'JOIN:\nNone' or 'WHERE:\nNone'\n\n"
                        "SQL:\n```sql\n" + s + "\n```"
                    )
                    messages = [
                        {"role": "system", "content": "You are a SQL parser. Output ONLY the requested format. No explanations."},
                        {"role": "user", "content": prompt},
                    ]
                    
                    if model_name.startswith("gpt-"):
                        from openai import AsyncOpenAI
                        client = AsyncOpenAI()
                        # Use Chat Completions API instead of Responses API to avoid 404 errors
                        resp = await client.chat.completions.create(model=model_name, messages=messages)
                        out = ""
                        if getattr(resp, "choices", None):
                            msg = resp.choices[0].message
                            out = msg.content if hasattr(msg, "content") else ""
                    else:
                        from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                        client = AsyncOciOpenAI(
                            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                            auth=OciUserPrincipalAuth(),
                            compartment_id=compartment_id,
                        )
                        resp = await client.chat.completions.create(model=model_name, messages=messages)
                        if getattr(resp, "choices", None):
                            msg = resp.choices[0].message
                            out = msg.content if hasattr(msg, "content") else ""
                        else:
                            out = ""

                    join_text = ""
                    where_text = ""
                    s2 = re.sub(r"```+\w*", "", str(out or ""))
                    m2 = re.search(r"JOIN:\s*([\s\S]*?)\n\s*WHERE:\s*([\s\S]*)$", s2, flags=re.IGNORECASE)
                    if m2:
                        join_text = m2.group(1).strip()
                        where_text = m2.group(2).strip()
                    if not join_text:
                        join_text = "None"
                    if not where_text:
                        where_text = "None"
                    return gr.Textbox(value=join_text, autoscroll=False), gr.Textbox(value=where_text, autoscroll=False)
                except Exception:
                    return gr.Textbox(value="None", autoscroll=False), gr.Textbox(value="None", autoscroll=False)

            def _view_join_where_ai_extract(model_name, ddl_text):
                import asyncio
                # 必須入力項目のチェック
                if not model_name or not str(model_name).strip():
                    yield gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                    return
                if not ddl_text or not str(ddl_text).strip():
                    yield gr.Markdown(visible=True, value="⚠️ CREATE VIEW SQLのDDLが空です。ビューを選択してください"), gr.Textbox(value="", autoscroll=False), gr.Textbox(value="", autoscroll=False)
                    return
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Textbox(value="...", autoscroll=False), gr.Textbox(value="...", autoscroll=False)
                    join_text, where_text = loop.run_until_complete(_view_join_where_ai_extract_async(model_name, ddl_text))
                    yield gr.Markdown(visible=True, value="✅ 分析完了"), join_text, where_text
                except Exception as e:
                    yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Textbox(value="Error", autoscroll=False), gr.Textbox(value=str(e), autoscroll=False)
                finally:
                    loop.close()

            view_ai_extract_btn.click(
                fn=_view_join_where_ai_extract,
                inputs=[view_analysis_model_input, view_ddl_text],
                outputs=[view_ai_extract_status_md,view_join_text, view_where_text],
            )

            create_view_btn.click(
                fn=execute_create_view_handler,
                inputs=[create_view_sql],
                outputs=[create_view_result]
            ).then(
                fn=refresh_view_list,
                outputs=[view_refresh_status, view_list_df]
            )
            
            clear_view_sql_btn.click(
                fn=clear_view_sql,
                outputs=[create_view_sql]
            )

            view_sql_file_input.change(
                fn=load_view_sql_file,
                inputs=[view_sql_file_input],
                outputs=[create_view_sql],
            )

            async def _view_ai_analyze_async(model_name, create_sql_text, exec_result_text):
                if not model_name.startswith("gpt-"):
                    from utils.chat_util import get_oci_region, get_compartment_id
                    region = get_oci_region()
                    compartment_id = get_compartment_id()
                    if not region or not compartment_id:
                        return gr.Markdown(visible=True, value="ℹ️ OCI設定が不足しています")
                try:
                    
                    sql_part = str(create_sql_text or "").strip()
                    result_part = str(exec_result_text or "").strip()
                    prompt = (
                        "以下のSQL/DDLと実行結果を分析してください。出力は次の3点に限定します。\n"
                        "1) エラー原因(該当する場合)\n"
                        "2) 解決方法(修正案や具体的手順)\n"
                        "3) 簡潔な結論(不要な詳細は省略)\n\n"
                        + ("SQL/DDL:\n```sql\n" + sql_part + "\n```\n" if sql_part else "")
                        + ("実行結果:\n" + result_part + "\n" if result_part else "")
                    )
                    messages = [
                        {"role": "system", "content": "あなたはシニアDBエンジニアです。SQLと実行結果の故障診断に特化し、エラー原因と実行可能な修復策のみを簡潔に提示してください。不要な詳細は出力しないでください。"},
                        {"role": "user", "content": prompt},
                    ]
                    
                    if model_name.startswith("gpt-"):
                        from openai import AsyncOpenAI
                        client = AsyncOpenAI()
                        # Use Chat Completions API instead of Responses API to avoid 404 errors
                        resp = await client.chat.completions.create(model=model_name, messages=messages)
                        text = ""
                        if getattr(resp, "choices", None):
                            msg = resp.choices[0].message
                            text = msg.content if hasattr(msg, "content") else ""
                    else:
                        from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                        client = AsyncOciOpenAI(
                            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                            auth=OciUserPrincipalAuth(),
                            compartment_id=compartment_id,
                        )
                        resp = await client.chat.completions.create(model=model_name, messages=messages)
                        text = ""
                        if getattr(resp, "choices", None):
                            msg = resp.choices[0].message
                            text = msg.content if hasattr(msg, "content") else ""
                            
                    return gr.Markdown(visible=True, value=text or "分析結果が空です")
                except Exception as e:
                    return gr.Markdown(visible=True, value=f"❌ エラー: {e}")

            def view_ai_analyze(model_name, create_sql_text, exec_result_text):
                import asyncio
                # 必須入力項目のチェック
                if not model_name or not str(model_name).strip():
                    yield gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Markdown(visible=False)
                    return
                if not create_sql_text or not str(create_sql_text).strip():
                    yield gr.Markdown(visible=True, value="⚠️ CREATE VIEW SQLを入力してください"), gr.Markdown(visible=False)
                    return
                if not exec_result_text or not str(exec_result_text).strip():
                    yield gr.Markdown(visible=True, value="⚠️ 実行結果がありません。先にビュー作成を実行してください"), gr.Markdown(visible=False)
                    return
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Markdown(visible=False)
                    result_md = loop.run_until_complete(_view_ai_analyze_async(model_name, create_sql_text, exec_result_text))
                    yield gr.Markdown(visible=True, value="✅ 完了"), result_md
                except Exception as e:
                    yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Markdown(visible=False)
                finally:
                    loop.close()

            view_ai_analyze_btn.click(
                fn=view_ai_analyze,
                inputs=[view_ai_model_input, create_view_sql, create_view_result],
                outputs=[view_ai_analyze_status_md, view_ai_result_md],
            )
        
        # Data Management Tab
        with gr.TabItem(label="データの管理"):
            # Feature 1: Table Data Display
            with gr.Accordion(label="1. テーブル・ビューデータの表示", open=True):
                data_refresh_btn = gr.Button("テーブル・ビュー一覧を取得（時間がかかる場合があります）", variant="primary")
                data_refresh_status = gr.Markdown(visible=False)
                
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("テーブル・ビュー選択*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                data_table_select = gr.Dropdown(
                                    show_label=False,
                                    choices=[("","")],
                                    value="",
                                    interactive=True,
                                    container=False,
                                )
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("取得件数*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                data_limit_input = gr.Number(
                                    show_label=False,
                                    value=100,
                                    minimum=1,
                                    maximum=10000,
                                    container=False,
                                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("WHERE条件（オプション）", elem_classes="input-label")
                    with gr.Column(scale=5):
                        data_where_input = gr.Textbox(
                            show_label=False,
                            placeholder="例: status = 'A' AND created_at > SYSDATE - 7",
                            lines=2,
                            max_lines=5,
                            container=False,
                        )

                with gr.Row():                
                    data_display_btn = gr.Button("データを表示", variant="primary")
                
                with gr.Row():                
                    data_display_status = gr.Markdown(visible=False)
                with gr.Row():
                    data_display = gr.Dataframe(
                        label="データ表示",
                        interactive=False,
                        wrap=True,
                        visible=False,
                        value=pd.DataFrame(),
                        elem_id="data_result_df",
                    )
            
            # Feature 2: CSV Upload
            with gr.Accordion(label="2. CSVアップロード", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("CSVファイル*", elem_classes="input-label")
                    with gr.Column(scale=5):
                        csv_file_input = gr.File(
                            show_label=False,
                            file_types=[".csv"],
                            type="filepath",
                        )
                
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("アップロード先テーブル*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                csv_table_select = gr.Dropdown(
                                    show_label=False,
                                    choices=[],
                                    interactive=True,
                                    visible=True,
                                    container=False,
                                )
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("")

                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("アップロードモード*", elem_classes="input-label")
                            with gr.Column(scale=5):
                                csv_upload_mode = gr.Radio(
                                    show_label=False,
                                    choices=["INSERT", "TRUNCATE & INSERT"],
                                    value="INSERT",
                                    container=False,
                                )
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("")
                
                csv_preview_info = gr.Markdown(
                    value="ℹ️ CSVファイルを選択するとプレビューが表示されます",
                    visible=True,
                )
                csv_preview = gr.Dataframe(
                    label="プレビュー（最初の10行）",
                    interactive=False,
                    wrap=True,
                    visible=False,
                    value=pd.DataFrame(),
                )
                
                csv_upload_btn = gr.Button("アップロード", variant="primary")
                csv_upload_status_md = gr.Markdown(visible=False)
                csv_upload_result = gr.Markdown(visible=False)
            
            # Feature 3: SQL Bulk Execution
            with gr.Accordion(label="3. SQL一括実行", open=True):
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("SQLテンプレート（オプション）", elem_classes="input-label")
                            with gr.Column(scale=5):
                                sql_template_select = gr.Dropdown(
                                    show_label=False,
                                    choices=[
                                        "",
                                        "INSERT - 単一行",
                                        "INSERT - 複数行",
                                        "UPDATE",
                                        "DELETE",
                                        "MERGE",
                                    ],
                                    value="",
                                    interactive=True,
                                    container=False,
                                )
                    with gr.Column(scale=5):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("ℹ️ 複数の文をセミコロンで区切って入力可能")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("SQL*", elem_classes="input-label")
                    with gr.Column(scale=5):
                        data_sql_input = gr.Textbox(
                            show_label=False,
                            placeholder="INSERT/UPDATE/DELETE/MERGE/TRUNCATE文を入力してください（注: SELECT文は禁止されています）\n例:\nINSERT INTO users (username, email, status) VALUES ('user1', 'user1@example.com', 'A');\nINSERT INTO users (username, email, status) VALUES ('user2', 'user2@example.com', 'A');\nUPDATE users SET status = 'A' WHERE user_id = 1;\nDELETE FROM users WHERE status = 'D';\nTRUNCATE TABLE temp_table;",
                            lines=10,
                            max_lines=15,
                            show_copy_button=True,
                            container=False,
                            autoscroll=False,
                        )
                
                with gr.Row():
                    data_clear_btn = gr.Button("クリア", variant="secondary")
                    data_execute_btn = gr.Button("実行", variant="primary")

                with gr.Row():
                    data_sql_status = gr.Markdown(visible=False)
                with gr.Row():
                    data_sql_result = gr.Markdown(visible=False)

                with gr.Accordion(label="AI分析と処理", open=True):
                    with gr.Row():
                        with gr.Column(scale=5):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("モデル*", elem_classes="input-label")
                                with gr.Column(scale=5):
                                    data_ai_model_input = gr.Dropdown(
                                        show_label=False,
                                        choices=[
                                            "xai.grok-code-fast-1",
                                            "xai.grok-3",
                                            "xai.grok-3-fast",
                                            "xai.grok-4",
                                            "xai.grok-4-fast-non-reasoning",
                                            "meta.llama-4-scout-17b-16e-instruct",
                                            "gpt-4o",
                                            "gpt-5.1",
                                        ],
                                        value="xai.grok-code-fast-1",
                                        interactive=True,
                                        container=False,
                                    )
                        with gr.Column(scale=5):
                            with gr.Row():
                                data_ai_analyze_btn = gr.Button("AI分析", variant="primary")
                    with gr.Row():
                        data_ai_status_md = gr.Markdown(visible=False)
                    with gr.Row():
                        data_ai_result_md = gr.Markdown(visible=False)
            
            # Event Handlers
            def refresh_data_table_list():
                try:
                    logger.info("テーブル・ビュー一覧を取得ボタンがクリックされました")
                    yield gr.Markdown(value="⏳ テーブル・ビュー一覧を取得中...", visible=True), gr.Dropdown(choices=[]), gr.Dropdown(choices=[], visible=False)
                    data_names = get_table_list_for_data(pool)
                    upload_tables = get_table_list_for_upload(pool)
                    status_text = "✅ 取得完了（データなし）" if (not data_names and not upload_tables) else "✅ 取得完了"
                    yield gr.Markdown(value=status_text, visible=True), gr.Dropdown(choices=data_names, visible=True), gr.Dropdown(choices=upload_tables, visible=True)
                except Exception as e:
                    yield gr.Markdown(value=f"❌ 取得に失敗しました: {str(e)}", visible=True), gr.Dropdown(choices=[]), gr.Dropdown(choices=[], visible=False)
            
            def display_data(table_name, limit, where_clause):
                try:
                    yield gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(value="⏳ データを取得中...", visible=True)
                    df = display_table_data(pool, table_name, limit, where_clause)
                    if df.empty:
                        yield gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(value="✅ 表示完了（データなし）", visible=True)
                        return
                    widths = []
                    sample = df.head(5)
                    columns = max(1, len(df.columns))
                    for col in df.columns:
                        series = sample[col].astype(str)
                        row_max = series.map(len).max() if len(series) > 0 else 0
                        length = max(len(str(col)), row_max)
                        widths.append(min(100 / columns, length))
                    total = sum(widths) if widths else 0
                    if total <= 0:
                        _ = ""
                    else:
                        col_widths = [max(5, int(100 * w / total)) for w in widths]
                        diff = 100 - sum(col_widths)
                        if diff != 0 and len(col_widths) > 0:
                            col_widths[0] = max(5, col_widths[0] + diff)
                        rules = []
                        rules.append("#data_result_df { width: 100% !important; }")
                        rules.append("#data_result_df .wrap { overflow-x: auto !important; }")
                        rules.append("#data_result_df table { table-layout: fixed !important; width: 100% !important; border-collapse: collapse !important; }")
                        for idx, pct in enumerate(col_widths, start=1):
                            rules.append(f"#data_result_df table th:nth-child({idx}), #data_result_df table td:nth-child({idx}) {{ width: {pct}% !important; overflow: hidden !important; text-overflow: ellipsis !important; }}")
                        _ = "<style>" + "\n".join(rules) + "</style>"

                    df_component = gr.Dataframe(
                        label=f"データ表示（件数: {len(df)}）",
                        interactive=False,
                        wrap=True,
                        visible=True,
                        value=df,
                        elem_id="data_result_df",
                    )
                    # style_component is removed
                    yield df_component, gr.Markdown(visible=True, value="✅ 表示完了")
                except Exception as e:
                    yield gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(value=f"❌ データ取得に失敗しました: {str(e)}", visible=True)
            
            def upload_csv(file, table_name, mode):
                """Upload CSV file."""
                yield gr.Dataframe(), gr.Markdown(visible=True, value="⏳ CSVアップロードを実行中..."), gr.Markdown(visible=False)
                preview, result = upload_csv_data(pool, file, table_name, mode)
                status_md = gr.Markdown(visible=True, value=result)
                yield gr.Dataframe(visible=True, value=preview), gr.Markdown(visible=False), status_md
            
            def execute_sql(sql):
                """Execute SQL statements."""
                sql_no_comment = remove_comments(sql)
                yield gr.Markdown(visible=True, value="⏳ SQL一括実行中..."), gr.Markdown(visible=False)
                result = execute_data_sql(pool, sql_no_comment)
                yield gr.Markdown(visible=False), gr.Markdown(visible=True, value=result)
            
            def clear_sql():
                """Clear SQL input."""
                return ""
            
            def apply_sql_template(template):
                """Apply SQL template to input."""
                if not template:
                    return ""
                
                templates = {
                    "INSERT - 単一行": "INSERT INTO table_name (column1, column2, column3) VALUES (value1, value2, value3);",
                    "INSERT - 複数行": "INSERT INTO table_name (column1, column2) VALUES (value1, value2);\nINSERT INTO table_name (column1, column2) VALUES (value3, value4);\nINSERT INTO table_name (column1, column2) VALUES (value5, value6);",
                    "UPDATE": "UPDATE table_name SET column1 = value1, column2 = value2 WHERE condition;",
                    "DELETE": "DELETE FROM table_name WHERE condition;",
                    "MERGE": "MERGE INTO target_table t\nUSING source_table s\nON (t.id = s.id)\nWHEN MATCHED THEN\n  UPDATE SET t.column1 = s.column1\nWHEN NOT MATCHED THEN\n  INSERT (id, column1) VALUES (s.id, s.column1);",
                }
                return templates.get(template, "")
            
            # Wire up events
            data_refresh_btn.click(
                fn=refresh_data_table_list,
                outputs=[data_refresh_status, data_table_select, csv_table_select]
            )
            
            data_display_btn.click(
                fn=display_data,
                inputs=[data_table_select, data_limit_input, data_where_input],
                outputs=[data_display, data_display_status]
            )
            
            def preview_csv(file):
                """Preview CSV file."""
                if file:
                    try:
                        preview_df = pd.read_csv(file.name).head(10)
                        if preview_df.empty:
                            return gr.Markdown(visible=True, value="⚠️ CSVファイルにデータがありません"), gr.Dataframe(visible=False, value=pd.DataFrame())
                        return gr.Markdown(visible=False), gr.Dataframe(visible=True, value=preview_df)
                    except Exception as e:
                        return gr.Markdown(visible=True, value=f"❌ CSV読み込みエラー: {str(e)}"), gr.Dataframe(visible=False, value=pd.DataFrame())
                else:
                    return gr.Markdown(visible=True, value="ℹ️ CSVファイルを選択するとプレビューが表示されます"), gr.Dataframe(visible=False, value=pd.DataFrame())
            
            csv_file_input.change(
                fn=preview_csv,
                inputs=[csv_file_input],
                outputs=[csv_preview_info, csv_preview]
            )
            
            csv_upload_btn.click(
                fn=upload_csv,
                inputs=[csv_file_input, csv_table_select, csv_upload_mode],
                outputs=[csv_preview, csv_upload_status_md, csv_upload_result]
            )
            
            data_execute_btn.click(
                fn=execute_sql,
                inputs=[data_sql_input],
                outputs=[data_sql_status, data_sql_result]
            )
            
            data_clear_btn.click(
                fn=clear_sql,
                outputs=[data_sql_input]
            )
            
            sql_template_select.change(
                fn=apply_sql_template,
                inputs=[sql_template_select],
                outputs=[data_sql_input]
            )

            async def _data_ai_analyze_async(model_name, create_sql_text, exec_result_text):
                if not model_name.startswith("gpt-"):
                    from utils.chat_util import get_oci_region, get_compartment_id
                    region = get_oci_region()
                    compartment_id = get_compartment_id()
                    if not region or not compartment_id:
                        return gr.Markdown(visible=True, value="ℹ️ OCI設定が不足しています")
                try:
                    
                    sql_part = str(create_sql_text or "").strip()
                    result_part = str(exec_result_text or "").strip()
                    prompt = (
                        "以下のSQLと実行結果を分析してください。出力は次の3点に限定します。\n"
                        "1) エラー原因(該当する場合)\n"
                        "2) 解決方法(修正案や具体的手順)\n"
                        "3) 簡潔な結論(不要な詳細は省略)\n\n"
                        + ("SQL:\n```sql\n" + sql_part + "\n```\n" if sql_part else "")
                        + ("実行結果:\n" + result_part + "\n" if result_part else "")
                    )
                    
                    messages = [
                        {"role": "system", "content": "あなたはシニアDBエンジニアです。SQLと実行結果の故障診断に特化し、エラー原因と実行可能な修復策のみを簡潔に提示してください。不要な詳細は出力しないでください。"},
                        {"role": "user", "content": prompt},
                    ]
                    
                    if model_name.startswith("gpt-"):
                        from openai import AsyncOpenAI
                        client = AsyncOpenAI()
                        # Use Chat Completions API instead of Responses API to avoid 404 errors
                        resp = await client.chat.completions.create(model=model_name, messages=messages)
                        text = ""
                        if getattr(resp, "choices", None):
                            msg = resp.choices[0].message
                            text = msg.content if hasattr(msg, "content") else ""
                    else:
                        from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                        client = AsyncOciOpenAI(
                            service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                            auth=OciUserPrincipalAuth(),
                            compartment_id=compartment_id,
                        )
                        resp = await client.chat.completions.create(model=model_name, messages=messages)
                        text = ""
                        if getattr(resp, "choices", None):
                            msg = resp.choices[0].message
                            text = msg.content if hasattr(msg, "content") else ""
                            
                    return gr.Markdown(visible=True, value=text or "分析結果が空です")
                except Exception as e:
                    return gr.Markdown(visible=True, value=f"❌ エラー: {e}")

            def data_ai_analyze(model_name, create_sql_text, exec_result_text):
                import asyncio
                # 必須入力項目のチェック
                if not model_name or not str(model_name).strip():
                    yield gr.Markdown(visible=True, value="⚠️ モデルを選択してください"), gr.Markdown(visible=False)
                    return
                if not create_sql_text or not str(create_sql_text).strip():
                    yield gr.Markdown(visible=True, value="⚠️ SQLを入力してください"), gr.Markdown(visible=False)
                    return
                if not exec_result_text or not str(exec_result_text).strip():
                    yield gr.Markdown(visible=True, value="⚠️ 実行結果がありません。先にSQLを実行してください"), gr.Markdown(visible=False)
                    return
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Markdown(visible=False)
                    result_md = loop.run_until_complete(_data_ai_analyze_async(model_name, create_sql_text, exec_result_text))
                    yield gr.Markdown(visible=True, value="✅ 完了"), result_md
                except Exception as e:
                    yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Markdown(visible=False)
                finally:
                    loop.close()

            data_ai_analyze_btn.click(
                fn=data_ai_analyze,
                inputs=[data_ai_model_input, data_sql_input, data_sql_result],
                outputs=[data_ai_status_md, data_ai_result_md],
            )
