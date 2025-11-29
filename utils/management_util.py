"""Management utility module for SQL Assist.

This module provides UI components for database management functions including
Table Management, View Management, and Data Management.
"""

import logging
import traceback
import re

import gradio as gr
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


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
                        create_sql += f"\nCOMMENT ON TABLE {table_name.upper()} IS '{table_comment.replace("'", "''")}';\n"
                    
                    # Add column comments
                    if col_comments:
                        for col_name, col_comment in col_comments:
                            create_sql += f"COMMENT ON COLUMN {table_name.upper()}.{col_name} IS '{col_comment.replace("'", "''")}';\n"
                
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
        return "❌ エラー: SQL文が入力されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Split SQL statements by semicolon
                sql_statements = [stmt.strip() for stmt in create_sql.split(';') if stmt.strip()]
                
                if not sql_statements:
                    logger.error("CREATE TABLEのSQL文が空です")
                    return "❌ エラー: SQL文が空です"

                disallowed = []
                for idx, sql_stmt in enumerate(sql_statements, 1):
                    stmt_upper = sql_stmt.strip().upper()
                    is_create_table = stmt_upper.startswith('CREATE TABLE') or bool(re.match(r'^CREATE\s+GLOBAL\s+TEMPORARY\s+TABLE\b', stmt_upper))
                    is_comment = stmt_upper.startswith('COMMENT ON TABLE') or stmt_upper.startswith('COMMENT ON COLUMN')
                    if not (is_create_table or is_comment):
                        disallowed.append((idx, sql_stmt))
                if disallowed:
                    first_idx, first_sql = disallowed[0]
                    error_msg = f"禁止された操作: CREATE TABLE / COMMENT ON 以外の文は実行できません。\n文{first_idx}: {first_sql[:100]}..."
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
                
                # Get CREATE VIEW statement
                cursor.execute(
                    "SELECT DBMS_METADATA.GET_DDL('VIEW', :view_name, 'ADMIN') FROM DUAL",
                    view_name=view_name.upper()
                )
                ddl_result = cursor.fetchone()
                create_sql = ddl_result[0].read() if ddl_result and ddl_result[0] else ""
                
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
                        create_sql += f"\nCOMMENT ON TABLE {view_name.upper()} IS '{view_comment.replace("'", "''")}';\n"
                    
                    # Add column comments
                    if col_comments:
                        for col_name, col_comment in col_comments:
                            create_sql += f"COMMENT ON COLUMN {view_name.upper()}.{col_name} IS '{col_comment.replace("'", "''")}';\n"
                
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
        return "❌ エラー: SQL文が入力されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Split SQL statements by semicolon
                sql_statements = [stmt.strip() for stmt in create_sql.split(';') if stmt.strip()]
                
                if not sql_statements:
                    logger.error("CREATE VIEWのSQL文が空です")
                    return "❌ エラー: SQL文が空です"

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
                    if not (is_create_view or is_comment):
                        disallowed.append((idx, sql_stmt))
                if disallowed:
                    first_idx, first_sql = disallowed[0]
                    error_msg = f"禁止された操作: VIEW作成に関係ない文は実行できません。\n文{first_idx}: {first_sql[:100]}..."
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
                # Get table columns
                cursor.execute(
                    "SELECT column_name FROM all_tab_columns WHERE owner = 'ADMIN' AND table_name = :table_name ORDER BY column_id",
                    table_name=table_name.upper()
                )
                table_columns = [row[0] for row in cursor.fetchall()]
                
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
                        values = [row[csv_col] if csv_col in column_mapping else None for csv_col in column_mapping.keys()]
                        # Convert NaN to None
                        values = [None if pd.isna(v) else v for v in values]
                        cursor.execute(insert_sql, values)
                        success_count += 1
                    except Exception as row_error:
                        error_count += 1
                        if error_count <= 5:  # Show first 5 errors
                            error_messages.append(f"行{idx+1}: {str(row_error)[:100]}")
                
                # Commit transaction
                conn.commit()
                
                # Prepare result message
                result = f"✅ 成功: {success_count}件のデータを挿入しました"
                if error_count > 0:
                    result += f"\n\n⚠️ 警告: {error_count}件のエラーが発生しました\n" + "\n".join(error_messages)
                    if error_count > 5:
                        result += f"\n... 他 {error_count - 5} 件のエラー"
                
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
        return "❌ エラー: SQL文が入力されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Split SQL statements by semicolon
                statements = [stmt.strip() for stmt in sql_statements.split(';') if stmt.strip()]
                
                if not statements:
                    logger.error("データSQLの文が空です")
                    return "❌ エラー: SQL文が空です"
                
                disallowed = []
                for idx, sql_stmt in enumerate(statements, 1):
                    stmt_upper = sql_stmt.strip().upper()
                    is_insert = stmt_upper.startswith('INSERT')
                    is_update = stmt_upper.startswith('UPDATE')
                    is_delete = stmt_upper.startswith('DELETE')
                    is_merge = stmt_upper.startswith('MERGE')
                    if not (is_insert or is_update or is_delete or is_merge):
                        disallowed.append((idx, sql_stmt))
                if disallowed:
                    first_idx, first_sql = disallowed[0]
                    error_msg = f"禁止された操作: INSERT, UPDATE, DELETE, MERGE 以外の文は実行できません。\n文{first_idx}: {first_sql[:100]}..."
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
    Only COMMENT ON TABLE / COMMENT ON COLUMN を許可。
    
    Args:
        pool: Oracle database connection pool
        sql_statements: Single or multiple SQL statements
        
    Returns:
        str: Result message
    """
    if not sql_statements or not sql_statements.strip():
        logger.error("COMMENT文が未入力です")
        return "❌ エラー: SQL文が入力されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                statements = [stmt.strip() for stmt in sql_statements.split(';') if stmt.strip()]
                if not statements:
                    logger.error("COMMENTのSQL文が空です")
                    return "❌ エラー: SQL文が空です"
                disallowed = []
                for idx, sql_stmt in enumerate(statements, 1):
                    stmt_upper = sql_stmt.strip().upper()
                    is_comment_table = stmt_upper.startswith('COMMENT ON TABLE')
                    is_comment_column = stmt_upper.startswith('COMMENT ON COLUMN')
                    if not (is_comment_table or is_comment_column):
                        disallowed.append((idx, sql_stmt))
                if disallowed:
                    first_idx, first_sql = disallowed[0]
                    error_msg = f"禁止された操作: COMMENT ON TABLE/COLUMN 以外の文は実行できません。\n文{first_idx}: {first_sql[:100]}..."
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
        return "❌ エラー: SQL文が入力されていません"
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                statements = [stmt.strip() for stmt in str(sql_statements).split(';') if stmt.strip()]
                if not statements:
                    logger.error("アノテーションのSQL文が空です")
                    return "❌ エラー: SQL文が空です"
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
                table_refresh_btn = gr.Button("テーブル一覧を取得", variant="primary")
                table_refresh_status = gr.Markdown(visible=False)
                table_list_df = gr.Dataframe(
                    label="テーブル一覧(行をクリックして詳細を表示)",
                    interactive=False,
                    wrap=True,
                    value=pd.DataFrame(columns=["Table Name", "Rows", "Comments"]),
                    headers=["Table Name", "Rows", "Comments"],
                    visible=False,
                    max_height=300,
                )
            
            # Feature 2: Table Details and Drop
            with gr.Accordion(label="2. テーブル詳細と削除", open=True):
                selected_table_name = gr.Textbox(
                    label="選択されたテーブル名",
                    interactive=False,
                )

                with gr.Row():
                    table_drop_btn = gr.Button("選択したテーブルを削除", variant="stop")
                with gr.Row():
                    table_drop_result = gr.Markdown(visible=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 列情報")
                        table_columns_df = gr.Dataframe(
                            label="列情報",
                            show_label=False,
                            interactive=False,
                            wrap=True,
                            headers=["Column Name", "Data Type", "Nullable", "Comments"],
                        )
                    
                    with gr.Column():
                        gr.Markdown("### CREATE TABLE SQL")
                        table_ddl_text = gr.Textbox(
                            label="DDL",
                            lines=15,
                            max_lines=15,
                            interactive=False,
                            show_copy_button=True,
                            autoscroll=False,
                        )
            
            # Feature 3: Create Table
            with gr.Accordion(label="3. テーブル作成", open=False):
                create_table_sql = gr.Textbox(
                    label="CREATE TABLE SQL文（複数の文をセミコロンで区切って入力可能）",
                    placeholder="CREATE TABLE文を入力してください\n例:\nCREATE TABLE test_table (\n  id NUMBER PRIMARY KEY,\n  name VARCHAR2(100)\n);\n\nCOMMENT ON TABLE test_table IS 'テストテーブル';\nCOMMENT ON COLUMN test_table.id IS 'ID';\nCOMMENT ON COLUMN test_table.name IS '名称';",
                    lines=10,
                    max_lines=15,
                    show_copy_button=True,
                )
                
                with gr.Row():
                    with gr.Column():
                        clear_sql_btn = gr.Button("クリア", variant="secondary")
                    with gr.Column():
                        create_table_btn = gr.Button("テーブルを作成", variant="primary")
                
                with gr.Row():
                    create_table_result = gr.Markdown(visible=False)

                with gr.Accordion(label="AI分析と処理", open=False):
                    table_ai_model_input = gr.Dropdown(
                        label="モデル",
                        choices=[
                            "xai.grok-code-fast-1",
                            "xai.grok-3",
                            "xai.grok-3-fast",
                            "xai.grok-4",
                            "xai.grok-4-fast-non-reasoning",
                            "meta.llama-4-scout-17b-16e-instruct",
                        ],
                        value="xai.grok-code-fast-1",
                        interactive=True,
                    )
                    table_ai_analyze_btn = gr.Button("AI分析", variant="primary")
                    table_ai_status_md = gr.Markdown(visible=False)
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
                            return table_name, col_df, ddl
                        else:
                            logger.warning(f"Row index {row_index} out of bounds, dataframe has {len(current_df)} rows")
                except Exception as e:
                    logger.error(f"Error in on_table_select: {e}")
                    logger.error(traceback.format_exc())
                
                return "", pd.DataFrame(), ""
            
            def refresh_table_list():
                try:
                    logger.info("テーブル一覧を取得ボタンがクリックされました")
                    yield gr.Markdown(value="⏳ テーブル一覧を取得中...", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Table Name", "Rows", "Comments"]))
                    df = get_table_list(pool)
                    yield gr.Markdown(value="✅ 取得完了", visible=True), gr.Dataframe(value=df, visible=True)
                except Exception as e:
                    yield gr.Markdown(value=f"❌ 取得に失敗しました: {str(e)}", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Table Name", "Rows", "Comments"]))
            
            def drop_selected_table(table_name):
                """Drop the selected table and refresh list."""
                yield (
                    gr.Markdown(visible=True, value="⏳ テーブルを削除中..."),
                    gr.Dataframe(visible=False, value=pd.DataFrame()),
                    gr.Textbox(value=str(table_name or "")),
                    gr.Dataframe(visible=False, value=pd.DataFrame()),
                    gr.Textbox(value=""),
                )
                result = drop_table(pool, table_name)
                new_list = get_table_list(pool)
                status_md = gr.Markdown(visible=True, value=result)
                yield (
                    status_md,
                    gr.Dataframe(value=new_list, visible=True),
                    gr.Textbox(value=""),
                    gr.Dataframe(value=pd.DataFrame()),
                    gr.Textbox(value=""),
                )
            
            def execute_create(sql):
                """Execute CREATE TABLE and refresh list."""
                yield gr.Markdown(visible=True, value="⏳ テーブル作成を実行中..."), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["Table Name", "Rows", "Comments"]))
                result = execute_create_table(pool, sql)
                new_list = get_table_list(pool)
                status_md = gr.Markdown(visible=True, value=result)
                yield status_md, gr.Dataframe(value=new_list, visible=True)
            
            def clear_sql():
                """Clear the SQL input."""
                return ""
            
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
                outputs=[table_drop_result, table_list_df, selected_table_name, 
                        table_columns_df, table_ddl_text]
            )
            
            create_table_btn.click(
                fn=execute_create,
                inputs=[create_table_sql],
                outputs=[create_table_result, table_list_df]
            )
            
            clear_sql_btn.click(
                fn=clear_sql,
                outputs=[create_table_sql]
            )

            async def _table_ai_analyze_async(model_name, table_name, columns_df_input, ddl_text, create_sql_text, exec_result_text):
                from utils.chat_util import get_oci_region, get_compartment_id
                region = get_oci_region()
                compartment_id = get_compartment_id()
                if not region or not compartment_id:
                    return gr.Markdown(visible=True, value="ℹ️ OCI設定が不足しています")
                try:
                    import pandas as pd
                    from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                    if isinstance(columns_df_input, dict) and "data" in columns_df_input:
                        headers = columns_df_input.get("headers", [])
                        columns_df = pd.DataFrame(columns_df_input["data"], columns=headers)
                    elif isinstance(columns_df_input, pd.DataFrame):
                        columns_df = columns_df_input
                    else:
                        columns_df = pd.DataFrame()
                    preview = columns_df.head(10).to_markdown(index=False) if not columns_df.empty else ""
                    sql_part = str(create_sql_text or "").strip()
                    if not sql_part:
                        sql_part = str(ddl_text or "").strip()
                    result_part = str(exec_result_text or "").strip()
                    prompt = (
                        "以下のSQLと実行結果を分析してください。出力は次の3点に限定します。\n"
                        "1) エラー原因（該当する場合）\n"
                        "2) 解決方法（修正案や具体的手順）\n"
                        "3) 簡潔な結論（不要な詳細は省略）\n\n"
                        + ("SQL:\n```sql\n" + sql_part + "\n```\n" if sql_part else "")
                        + ("実行結果:\n" + result_part + "\n" if result_part else "")
                        + ("列情報プレビュー:\n" + preview + "\n" if preview else "")
                    )
                    client = AsyncOciOpenAI(
                        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                        auth=OciUserPrincipalAuth(),
                        compartment_id=compartment_id,
                    )
                    messages = [
                        {"role": "system", "content": "あなたはシニアDBエンジニアです。SQLと実行結果の故障診断に特化し、エラー原因と実行可能な修復策のみを簡潔に提示してください。不要な詳細は出力しないでください。"},
                        {"role": "user", "content": prompt},
                    ]
                    resp = await client.chat.completions.create(model=model_name, messages=messages)
                    text = ""
                    if getattr(resp, "choices", None):
                        msg = resp.choices[0].message
                        text = msg.content if hasattr(msg, "content") else ""
                    return gr.Markdown(visible=True, value=text or "分析結果が空です")
                except Exception as e:
                    return gr.Markdown(visible=True, value=f"❌ エラー: {e}")

            def table_ai_analyze(model_name, table_name, columns_df_input, ddl_text, create_sql_text, exec_result_text):
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Markdown(visible=False)
                    result_md = loop.run_until_complete(_table_ai_analyze_async(model_name, table_name, columns_df_input, ddl_text, create_sql_text, exec_result_text))
                    yield gr.Markdown(visible=True, value="✅ 完了"), result_md
                except Exception as e:
                    yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Markdown(visible=False)
                finally:
                    loop.close()

            table_ai_analyze_btn.click(
                fn=table_ai_analyze,
                inputs=[table_ai_model_input, selected_table_name, table_columns_df, table_ddl_text, create_table_sql, create_table_result],
                outputs=[table_ai_status_md, table_ai_result_md],
            )
        
        # View Management Tab
        with gr.TabItem(label="ビューの管理"):
            # Feature 1: View List
            with gr.Accordion(label="1. ビュー一覧", open=True):
                view_refresh_btn = gr.Button("ビュー一覧を取得", variant="primary")
                view_refresh_status = gr.Markdown(visible=False)
                view_list_df = gr.Dataframe(
                    label="ビュー一覧(行をクリックして詳細を表示)",
                    interactive=False,
                    wrap=True,
                    value=pd.DataFrame(columns=["View Name", "Comments"]),
                    headers=["View Name", "Comments"],
                    visible=False,
                )
            
            # Feature 2: View Details and Drop
            with gr.Accordion(label="2. ビュー詳細と削除", open=True):
                selected_view_name = gr.Textbox(
                    label="選択されたビュー名",
                    interactive=False,
                )

                with gr.Row():
                    view_drop_btn = gr.Button("選択したビューを削除", variant="stop")
                with gr.Row():
                    view_drop_result = gr.Markdown(visible=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 列情報")
                        view_columns_df = gr.Dataframe(
                            label="列情報",
                            show_label=False,
                            interactive=False,
                            wrap=True,
                            headers=["Column Name", "Data Type", "Nullable", "Comments"],
                        )
                    
                    with gr.Column():
                        gr.Markdown("### CREATE VIEW SQL")
                        view_ddl_text = gr.Textbox(
                            label="DDL",
                            lines=15,
                            max_lines=15,
                            interactive=False,
                            show_copy_button=True,
                            autoscroll=False,
                        )

                with gr.Row():
                    view_analysis_model_input = gr.Dropdown(
                        label="モデル",
                        choices=[
                            "xai.grok-code-fast-1",
                            "xai.grok-3",
                            "xai.grok-3-fast",
                            "xai.grok-4",
                            "xai.grok-4-fast-non-reasoning",
                            "meta.llama-4-scout-17b-16e-instruct",
                        ],
                        value="xai.grok-code-fast-1",
                        interactive=True,
                    )

                with gr.Row():
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
            with gr.Accordion(label="3. ビュー作成", open=False):
                create_view_sql = gr.Textbox(
                    label="CREATE VIEW SQL文（複数の文をセミコロンで区切って入力可能）",
                    placeholder="CREATE VIEW文を入力してください\n例:\nCREATE VIEW test_view AS\nSELECT id, name FROM test_table;\n\nCOMMENT ON TABLE test_view IS 'テストビュー';\nCOMMENT ON COLUMN test_view.id IS 'ID';\nCOMMENT ON COLUMN test_view.name IS '名称';",
                    lines=10,
                    max_lines=15,
                    show_copy_button=True,
                )
                
                with gr.Row():
                    with gr.Column():
                        clear_view_sql_btn = gr.Button("クリア", variant="secondary")
                    with gr.Column():
                        create_view_btn = gr.Button("ビューを作成", variant="primary")
                with gr.Row():                
                    create_view_result = gr.Markdown(visible=False)

                with gr.Accordion(label="AI分析と処理", open=False):
                    with gr.Row():
                        view_ai_model_input = gr.Dropdown(
                            label="モデル",
                            choices=[
                                "xai.grok-code-fast-1",
                                "xai.grok-3",
                                "xai.grok-3-fast",
                                "xai.grok-4",
                                "xai.grok-4-fast-non-reasoning",
                                "meta.llama-4-scout-17b-16e-instruct",
                            ],
                            value="xai.grok-code-fast-1",
                            interactive=True,
                        )
                    with gr.Row():
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
                            return view_name, col_df, ddl
                        else:
                            logger.warning(f"Row index {row_index} out of bounds, dataframe has {len(current_df)} rows")
                except Exception as e:
                    logger.error(f"Error in on_view_select: {e}")
                    logger.error(traceback.format_exc())
                
                return "", pd.DataFrame(), ""
            
            def refresh_view_list():
                try:
                    logger.info("ビュー一覧を取得ボタンがクリックされました")
                    yield gr.Markdown(value="⏳ ビュー一覧を取得中...", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["View Name", "Comments"]))
                    df = get_view_list(pool)
                    yield gr.Markdown(value="✅ 取得完了", visible=True), gr.Dataframe(value=df, visible=True)
                except Exception as e:
                    yield gr.Markdown(value=f"❌ 取得に失敗しました: {str(e)}", visible=True), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["View Name", "Comments"]))
            
            def drop_selected_view(view_name):
                """Drop the selected view and refresh list."""
                yield (
                    gr.Markdown(visible=True, value="⏳ ビューを削除中..."),
                    gr.Dataframe(visible=False, value=pd.DataFrame()),
                    "",
                    pd.DataFrame(),
                    "",
                    "",
                    "",
                )
                result = drop_view(pool, view_name)
                new_list = get_view_list(pool)
                status_md = gr.Markdown(visible=True, value=result)
                yield result, gr.Dataframe(value=new_list, visible=True), "", pd.DataFrame(), "", "", ""
            
            def execute_create_view_handler(sql):
                """Execute CREATE VIEW and refresh list."""
                yield gr.Markdown(visible=True, value="⏳ ビュー作成を実行中..."), gr.Dataframe(visible=False, value=pd.DataFrame(columns=["View Name", "Comments"]))
                result = execute_create_view(pool, sql)
                new_list = get_view_list(pool)
                status_md = gr.Markdown(visible=True, value=result)
                yield status_md, gr.Dataframe(value=new_list, visible=True)
            
            def clear_view_sql():
                """Clear the SQL input."""
                return ""
            
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
                outputs=[view_drop_result, view_list_df, selected_view_name, 
                        view_columns_df, view_ddl_text, view_join_text, view_where_text]
            )

            async def _view_join_where_ai_extract_async(model_name, ddl_text):
                from utils.chat_util import get_oci_region, get_compartment_id
                region = get_oci_region()
                compartment_id = get_compartment_id()
                if not region or not compartment_id:
                    return gr.Textbox(value=""), gr.Textbox(value="")
                try:
                    s = str(ddl_text or "").strip()
                    if not s:
                        return gr.Textbox(value=""), gr.Textbox(value="")
                    m = re.search(r"\b(SELECT|WITH)\b[\s\S]*", s, flags=re.IGNORECASE)
                    if m:
                        s = m.group(0)
                    from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                    client = AsyncOciOpenAI(
                        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                        auth=OciUserPrincipalAuth(),
                        compartment_id=compartment_id,
                    )
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
                    resp = await client.chat.completions.create(model=model_name, messages=messages)
                    join_text = ""
                    where_text = ""
                    if getattr(resp, "choices", None):
                        msg = resp.choices[0].message
                        out = msg.content if hasattr(msg, "content") else ""
                        s2 = re.sub(r"```+\w*", "", str(out or ""))
                        m2 = re.search(r"JOIN:\s*([\s\S]*?)\n\s*WHERE:\s*([\s\S]*)$", s2, flags=re.IGNORECASE)
                        if m2:
                            join_text = m2.group(1).strip()
                            where_text = m2.group(2).strip()
                    if not join_text:
                        join_text = "None"
                    if not where_text:
                        where_text = "None"
                    return gr.Textbox(value=join_text), gr.Textbox(value=where_text)
                except Exception:
                    return gr.Textbox(value="None"), gr.Textbox(value="None")

            def _view_join_where_ai_extract(model_name, ddl_text):
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Textbox(value="..."), gr.Textbox(value="...")
                    join_text, where_text = loop.run_until_complete(_view_join_where_ai_extract_async(model_name, ddl_text))
                    yield gr.Markdown(visible=True, value="✅ 分析完了"), join_text, where_text
                except Exception as e:
                    yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Textbox(value="Error"), gr.Textbox(value=str(e))
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
                outputs=[create_view_result, view_list_df]
            )
            
            clear_view_sql_btn.click(
                fn=clear_view_sql,
                outputs=[create_view_sql]
            )

            async def _view_ai_analyze_async(model_name, view_name, columns_df_input, ddl_text, create_sql_text, exec_result_text):
                from utils.chat_util import get_oci_region, get_compartment_id
                region = get_oci_region()
                compartment_id = get_compartment_id()
                if not region or not compartment_id:
                    return gr.Markdown(visible=True, value="ℹ️ OCI設定が不足しています")
                try:
                    import pandas as pd
                    from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                    if isinstance(columns_df_input, dict) and "data" in columns_df_input:
                        headers = columns_df_input.get("headers", [])
                        columns_df = pd.DataFrame(columns_df_input["data"], columns=headers)
                    elif isinstance(columns_df_input, pd.DataFrame):
                        columns_df = columns_df_input
                    else:
                        columns_df = pd.DataFrame()
                    preview = columns_df.head(10).to_markdown(index=False) if not columns_df.empty else ""
                    sql_part = str(create_sql_text or "").strip()
                    if not sql_part:
                        sql_part = str(ddl_text or "").strip()
                    result_part = str(exec_result_text or "").strip()
                    prompt = (
                        "以下のSQL/DDLと実行結果を分析してください。出力は次の3点に限定します。\n"
                        "1) エラー原因（該当する場合）\n"
                        "2) 解決方法（修正案や具体的手順）\n"
                        "3) 簡潔な結論（不要な詳細は省略）\n\n"
                        + ("SQL/DDL:\n```sql\n" + sql_part + "\n```\n" if sql_part else "")
                        + ("実行結果:\n" + result_part + "\n" if result_part else "")
                        + ("列情報プレビュー:\n" + preview + "\n" if preview else "")
                    )
                    client = AsyncOciOpenAI(
                        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                        auth=OciUserPrincipalAuth(),
                        compartment_id=compartment_id,
                    )
                    messages = [
                        {"role": "system", "content": "あなたはシニアDBエンジニアです。SQLと実行結果の故障診断に特化し、エラー原因と実行可能な修復策のみを簡潔に提示してください。不要な詳細は出力しないでください。"},
                        {"role": "user", "content": prompt},
                    ]
                    resp = await client.chat.completions.create(model=model_name, messages=messages)
                    text = ""
                    if getattr(resp, "choices", None):
                        msg = resp.choices[0].message
                        text = msg.content if hasattr(msg, "content") else ""
                    return gr.Markdown(visible=True, value=text or "分析結果が空です")
                except Exception as e:
                    return gr.Markdown(visible=True, value=f"❌ エラー: {e}")

            def view_ai_analyze(model_name, view_name, columns_df_input, ddl_text, create_sql_text, exec_result_text):
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Markdown(visible=False)
                    result_md = loop.run_until_complete(_view_ai_analyze_async(model_name, view_name, columns_df_input, ddl_text, create_sql_text, exec_result_text))
                    yield gr.Markdown(visible=True, value="✅ 完了"), result_md
                except Exception as e:
                    yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Markdown(visible=False)
                finally:
                    loop.close()

            view_ai_analyze_btn.click(
                fn=view_ai_analyze,
                inputs=[view_ai_model_input, selected_view_name, view_columns_df, view_ddl_text, create_view_sql, create_view_result],
                outputs=[view_ai_analyze_status_md, view_ai_result_md],
            )
        
        # Data Management Tab
        with gr.TabItem(label="データの管理"):
            # Feature 1: Table Data Display
            with gr.Accordion(label="1. テーブル・ビューデータの表示", open=True):
                data_refresh_btn = gr.Button("テーブル・ビュー一覧を取得", variant="primary")
                data_refresh_status = gr.Markdown(visible=False)
                
                with gr.Row():
                    with gr.Column():
                        data_table_select = gr.Dropdown(
                            label="テーブル・ビュー選択",
                            choices=[],
                            interactive=True,
                        )
                    with gr.Column():
                        data_limit_input = gr.Number(
                            label="取得件数",
                            value=100,
                            minimum=1,
                            maximum=10000,
                        )
                
                with gr.Row():
                    data_where_input = gr.Textbox(
                        label="WHERE条件（オプション）",
                        placeholder="例: status = 'A' AND created_at > SYSDATE - 7",
                        lines=2,
                        max_lines=5,
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
                    csv_file_input = gr.File(
                        label="CSVファイル",
                        file_types=[".csv"],
                        type="filepath",
                    )
                
                with gr.Row():
                    with gr.Column():
                        csv_table_select = gr.Dropdown(
                            label="アップロード先テーブル",
                            choices=[],
                            interactive=True,
                            visible=False,
                        )
                    with gr.Column():
                        csv_upload_mode = gr.Radio(
                            label="アップロードモード",
                            choices=["INSERT", "TRUNCATE & INSERT"],
                            value="INSERT",
                        )
                
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
            with gr.Accordion(label="3. SQL一括実行", open=False):
                sql_template_select = gr.Dropdown(
                    label="SQLテンプレート（オプション）",
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
                )
                
                data_sql_input = gr.Textbox(
                    label="SQL文（複数の文をセミコロンで区切って入力可能）",
                    placeholder="INSERT/UPDATE/DELETE/MERGE文を入力してください（注: SELECT文は禁止されています）\n例:\nINSERT INTO users (username, email, status) VALUES ('user1', 'user1@example.com', 'A');\nINSERT INTO users (username, email, status) VALUES ('user2', 'user2@example.com', 'A');\nUPDATE users SET status = 'A' WHERE user_id = 1;\nDELETE FROM users WHERE status = 'D';",
                    lines=10,
                    max_lines=15,
                    show_copy_button=True,
                )
                
                with gr.Row():
                    data_clear_btn = gr.Button("クリア", variant="secondary")
                    data_execute_btn = gr.Button("実行", variant="primary")

                with gr.Row():        
                    data_sql_status = gr.Markdown(visible=False)
                    data_sql_result = gr.Markdown(visible=False)

                with gr.Accordion(label="AI分析と処理", open=False):
                    data_ai_model_input = gr.Dropdown(
                        label="モデル",
                        choices=[
                            "xai.grok-code-fast-1",
                            "xai.grok-3",
                            "xai.grok-3-fast",
                            "xai.grok-4",
                            "xai.grok-4-fast-non-reasoning",
                            "meta.llama-4-scout-17b-16e-instruct",
                        ],
                        value="xai.grok-code-fast-1",
                        interactive=True,
                    )
                    data_ai_analyze_btn = gr.Button("AI分析", variant="primary")
                    data_ai_status_md = gr.Markdown(visible=False)
                    data_ai_result_md = gr.Markdown(visible=False)
            
            # Event Handlers
            def refresh_data_table_list():
                try:
                    logger.info("テーブル・ビュー一覧を取得ボタンがクリックされました")
                    yield gr.Markdown(value="⏳ テーブル・ビュー一覧を取得中...", visible=True), gr.Dropdown(choices=[]), gr.Dropdown(choices=[], visible=False)
                    data_names = get_table_list_for_data(pool)
                    upload_tables = get_table_list_for_upload(pool)
                    yield gr.Markdown(value="✅ 取得完了", visible=True), gr.Dropdown(choices=data_names, visible=True), gr.Dropdown(choices=upload_tables, visible=True)
                except Exception as e:
                    yield gr.Markdown(value=f"❌ 取得に失敗しました: {str(e)}", visible=True), gr.Dropdown(choices=[]), gr.Dropdown(choices=[], visible=False)
            
            def display_data(table_name, limit, where_clause):
                try:
                    yield gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(value="⏳ データを取得中...", visible=True)
                    df = display_table_data(pool, table_name, limit, where_clause)
                    if df.empty:
                        yield gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(value="✅ 取得完了(データなし)", visible=True)
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
                        style_value = ""
                    else:
                        col_widths = [max(5, int(100 * w / total)) for w in widths]
                        diff = 100 - sum(col_widths)
                        if diff != 0 and len(col_widths) > 0:
                            col_widths[0] = max(5, col_widths[0] + diff)
                        rules = ["#data_result_df table { table-layout: fixed; width: 100%; }"]
                        for idx, pct in enumerate(col_widths, start=1):
                            rules.append(f"#data_result_df table th:nth-child({idx}), #data_result_df table td:nth-child({idx}) {{ width: {pct}%; }}")
                        style_value = "<style>" + "\n".join(rules) + "</style>"

                    df_component = gr.Dataframe(
                        label=f"データ表示（件数: {len(df)}）",
                        interactive=False,
                        wrap=True,
                        visible=True,
                        value=df,
                        elem_id="data_result_df",
                    )
                    # style_component is removed
                    yield df_component, gr.Markdown(visible=False)
                except Exception as e:
                    yield gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(value=f"❌ データ取得に失敗しました: {str(e)}", visible=True)
            
            def upload_csv(file, table_name, mode):
                """Upload CSV file."""
                yield gr.Dataframe(visible=False, value=pd.DataFrame()), gr.Markdown(visible=True, value="⏳ CSVアップロードを実行中..."), gr.Markdown(visible=False)
                preview, result = upload_csv_data(pool, file, table_name, mode)
                status_md = gr.Markdown(visible=True, value=result)
                yield preview, gr.Markdown(visible=False), status_md
            
            def execute_sql(sql):
                """Execute SQL statements."""
                yield gr.Markdown(visible=True, value="⏳ SQL一括実行中..."), gr.Markdown(visible=False)
                result = execute_data_sql(pool, sql)
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

            async def _data_ai_analyze_async(model_name, obj_name, limit_value, where_text, df_input):
                from utils.chat_util import get_oci_region, get_compartment_id
                region = get_oci_region()
                compartment_id = get_compartment_id()
                if not region or not compartment_id:
                    return gr.Markdown(visible=True, value="ℹ️ OCI設定が不足しています")
                try:
                    import pandas as pd
                    from oci_openai import AsyncOciOpenAI, OciUserPrincipalAuth
                    if isinstance(df_input, dict) and "data" in df_input:
                        headers = df_input.get("headers", [])
                        df = pd.DataFrame(df_input["data"], columns=headers)
                    elif isinstance(df_input, pd.DataFrame):
                        df = df_input
                    else:
                        df = pd.DataFrame()
                    preview = df.head(20).to_markdown(index=False) if not df.empty else ""
                    where_part = f" WHERE {str(where_text).strip()}" if where_text and str(where_text).strip() else ""
                    sql_text = f"SELECT * FROM ADMIN.{str(obj_name or '').upper()}{where_part} FETCH FIRST {int(limit_value or 0)} ROWS ONLY" if obj_name else ""
                    prompt = (
                        "以下のSQLと実行結果を分析してください。出力は次の3点に限定します。\n"
                        "1) エラー原因（該当する場合）\n"
                        "2) 解決方法（修正案や具体的手順）\n"
                        "3) 簡潔な結論（不要な詳細は省略）\n\n"
                        + ("SQL:\n```sql\n" + sql_text + "\n```\n" if sql_text else "")
                        + ("結果プレビュー:\n" + preview + "\n" if preview else "")
                    )
                    client = AsyncOciOpenAI(
                        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
                        auth=OciUserPrincipalAuth(),
                        compartment_id=compartment_id,
                    )
                    messages = [
                        {"role": "system", "content": "あなたはシニアDBエンジニアです。SQLと実行結果の故障診断に特化し、エラー原因と実行可能な修復策のみを簡潔に提示してください。不要な詳細は出力しないでください。"},
                        {"role": "user", "content": prompt},
                    ]
                    resp = await client.chat.completions.create(model=model_name, messages=messages)
                    text = ""
                    if getattr(resp, "choices", None):
                        msg = resp.choices[0].message
                        text = msg.content if hasattr(msg, "content") else ""
                    return gr.Markdown(visible=True, value=text or "分析結果が空です")
                except Exception as e:
                    return gr.Markdown(visible=True, value=f"❌ エラー: {e}")

            def data_ai_analyze(model_name, obj_name, limit_value, where_text, df_input):
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    yield gr.Markdown(visible=True, value="⏳ AI分析を実行中..."), gr.Markdown(visible=False)
                    result_md = loop.run_until_complete(_data_ai_analyze_async(model_name, obj_name, limit_value, where_text, df_input))
                    yield gr.Markdown(visible=True, value="✅ 完了"), result_md
                except Exception as e:
                    yield gr.Markdown(visible=True, value=f"❌ エラー: {e}"), gr.Markdown(visible=False)
                finally:
                    loop.close()

            data_ai_analyze_btn.click(
                fn=data_ai_analyze,
                inputs=[data_ai_model_input, data_table_select, data_limit_input, data_where_input, data_display],
                outputs=[data_ai_status_md, data_ai_result_md],
            )
