"""Management utility module for SQL Assist.

This module provides UI components for database management functions including
Table Management, View Management, and Data Management.
"""

import logging
import traceback

import gradio as gr
import pandas as pd
from oracledb import DatabaseError

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
                WHERE t.owner = 'ADMIN'
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
                        data.append((tn, cnt, comment))
                    df = pd.DataFrame(data, columns=["Table Name", "Rows", "Comments"])
                    logger.info(f"Retrieved {len(df)} tables for ADMIN user")
                    return df
                else:
                    logger.info("No tables found for ADMIN user")
                    return pd.DataFrame(columns=["Table Name", "Rows", "Comments"])
    except Exception as e:
        logger.error(f"Error getting table list: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"テーブル一覧の取得に失敗しました: {str(e)}")
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
                    columns = [desc[0] for desc in cursor.description]
                    col_df = pd.DataFrame(col_rows, columns=columns)
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
                table_comment = table_comment_result[0] if table_comment_result and table_comment_result[0] else None
                
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
        gr.Warning(f"テーブル詳細の取得に失敗しました: {str(e)}")
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
        gr.Warning("テーブル名を指定してください")
        return "エラー: テーブル名が指定されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                sql = f"DROP TABLE ADMIN.{table_name.upper()} PURGE"
                cursor.execute(sql)
                conn.commit()
                logger.info(f"Table dropped: {table_name}")
                gr.Info(f"テーブル '{table_name}' を削除しました")
                return f"成功: テーブル '{table_name}' を削除しました"
    except Exception as e:
        logger.error(f"Error dropping table: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"テーブルの削除に失敗しました: {str(e)}")
        return f"エラー: {str(e)}"


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
        gr.Warning("CREATE TABLE文を入力してください")
        return "エラー: SQL文が入力されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Split SQL statements by semicolon
                sql_statements = [stmt.strip() for stmt in create_sql.split(';') if stmt.strip()]
                
                if not sql_statements:
                    gr.Warning("SQL文が空です")
                    return "エラー: SQL文が空です"
                
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
                    result = f"部分的に成功: {executed_count}/{len(sql_statements)}件の文を実行しました\n\nエラー:\n" + "\n".join(error_messages)
                    gr.Warning(result)
                    logger.warning(f"Partial success: {executed_count}/{len(sql_statements)} statements executed")
                    return result
                else:
                    result = f"成功: {executed_count}件の文をすべて実行しました"
                    gr.Info(result)
                    logger.info(f"All {executed_count} statements executed successfully")
                    return result
                    
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"テーブルの作成に失敗しました: {str(e)}")
        return f"エラー: {str(e)}"


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
                WHERE v.owner = 'ADMIN'
                ORDER BY v.view_name
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                if rows:
                    columns = [desc[0] for desc in cursor.description]
                    df = pd.DataFrame(rows, columns=columns)
                    logger.info(f"Retrieved {len(df)} views for ADMIN user")
                    return df
                else:
                    logger.info("No views found for ADMIN user")
                    return pd.DataFrame(columns=["View Name", "Comments"])
    except Exception as e:
        logger.error(f"Error getting view list: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"ビュー一覧の取得に失敗しました: {str(e)}")
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
                    columns = [desc[0] for desc in cursor.description]
                    col_df = pd.DataFrame(col_rows, columns=columns)
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
                view_comment = view_comment_result[0] if view_comment_result and view_comment_result[0] else None
                
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
                
                # Append COMMENT statements to the DDL
                if create_sql:
                    create_sql = create_sql.rstrip()
                    if not create_sql.endswith(';'):
                        create_sql += ';'
                    create_sql += '\n'
                    
                    # Add view comment
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
        gr.Warning(f"ビュー詳細の取得に失敗しました: {str(e)}")
        return pd.DataFrame(columns=["Column Name", "Data Type", "Nullable", "Comments"]), ""


def drop_view(pool, view_name):
    """Drop a view.
    
    Args:
        pool: Oracle database connection pool
        view_name: Name of the view to drop
        
    Returns:
        str: Result message
    """
    if not view_name:
        gr.Warning("ビュー名を指定してください")
        return "エラー: ビュー名が指定されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                sql = f"DROP VIEW ADMIN.{view_name.upper()}"
                cursor.execute(sql)
                conn.commit()
                logger.info(f"View dropped: {view_name}")
                gr.Info(f"ビュー '{view_name}' を削除しました")
                return f"成功: ビュー '{view_name}' を削除しました"
    except Exception as e:
        logger.error(f"Error dropping view: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"ビューの削除に失敗しました: {str(e)}")
        return f"エラー: {str(e)}"


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
        gr.Warning("CREATE VIEW文を入力してください")
        return "エラー: SQL文が入力されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Split SQL statements by semicolon
                sql_statements = [stmt.strip() for stmt in create_sql.split(';') if stmt.strip()]
                
                if not sql_statements:
                    gr.Warning("SQL文が空です")
                    return "エラー: SQL文が空です"
                
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
                    result = f"部分的に成功: {executed_count}/{len(sql_statements)}件の文を実行しました\n\nエラー:\n" + "\n".join(error_messages)
                    gr.Warning(result)
                    logger.warning(f"Partial success: {executed_count}/{len(sql_statements)} statements executed")
                    return result
                else:
                    result = f"成功: {executed_count}件の文をすべて実行しました"
                    gr.Info(result)
                    logger.info(f"All {executed_count} statements executed successfully")
                    return result
                    
    except Exception as e:
        logger.error(f"Error executing SQL: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"ビューの作成に失敗しました: {str(e)}")
        return f"エラー: {str(e)}"


def get_table_list_for_data(pool):
    """Get list of table and view names for data management dropdown.
    
    Args:
        pool: Oracle database connection pool
        
    Returns:
        list: List of table and view names
    """
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Query to get both tables and views
                sql = """
                SELECT table_name FROM all_tables WHERE owner = 'ADMIN'
                UNION
                SELECT view_name FROM all_views WHERE owner = 'ADMIN'
                ORDER BY 1
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                names = [row[0] for row in rows] if rows else []
                logger.info(f"Retrieved {len(names)} tables and views for data management")
                return names
    except Exception as e:
        logger.error(f"Error getting table/view list for data: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"テーブル・ビュー一覧の取得に失敗しました: {str(e)}")
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
        gr.Warning("テーブルまたはビューを選択してください")
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
                    columns = [desc[0] for desc in cursor.description]
                    df = pd.DataFrame(rows, columns=columns)
                    logger.info(f"Retrieved {len(df)} rows from {table_name}")
                    gr.Info(f"{len(df)}件のデータを取得しました")
                    return df
                else:
                    logger.info(f"No data found in {table_name}")
                    gr.Info("データが見つかりませんでした")
                    return pd.DataFrame()
                    
    except Exception as e:
        logger.error(f"Error displaying table/view data: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"データの取得に失敗しました: {str(e)}")
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
        gr.Warning("CSVファイルを選択してください")
        return pd.DataFrame(), "エラー: ファイルが選択されていません"
    
    if not table_name:
        gr.Warning("テーブルを選択してください")
        return pd.DataFrame(), "エラー: テーブルが選択されていません"
    
    try:
        # Read CSV file
        import csv
        df = pd.read_csv(file.name)
        logger.info(f"CSV file loaded: {len(df)} rows, {len(df.columns)} columns")
        
        if df.empty:
            gr.Warning("CSVファイルが空です")
            return df, "エラー: CSVファイルにデータがありません"
        
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
                    gr.Error(f"テーブル '{table_name}' が見つかりません")
                    return preview_df, f"エラー: テーブル '{table_name}' が見つかりません"
                
                # Match CSV columns to table columns (case-insensitive)
                csv_columns = df.columns.tolist()
                column_mapping = {}
                for csv_col in csv_columns:
                    for tbl_col in table_columns:
                        if csv_col.upper() == tbl_col.upper():
                            column_mapping[csv_col] = tbl_col
                            break
                
                if not column_mapping:
                    gr.Error("CSVの列名がテーブルの列名と一致しません")
                    return preview_df, "エラー: CSVの列名がテーブルの列名と一致しません"
                
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
                result = f"成功: {success_count}件のデータを挿入しました"
                if error_count > 0:
                    result += f"\n\n警告: {error_count}件のエラーが発生しました\n" + "\n".join(error_messages)
                    if error_count > 5:
                        result += f"\n... 他 {error_count - 5} 件のエラー"
                    gr.Warning(result)
                else:
                    gr.Info(result)
                
                logger.info(f"CSV upload completed: {success_count} success, {error_count} errors")
                return preview_df, result
                
    except Exception as e:
        logger.error(f"Error uploading CSV: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"CSVアップロードに失敗しました: {str(e)}")
        return pd.DataFrame(), f"エラー: {str(e)}"


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
        gr.Warning("SQL文を入力してください")
        return "エラー: SQL文が入力されていません"
    
    try:
        with pool.acquire() as conn:
            with conn.cursor() as cursor:
                # Split SQL statements by semicolon
                statements = [stmt.strip() for stmt in sql_statements.split(';') if stmt.strip()]
                
                if not statements:
                    gr.Warning("SQL文が空です")
                    return "エラー: SQL文が空です"
                
                # Check for SELECT statements (prohibited)
                for idx, sql_stmt in enumerate(statements, 1):
                    # Remove leading/trailing whitespace and convert to uppercase for checking
                    stmt_upper = sql_stmt.strip().upper()
                    # Check if statement starts with SELECT (including WITH clauses)
                    if stmt_upper.startswith('SELECT') or (stmt_upper.startswith('WITH') and 'SELECT' in stmt_upper):
                        error_msg = f"禁止された操作: SELECT文は実行できません。\n文{idx}: {sql_stmt[:100]}..."
                        logger.warning(f"SELECT statement prohibited: {sql_stmt[:100]}...")
                        gr.Error(error_msg)
                        return f"エラー: {error_msg}"
                
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
                    result = f"部分的に成功: {executed_count}/{len(statements)}件の文を実行しました（{total_rows}行に影響）\n\nエラー:\n" + "\n".join(error_messages)
                    gr.Warning(result)
                    logger.warning(f"Partial success: {executed_count}/{len(statements)} statements executed")
                    return result
                else:
                    total_rows = sum(affected_rows)
                    result = f"成功: {executed_count}件の文をすべて実行しました（{total_rows}行に影響）"
                    gr.Info(result)
                    logger.info(f"All {executed_count} statements executed successfully")
                    return result
                    
    except Exception as e:
        logger.error(f"Error executing data SQL: {e}")
        logger.error(traceback.format_exc())
        gr.Error(f"SQL実行に失敗しました: {str(e)}")
        return f"エラー: {str(e)}"


def build_management_tab(pool):
    """Build the Management Function tab with three sub-functions.
    
    Args:
        pool: Oracle database connection pool
    """
    with gr.Tabs():
        # Table Management Tab
        with gr.TabItem(label="Tableの管理"):
            # Feature 1: Table List
            with gr.Accordion(label="1. テーブル一覧", open=True):
                table_refresh_btn = gr.Button("テーブル一覧を更新", variant="primary")
                table_list_df = gr.Dataframe(
                    label="テーブル一覧(行をクリックして詳細を表示)",
                    interactive=False,
                    wrap=True,
                    value=get_table_list(pool),
                )
            
            # Feature 2: Table Details and Drop
            with gr.Accordion(label="2. テーブル詳細と削除", open=True):
                selected_table_name = gr.Textbox(
                    label="選択されたテーブル名",
                    interactive=False,
                )
                
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
                            max_lines=30,
                            interactive=False,
                            show_copy_button=True,
                        )
                
                table_drop_btn = gr.Button("選択したテーブルを削除", variant="stop")
                table_drop_result = gr.Textbox(
                    label="削除結果",
                    interactive=False,
                )
            
            # Feature 3: Create Table
            with gr.Accordion(label="3. テーブル作成", open=False):
                create_table_sql = gr.Textbox(
                    label="CREATE TABLE SQL文（複数の文をセミコロンで区切って入力可能）",
                    placeholder="CREATE TABLE文を入力してください\n例:\nCREATE TABLE test_table (\n  id NUMBER PRIMARY KEY,\n  name VARCHAR2(100)\n);\n\nCOMMENT ON TABLE test_table IS 'テストテーブル';\nCOMMENT ON COLUMN test_table.id IS 'ID';\nCOMMENT ON COLUMN test_table.name IS '名称';",
                    lines=10,
                    max_lines=30,
                    show_copy_button=True,
                )
                
                with gr.Row():
                    clear_sql_btn = gr.Button("クリア", variant="secondary")
                    create_table_btn = gr.Button("テーブルを作成", variant="primary")
                
                create_table_result = gr.Textbox(
                    label="作成結果",
                    interactive=False,
                )
            
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
                    gr.Error(f"テーブル選択エラー: {str(e)}")
                
                return "", pd.DataFrame(), ""
            
            def refresh_table_list():
                """Refresh the table list."""
                return get_table_list(pool)
            
            def drop_selected_table(table_name):
                """Drop the selected table and refresh list."""
                result = drop_table(pool, table_name)
                new_list = get_table_list(pool)
                return result, new_list, "", pd.DataFrame(), ""
            
            def execute_create(sql):
                """Execute CREATE TABLE and refresh list."""
                result = execute_create_table(pool, sql)
                new_list = get_table_list(pool)
                return result, new_list
            
            def clear_sql():
                """Clear the SQL input."""
                return ""
            
            # Wire up events
            table_refresh_btn.click(
                fn=refresh_table_list,
                outputs=[table_list_df]
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
        
        # View Management Tab
        with gr.TabItem(label="Viewの管理"):
            # Feature 1: View List
            with gr.Accordion(label="1. ビュー一覧", open=True):
                view_refresh_btn = gr.Button("ビュー一覧を更新", variant="primary")
                view_list_df = gr.Dataframe(
                    label="ビュー一覧(行をクリックして詳細を表示)",
                    interactive=False,
                    wrap=True,
                    value=get_view_list(pool),
                )
            
            # Feature 2: View Details and Drop
            with gr.Accordion(label="2. ビュー詳細と削除", open=True):
                selected_view_name = gr.Textbox(
                    label="選択されたビュー名",
                    interactive=False,
                )
                
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
                            max_lines=30,
                            interactive=False,
                            show_copy_button=True,
                        )
                
                view_drop_btn = gr.Button("選択したビューを削除", variant="stop")
                view_drop_result = gr.Textbox(
                    label="削除結果",
                    interactive=False,
                )
            
            # Feature 3: Create View
            with gr.Accordion(label="3. ビュー作成", open=False):
                create_view_sql = gr.Textbox(
                    label="CREATE VIEW SQL文（複数の文をセミコロンで区切って入力可能）",
                    placeholder="CREATE VIEW文を入力してください\n例:\nCREATE VIEW test_view AS\nSELECT id, name FROM test_table;\n\nCOMMENT ON TABLE test_view IS 'テストビュー';\nCOMMENT ON COLUMN test_view.id IS 'ID';\nCOMMENT ON COLUMN test_view.name IS '名称';",
                    lines=10,
                    max_lines=30,
                    show_copy_button=True,
                )
                
                with gr.Row():
                    clear_view_sql_btn = gr.Button("クリア", variant="secondary")
                    create_view_btn = gr.Button("ビューを作成", variant="primary")
                
                create_view_result = gr.Textbox(
                    label="作成結果",
                    interactive=False,
                )
            
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
                    gr.Error(f"ビュー選択エラー: {str(e)}")
                
                return "", pd.DataFrame(), ""
            
            def refresh_view_list():
                """Refresh the view list."""
                return get_view_list(pool)
            
            def drop_selected_view(view_name):
                """Drop the selected view and refresh list."""
                result = drop_view(pool, view_name)
                new_list = get_view_list(pool)
                return result, new_list, "", pd.DataFrame(), ""
            
            def execute_create_view_handler(sql):
                """Execute CREATE VIEW and refresh list."""
                result = execute_create_view(pool, sql)
                new_list = get_view_list(pool)
                return result, new_list
            
            def clear_view_sql():
                """Clear the SQL input."""
                return ""
            
            # Wire up events
            view_refresh_btn.click(
                fn=refresh_view_list,
                outputs=[view_list_df]
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
                        view_columns_df, view_ddl_text]
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
        
        # Data Management Tab
        with gr.TabItem(label="Dataの管理"):
            # Feature 1: Table Data Display
            with gr.Accordion(label="1. テーブル・ビューデータの表示", open=True):
                data_refresh_btn = gr.Button("テーブル・ビュー一覧を更新", variant="primary")
                
                with gr.Row():
                    data_table_select = gr.Dropdown(
                        label="テーブル・ビュー選択",
                        choices=get_table_list_for_data(pool),
                        interactive=True,
                    )
                    data_limit_input = gr.Number(
                        label="取得件数",
                        value=100,
                        minimum=1,
                        maximum=10000,
                    )
                
                data_where_input = gr.Textbox(
                    label="WHERE条件（オプション）",
                    placeholder="例: status = 'A' AND created_at > SYSDATE - 7",
                    lines=2,
                )
                
                data_display_btn = gr.Button("データを表示", variant="primary")
                
                data_display_info = gr.Markdown(
                    value="ℹ️ テーブルまたはビューを選択して「データを表示」ボタンをクリックしてください",
                    visible=True,
                )
                data_display = gr.Dataframe(
                    label="データ表示",
                    interactive=False,
                    wrap=True,
                    visible=False,
                    value=pd.DataFrame(),
                )
            
            # Feature 2: CSV Upload
            with gr.Accordion(label="2. CSVアップロード", open=False):
                csv_file_input = gr.File(
                    label="CSVファイル",
                    file_types=[".csv"],
                    type="filepath",
                )
                
                with gr.Row():
                    csv_table_select = gr.Dropdown(
                        label="アップロード先テーブル",
                        choices=get_table_list_for_data(pool),
                        interactive=True,
                    )
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
                csv_upload_result = gr.Textbox(
                    label="アップロード結果",
                    lines=5,
                    interactive=False,
                )
            
            # Feature 3: SQL Bulk Execution
            with gr.Accordion(label="3. SQL一括実行", open=False):
                sql_template_select = gr.Dropdown(
                    label="SQLテンプレート（オプション）",
                    choices=[
                        "INSERT - 単一行",
                        "INSERT - 複数行",
                        "UPDATE",
                        "DELETE",
                        "MERGE",
                    ],
                    interactive=True,
                )
                
                data_sql_input = gr.Textbox(
                    label="SQL文（複数の文をセミコロンで区切って入力可能）",
                    placeholder="INSERT/UPDATE/DELETE/MERGE文を入力してください（注: SELECT文は禁止されています）\n例:\nINSERT INTO users (username, email, status) VALUES ('user1', 'user1@example.com', 'A');\nINSERT INTO users (username, email, status) VALUES ('user2', 'user2@example.com', 'A');\nUPDATE users SET status = 'A' WHERE user_id = 1;\nDELETE FROM users WHERE status = 'D';",
                    lines=10,
                    max_lines=30,
                    show_copy_button=True,
                )
                
                with gr.Row():
                    data_clear_btn = gr.Button("クリア", variant="secondary")
                    data_execute_btn = gr.Button("実行", variant="primary")
                
                data_sql_result = gr.Textbox(
                    label="実行結果",
                    lines=5,
                    interactive=False,
                )

            
            # Event Handlers
            def refresh_data_table_list():
                """Refresh table and view list for data management."""
                tables = get_table_list_for_data(pool)
                return gr.Dropdown(choices=tables), gr.Dropdown(choices=tables)
            
            def display_data(table_name, limit, where_clause):
                """Display table data."""
                df = display_table_data(pool, table_name, limit, where_clause)
                if df.empty:
                    return gr.Markdown(visible=True), gr.Dataframe(visible=False, value=pd.DataFrame())
                else:
                    return gr.Markdown(visible=False), gr.Dataframe(visible=True, value=df)
            
            def upload_csv(file, table_name, mode):
                """Upload CSV file."""
                preview, result = upload_csv_data(pool, file, table_name, mode)
                return preview, result
            
            def execute_sql(sql):
                """Execute SQL statements."""
                return execute_data_sql(pool, sql)
            
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
                outputs=[data_table_select, csv_table_select]
            )
            
            data_display_btn.click(
                fn=display_data,
                inputs=[data_table_select, data_limit_input, data_where_input],
                outputs=[data_display_info, data_display]
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
                outputs=[csv_preview, csv_upload_result]
            )
            
            data_execute_btn.click(
                fn=execute_sql,
                inputs=[data_sql_input],
                outputs=[data_sql_result]
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
