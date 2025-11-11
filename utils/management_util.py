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
                # Query to get table list with comments
                sql = """
                SELECT 
                    t.table_name AS "Table Name",
                    NVL(t.num_rows, 0) AS "Rows",
                    NVL(c.comments, ' ') AS "Comments"
                FROM all_tables t
                LEFT JOIN all_tab_comments c ON t.table_name = c.table_name AND t.owner = c.owner
                WHERE t.owner = 'ADMIN'
                ORDER BY t.table_name
                """
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                if rows:
                    columns = [desc[0] for desc in cursor.description]
                    df = pd.DataFrame(rows, columns=columns)
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


def build_management_tab(pool):
    """Build the Management Function tab with three sub-functions.
    
    Args:
        pool: Oracle database connection pool
    """
    with gr.Tabs():
        # Table Management Tab
        with gr.TabItem(label="Table管理"):
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
            def on_table_select(evt: gr.SelectData):
                """Handle table row selection.
                
                Always extracts the table name from the first column (Table Name),
                regardless of which column in the row was clicked.
                """
                try:
                    row_index = evt.index[0]
                    logger.info(f"Row clicked: {row_index}, Value clicked: {evt.value}")
                    
                    if row_index >= 0:
                        # Get the current dataframe value
                        current_df = table_list_df.value
                        logger.info(f"Dataframe type: {type(current_df)}")
                        
                        if current_df is not None:
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
        with gr.TabItem(label="View管理"):
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
            def on_view_select(evt: gr.SelectData):
                """Handle view row selection.
                
                Always extracts the view name from the first column (View Name),
                regardless of which column in the row was clicked.
                """
                try:
                    row_index = evt.index[0]
                    logger.info(f"Row clicked: {row_index}, Value clicked: {evt.value}")
                    
                    if row_index >= 0:
                        # Get the current dataframe value
                        current_df = view_list_df.value
                        logger.info(f"Dataframe type: {type(current_df)}")
                        
                        if current_df is not None:
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
        with gr.TabItem(label="Data管理"):
            gr.Markdown("## Data管理")
            gr.Markdown("データ管理機能（実装予定）")
            
            with gr.Accordion(label="データ操作", open=True):
                with gr.Row():
                    data_table_select = gr.Dropdown(
                        label="テーブル選択",
                        choices=[],
                        interactive=True,
                    )
                    data_limit_input = gr.Number(
                        label="取得件数",
                        value=100,
                        minimum=1,
                        maximum=10000,
                    )
                
                data_display = gr.Dataframe(
                    label="データ表示",
                    interactive=False,
                )
            
            with gr.Accordion(label="データ編集", open=False):
                data_sql_input = gr.Textbox(
                    label="SQL文",
                    placeholder="INSERT/UPDATE/DELETE文を入力してください",
                    lines=5,
                )
                
                with gr.Row():
                    data_execute_btn = gr.Button("実行", variant="primary")
                    data_clear_btn = gr.Button("クリア", variant="secondary")
                
                data_output = gr.Textbox(
                    label="実行結果",
                    lines=5,
                    interactive=False,
                )
