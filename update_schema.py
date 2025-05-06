"""
Database schema update script to add attachment fields to chat_message table.
"""
import os
import sys
import logging
from sqlalchemy import create_engine, MetaData, Table, Column, Boolean, String, Text
from sqlalchemy.sql import text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get database URL from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set")
    sys.exit(1)

try:
    # Create SQLAlchemy engine
    engine = create_engine(DATABASE_URL)
    
    # Create metadata object
    metadata = MetaData()
    
    # Reflect existing tables
    logger.info("Reflecting existing database tables...")
    metadata.reflect(bind=engine)
    
    # Check if chat_message table exists
    if 'chat_message' not in metadata.tables:
        logger.error("chat_message table does not exist in the database")
        sys.exit(1)
    
    # Get the chat_message table
    chat_message = metadata.tables['chat_message']
    
    # List of columns we want to add
    columns_to_add = [
        {'name': 'has_attachment', 'type': Boolean(), 'default': 'FALSE', 'nullable': True},
        {'name': 'attachment_filename', 'type': String(255), 'nullable': True},
        {'name': 'attachment_original_filename', 'type': String(255), 'nullable': True},
        {'name': 'attachment_type', 'type': String(50), 'nullable': True},
        {'name': 'attachment_text', 'type': Text(), 'nullable': True}
    ]
    
    # Check which columns need to be added
    columns_to_execute = []
    for column in columns_to_add:
        if column['name'] not in chat_message.columns:
            logger.info(f"Will add column: {column['name']}")
            columns_to_execute.append(column)
        else:
            logger.info(f"Column {column['name']} already exists, skipping")
    
    # Add columns if needed
    if columns_to_execute:
        connection = engine.connect()
        
        # Begin transaction
        trans = connection.begin()
        
        try:
            # Add each missing column
            for column in columns_to_execute:
                column_name = column['name']
                column_type = column['type']
                nullable = column.get('nullable', True)
                default = column.get('default')
                
                # Build SQL statement
                sql = f"ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS {column_name}"
                
                # Add type information
                if isinstance(column_type, Boolean):
                    sql += " BOOLEAN"
                elif isinstance(column_type, String):
                    sql += f" VARCHAR({column_type.length})"
                elif isinstance(column_type, Text):
                    sql += " TEXT"
                
                # Add constraints
                if not nullable:
                    if default:
                        sql += f" DEFAULT {default} NOT NULL"
                    else:
                        sql += " NOT NULL"
                elif default:
                    sql += f" DEFAULT {default}"
                
                # Execute the query
                logger.info(f"Executing: {sql}")
                connection.execute(text(sql))
            
            # Commit transaction
            trans.commit()
            logger.info("Schema update completed successfully")
            
        except Exception as e:
            # Rollback transaction on error
            trans.rollback()
            logger.error(f"Error updating schema: {e}")
            sys.exit(1)
        finally:
            # Close connection
            connection.close()
    else:
        logger.info("No schema changes needed")

except Exception as e:
    logger.error(f"Unexpected error: {e}")
    sys.exit(1)

logger.info("Schema update script completed")