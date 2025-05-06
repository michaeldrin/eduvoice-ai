-- Add attachment columns to chat_message table

-- Add has_attachment column
ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS has_attachment BOOLEAN DEFAULT FALSE;

-- Add attachment_filename column
ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS attachment_filename VARCHAR(255);

-- Add attachment_original_filename column
ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS attachment_original_filename VARCHAR(255);

-- Add attachment_type column
ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS attachment_type VARCHAR(50);

-- Add attachment_text column
ALTER TABLE chat_message ADD COLUMN IF NOT EXISTS attachment_text TEXT;

-- Add learning_suggestions column to document table
ALTER TABLE document ADD COLUMN IF NOT EXISTS learning_suggestions TEXT;