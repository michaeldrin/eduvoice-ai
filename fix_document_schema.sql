-- Fix missing columns in document table
ALTER TABLE document ADD COLUMN IF NOT EXISTS text_filename VARCHAR(255);
ALTER TABLE document ADD COLUMN IF NOT EXISTS audio_filename VARCHAR(255);
ALTER TABLE document ADD COLUMN IF NOT EXISTS auto_processed BOOLEAN DEFAULT FALSE;
ALTER TABLE document ADD COLUMN IF NOT EXISTS language VARCHAR(10);
ALTER TABLE document ADD COLUMN IF NOT EXISTS translated_summary TEXT;
ALTER TABLE document ADD COLUMN IF NOT EXISTS translated_content TEXT;