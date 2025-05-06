-- Add translation fields to document table
ALTER TABLE document ADD COLUMN has_translation BOOLEAN DEFAULT FALSE;
ALTER TABLE document ADD COLUMN translated_text TEXT;
ALTER TABLE document ADD COLUMN translation_language VARCHAR(10);
ALTER TABLE document ADD COLUMN language_guide TEXT;