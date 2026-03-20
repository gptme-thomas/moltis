-- Add context_command column to projects table
-- When set, the command is run at session start and its stdout
-- is appended to the project context for system prompt injection.

ALTER TABLE projects ADD COLUMN context_command TEXT;
