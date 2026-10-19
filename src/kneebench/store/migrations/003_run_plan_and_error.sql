-- Planned point count and failure reason, so progress and failures come from the database.
ALTER TABLE runs ADD COLUMN planned_points INTEGER;
ALTER TABLE runs ADD COLUMN error TEXT;
