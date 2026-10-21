-- Compact per-point summaries that don't fit metric/stat rows, such as what streaming
-- answers felt while long prompts arrived (lesson 44).
CREATE TABLE point_series (
    point_id   INTEGER NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    data_json  TEXT NOT NULL,
    PRIMARY KEY (point_id, name)
);
