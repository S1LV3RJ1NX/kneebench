-- Quality gate (lesson 54): the same questions asked before and after a change. The set is
-- stored whole, so the after-change pass asks exactly what the before pass asked.
CREATE TABLE quality_sets (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    digest      TEXT NOT NULL UNIQUE,
    text        TEXT NOT NULL,
    items       INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE quality_runs (
    id             INTEGER PRIMARY KEY,
    set_id         INTEGER NOT NULL REFERENCES quality_sets(id),
    endpoint_id    INTEGER NOT NULL REFERENCES endpoints(id),
    experiment_id  INTEGER REFERENCES experiments(id) ON DELETE SET NULL,
    role           TEXT NOT NULL,  -- before, floor (before, asked again) or after
    model          TEXT NOT NULL,
    label          TEXT,
    concurrency    INTEGER NOT NULL,
    status         TEXT NOT NULL,
    error          TEXT,
    created_at     TEXT NOT NULL,
    finished_at    TEXT
);

CREATE TABLE quality_answers (
    run_id         INTEGER NOT NULL REFERENCES quality_runs(id) ON DELETE CASCADE,
    item_id        TEXT NOT NULL,
    category       TEXT NOT NULL,
    text           TEXT,
    correct        INTEGER NOT NULL,
    finish_reason  TEXT,
    output_tokens  INTEGER,
    error          TEXT,
    PRIMARY KEY (run_id, item_id)
);

-- The pass rule is written before the before-change answers are recorded (lesson 54 section 7).
ALTER TABLE experiments ADD COLUMN quality_rule_json TEXT;
