-- Guided tuning: one proposed server change at a time, applied by the user, then verified by a
-- short run and compared with the run that proposed it. Deleting a run keeps the ledger.
ALTER TABLE endpoints ADD COLUMN deploy_style TEXT;

CREATE TABLE experiments (
    id             INTEGER PRIMARY KEY,
    endpoint_id    INTEGER NOT NULL REFERENCES endpoints(id),
    base_run_id    INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    verify_run_id  INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    rule_id        TEXT NOT NULL,
    title          TEXT NOT NULL,
    reason         TEXT NOT NULL,
    flag           TEXT NOT NULL,
    setting        TEXT,
    value          TEXT,
    expected       TEXT,
    risk           TEXT,
    status         TEXT NOT NULL,
    check_json     TEXT,
    outcome        TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
