-- API keys are never stored in any table.

CREATE TABLE endpoints (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    base_url    TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);

CREATE TABLE snapshots (
    id                 INTEGER PRIMARY KEY,
    endpoint_id        INTEGER NOT NULL REFERENCES endpoints(id),
    taken_at           TEXT NOT NULL,
    vllm_version       TEXT,
    served_model       TEXT,
    max_model_len      INTEGER,
    hf_model_id        TEXT,
    models_json        TEXT NOT NULL,
    cache_config_json  TEXT,
    hf_config_json     TEXT,
    safetensors_json   TEXT,
    metrics_text_z     BLOB
);

-- The server as the user described it; source_json says what was detected vs entered.
CREATE TABLE setups (
    id              INTEGER PRIMARY KEY,
    endpoint_id     INTEGER NOT NULL REFERENCES endpoints(id),
    snapshot_id     INTEGER NOT NULL REFERENCES snapshots(id),
    gpu_json        TEXT NOT NULL,
    gpu_count       INTEGER NOT NULL,
    max_num_seqs    INTEGER,
    enforce_eager   INTEGER,
    source_json     TEXT NOT NULL
);

CREATE TABLE workloads (
    id              INTEGER PRIMARY KEY,
    product_type    TEXT NOT NULL,
    shapes_json     TEXT NOT NULL,
    slo_json        TEXT NOT NULL,
    time_budget_s   REAL,
    gpu_hourly_usd  REAL
);

CREATE TABLE runs (
    id                 INTEGER PRIMARY KEY,
    endpoint_id        INTEGER NOT NULL REFERENCES endpoints(id),
    setup_id           INTEGER NOT NULL REFERENCES setups(id),
    workload_id        INTEGER NOT NULL REFERENCES workloads(id),
    label              TEXT,
    status             TEXT NOT NULL,
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    kneebench_version  TEXT NOT NULL,
    aiperf_version     TEXT NOT NULL,
    vllm_version       TEXT,
    client_json        TEXT NOT NULL,
    network_json       TEXT,
    theory_json        TEXT
);

CREATE TABLE points (
    id             INTEGER PRIMARY KEY,
    run_id         INTEGER NOT NULL REFERENCES runs(id),
    seq            INTEGER NOT NULL,
    phase          TEXT NOT NULL,
    spec_json      TEXT NOT NULL,
    isl            INTEGER NOT NULL,
    osl            INTEGER NOT NULL,
    mode           TEXT NOT NULL,
    concurrency    INTEGER,
    request_rate   REAL,
    status         TEXT NOT NULL,
    stop_reason    TEXT,
    exit_code      INTEGER,
    argv_json      TEXT,
    artifact_dir   TEXT,
    wall_time_s    REAL,
    overhead_s     REAL,
    started_at     TEXT,
    finished_at    TEXT,
    UNIQUE (run_id, seq)
);

CREATE TABLE point_metrics (
    point_id  INTEGER NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    metric    TEXT NOT NULL,
    stat      TEXT NOT NULL,
    value     REAL NOT NULL,
    unit      TEXT,
    PRIMARY KEY (point_id, metric, stat)
);

CREATE TABLE requests (
    point_id  INTEGER NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    idx       INTEGER NOT NULL,
    phase     TEXT NOT NULL,
    start_ns  INTEGER NOT NULL,
    end_ns    INTEGER NOT NULL,
    ttft_ms   REAL,
    tpot_ms   REAL,
    e2e_ms    REAL,
    isl       INTEGER,
    osl       INTEGER,
    error     TEXT,
    PRIMARY KEY (point_id, idx)
);

CREATE TABLE timeslices (
    point_id   INTEGER NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    slice_idx  INTEGER NOT NULL,
    start_ns   INTEGER NOT NULL,
    end_ns     INTEGER NOT NULL,
    metric     TEXT NOT NULL,
    stat       TEXT NOT NULL,
    value      REAL NOT NULL,
    PRIMARY KEY (point_id, slice_idx, metric, stat)
);

CREATE TABLE server_metrics (
    point_id     INTEGER NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    metric       TEXT NOT NULL,
    labels_json  TEXT NOT NULL,
    kind         TEXT NOT NULL,
    stat         TEXT NOT NULL,
    value        REAL NOT NULL
);
CREATE INDEX server_metrics_point ON server_metrics(point_id, metric);

CREATE TABLE artifacts (
    point_id   INTEGER NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    path       TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    size       INTEGER NOT NULL,
    content_z  BLOB,
    PRIMARY KEY (point_id, name)
);

CREATE TABLE warnings (
    point_id  INTEGER NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    source    TEXT NOT NULL,
    code      TEXT NOT NULL,
    message   TEXT NOT NULL
);

-- A cache: always regenerable from the tables above.
CREATE TABLE findings (
    run_id         INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    rule_id        TEXT NOT NULL,
    status         TEXT NOT NULL,
    severity       INTEGER NOT NULL,
    data_json      TEXT NOT NULL,
    rules_version  TEXT NOT NULL,
    PRIMARY KEY (run_id, rule_id)
);
