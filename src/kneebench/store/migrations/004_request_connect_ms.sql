-- Time a request spent opening a new connection (DNS, TCP, TLS), so first-token times can
-- be judged without it.
ALTER TABLE requests ADD COLUMN connect_ms REAL;
