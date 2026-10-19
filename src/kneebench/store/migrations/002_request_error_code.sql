-- HTTP status of failed requests, so gateway limits (429, 5xx) can be told apart.
ALTER TABLE requests ADD COLUMN error_code INTEGER;
