-- Applied 2026-04-18T09:15:00Z
ALTER TABLE users
    ALTER COLUMN id TYPE UUID
    USING lpad(to_hex(id), 32, '0')::uuid;

ALTER TABLE sessions
    ALTER COLUMN user_id TYPE UUID
    USING lpad(to_hex(user_id), 32, '0')::uuid;
