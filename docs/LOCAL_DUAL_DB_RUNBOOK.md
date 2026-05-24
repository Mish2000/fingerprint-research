# Local Dual-Database Runbook

This project can run the identification store in a split PostgreSQL layout:

- biometric database: fingerprint templates, feature vectors, retrieval vectors
- identity database: national-ID mapping / identity metadata

The values below are local-development placeholders only. For any shared, production, or demo machine, set private values through your shell or a local `.env` file that is not committed.

## Start local databases

```powershell
$env:BIOMETRIC_POSTGRES_PASSWORD = "change_me_biometric_dev_password"
$env:IDENTITY_POSTGRES_PASSWORD = "change_me_identity_dev_password"

docker compose -f apps/api/docker-compose.yml up -d biometric_db identity_db
```

## Point integration tests at the local databases

```powershell
$env:IDENTIFICATION_TEST_BIOMETRIC_DATABASE_URL = "postgresql://admin:$env:BIOMETRIC_POSTGRES_PASSWORD@127.0.0.1:5432/biometric_db"
$env:IDENTIFICATION_TEST_IDENTITY_DATABASE_URL = "postgresql://admin:$env:IDENTITY_POSTGRES_PASSWORD@127.0.0.1:5433/identity_db"
```

## Run the PostgreSQL integration tests

```powershell
python -m pytest tests/test_secure_split_store.py tests/test_secure_split_store_migration_postgres.py -q
```

## Runtime configuration

For application runtime, prefer environment variables instead of hardcoded URLs:

```powershell
$env:DATABASE_URL = "postgresql://admin:$env:BIOMETRIC_POSTGRES_PASSWORD@127.0.0.1:5432/biometric_db"
$env:IDENTITY_DATABASE_URL = "postgresql://admin:$env:IDENTITY_POSTGRES_PASSWORD@127.0.0.1:5433/identity_db"
```

If `IDENTITY_DATABASE_URL` is omitted, the store falls back to the biometric database URL and behaves as a single-database deployment.

The application fallback URL intentionally has no embedded credentials. If local PostgreSQL requires authentication,
`/identify/stats` will keep the existing service-unavailable error path and include the missing env var in the
message, while `/identify/admin/layout` will still return a readiness inspection payload with a
`database_connection_failed` issue.
