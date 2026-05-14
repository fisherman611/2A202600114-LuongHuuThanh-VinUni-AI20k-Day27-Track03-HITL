#!/bin/bash
set -e

# Start Postgres (audit + checkpointer store) and wait until it is ready.

echo "Starting Postgres via docker compose..."
docker compose up -d postgres

echo "Waiting for Postgres to accept connections..."
until docker compose exec -T postgres pg_isready -U hitl -d hitl_audit >/dev/null 2>&1; do
  sleep 1
done

echo ""
echo "Postgres is ready:    postgresql://hitl:hitl@localhost:1505/hitl_audit"
echo ""
echo "Next steps:"
echo "  uv sync                                              # install Python deps"
echo "  cp .env.example .env && \$EDITOR .env                 # set OPENROUTER_API_KEY + GITHUB_TOKEN"
echo "  uv run python exercises/exercise_1_confidence.py \\"
echo "        --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1     # start here (exercise 1)"
echo ""
echo "Stop with: docker compose down"
