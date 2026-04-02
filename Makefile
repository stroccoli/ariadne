APP     ?= ariadne-api
FLYCTL  := $(shell command -v flyctl 2>/dev/null || command -v fly 2>/dev/null)

# ── Offline pipelines ──────────────────────────────────────────────────────────
#
# These targets install only their required extras (not shipped to production)
# and run the corresponding offline script.
#
#   make eval                           # full evaluation (50 samples)
#   make eval ARGS="--num-samples 5"    # quick smoke-test
#   make ingest                         # full DVC ingestion pipeline
#   make ingest ARGS="chunk index"      # run specific DVC stages

ARGS ?=

.PHONY: eval eval-setup ingest

eval:
	pip install -e ".[evals]" --quiet
	python evals/ragas_eval.py $(ARGS)

# Upload / refresh the LangSmith dataset then run the evaluation.
# Adds --force-refresh so stale or missing reference fields are fixed first.
#   make eval-setup                          # all 50 samples
#   make eval-setup ARGS="--num-samples 5"   # quick smoke-test
eval-setup:
	pip install -e ".[evals]" --quiet
	python evals/ragas_eval.py --force-refresh $(ARGS)

ingest:
	pip install -e ".[ingestion]" --quiet
	dvc repro $(ARGS)

# Index all three embedding providers into separate Qdrant collections.
# Runs collect → preprocess → chunk once, then indexes for each provider.
#   make ingest-all
ingest-all:
	pip install -e ".[ingestion]" --quiet
	dvc repro collect preprocess chunk
	python scripts/stages/index.py --provider openai
	python scripts/stages/index.py --provider ollama
	python scripts/stages/index.py --provider gemini

# Incremental index: skip chunks whose content hasn't changed (by content_hash).
# Saves embedding API quota when the corpus is mostly unchanged.
#   make ingest-incremental
ingest-incremental:
	pip install -e ".[ingestion]" --quiet
	dvc repro collect preprocess chunk
	python scripts/stages/index.py --provider openai --incremental
	python scripts/stages/index.py --provider ollama --incremental
	python scripts/stages/index.py --provider gemini --incremental

# Run evaluations for all three providers, producing separate LangSmith
# experiments (e.g. ragas_openai, ragas_ollama, ragas_gemini).
#   make eval-all
#   make eval-all ARGS="--num-samples 5"
eval-all:
	pip install -e ".[evals]" --quiet
	python evals/ragas_eval.py --provider openai $(ARGS)
	python evals/ragas_eval.py --provider ollama $(ARGS)
	python evals/ragas_eval.py --provider gemini $(ARGS)

# ── Fly.io ─────────────────────────────────────────────────────────────────────
#
# Reads production-relevant variables from your local .env and pushes them
# to Fly.io as encrypted secrets. Run once before the first deploy, and
# again whenever a secret changes.
#
#   make fly-launch            # create the app on Fly.io (first time only)
#   make fly-secrets           # push secrets from .env to Fly.io
#   make fly-deploy            # deploy the current code
#   make fly-status            # show machine status
#   make fly-logs              # tail live logs

.PHONY: fly-launch fly-secrets fly-deploy fly-status fly-logs

_check-flyctl:
	@[ -n "$(FLYCTL)" ] || { \
	  echo ""; \
	  echo "Error: flyctl is not installed."; \
	  echo "Install it with:  curl -L https://fly.io/install.sh | sh"; \
	  echo "Then add to PATH: export PATH=\"\$$HOME/.fly/bin:\$$PATH\""; \
	  echo ""; \
	  exit 1; \
	}

# First-time setup: creates the app on Fly.io using our existing config.
# Run this ONCE, then use fly-secrets + fly-deploy for all subsequent deploys.
fly-launch: _check-flyctl
	$(FLYCTL) launch \
	  --no-deploy \
	  --config infra/fly.toml \
	  --name $(APP) \
	  --region iad \
	  --copy-config

fly-secrets: _check-flyctl
	@[ -f .env.production ] || { echo "Error: .env.production not found at project root"; exit 1; }
	@set -a && . ./.env.production && set +a && \
	$(FLYCTL) secrets set \
	  LLM_PROVIDER="$${LLM_PROVIDER}" \
	  GEMINI_API_KEY="$${GEMINI_API_KEY}" \
	  GEMINI_MODEL="$${GEMINI_MODEL:-gemini-2.5-flash}" \
	  EMBEDDING_PROVIDER="$${EMBEDDING_PROVIDER}" \
	  VECTOR_STORE="$${VECTOR_STORE:-qdrant}" \
	  QDRANT_URL="$${QDRANT_URL}" \
	  QDRANT_API_KEY="$${QDRANT_API_KEY}" \
	  ALLOWED_ORIGINS="$${ALLOWED_ORIGINS:-https://ariadne.vercel.app}" \
	  API_KEY="$${API_KEY}" \
	  SENTRY_DSN="$${SENTRY_DSN}" \
	  LOG_FORMAT="json" \
	  -a $(APP)

fly-deploy: _check-flyctl
	@[ -f .env.production ] || { echo "Error: .env.production not found at project root"; exit 1; }
	@set -a && . ./.env.production && set +a && \
	$(FLYCTL) deploy \
	  --config infra/fly.toml \
	  --remote-only \
	  --build-arg NEXT_PUBLIC_API_KEY="$${API_KEY}" \
	  -a $(APP)

fly-status: _check-flyctl
	$(FLYCTL) status -a $(APP)

fly-logs: _check-flyctl
	$(FLYCTL) logs -a $(APP)
