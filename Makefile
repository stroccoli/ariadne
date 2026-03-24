APP     ?= ariadne-api
FLYCTL  := $(shell command -v flyctl 2>/dev/null || command -v fly 2>/dev/null)

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
	$(FLYCTL) scale count 1 -a $(APP)

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
