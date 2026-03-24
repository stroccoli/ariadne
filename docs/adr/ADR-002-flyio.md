# ADR-002: Fly.io over AWS for backend hosting

## Status: Accepted

## Date: 2026-03-24

---

## Context

Ariadne is a portfolio project running in production with approximately zero traffic outside of demos and evaluation runs. The backend is a containerised FastAPI application (`infra/Dockerfile`) that needs:

- A public HTTPS endpoint with no perceptible cold-start on top of already-present 1–5s LLM latency
- Health check support (`GET /health`)
- Ability to run for months at effectively zero cost
- Docker-first deployment — the container is already written

The application also serves a pre-built Next.js static export from the same container (multi-stage Docker build), so the hosting platform runs a single container serving both API and UI.

Budget constraint: $0/month. Non-negotiable for a public portfolio project with no paying users.

---

## Decision

Deploy on **Fly.io** using a `shared-cpu-1x` machine (1 shared vCPU, 512 MB RAM) in the `iad` (US East) region.

Key configuration in `infra/fly.toml`:
- `auto_stop_machines = 'stop'` + `min_machines_running = 0`: the machine stops when idle. Effective cost: $0/month.
- `force_https = true`: HTTPS enforced at the edge with no additional configuration.
- Health check: `GET /health` every 30s, 5s timeout, 10s grace period.
- Concurrency: `soft_limit = 20`, `hard_limit = 25`.

The Docker image is built by Fly.io's remote builder from `infra/Dockerfile`. `fly deploy` is the entire deployment command.

---

## Alternatives considered

### AWS Lambda

Lambda's free tier is generous (1M requests/month, 400,000 GB-seconds compute).

**Why rejected:**
- Packaging a FastAPI app for Lambda requires `mangum` (ASGI adapter) — added complexity with no benefit.
- The LangGraph pipeline holds an `lru_cache`'d LLM client. Lambda's frozen execution model makes cache behaviour unpredictable across cold starts.
- Python 3.11 container image cold starts add 1–3s on top of LLM latency — visible in demos.
- No native Docker-first workflow.

### AWS App Runner

App Runner runs containers, handles HTTPS, auto-scales.

**Why rejected:** no free tier. Minimum cost ~$5/month for the smallest instance. Breaks the budget constraint.

### EC2 t2.micro (always-on)

AWS Free Tier provides 750 hours/month of `t2.micro` for the first 12 months — then ~$8/month.

**Why rejected:** 12-month time limit, plus manual HTTPS setup, Docker daemon management, and SSH deployments. Not worth the operational overhead for a demo project.

### Render

Render's free tier spins down after 15 minutes of inactivity with a 30–90 second cold start.

**Why rejected:** 30–90s cold start makes the demo experience unusable — the machine wakes up while the user is waiting to see an LLM response, producing a multi-minute first visit.

### Fly.io ✓

Fly.io's free tier includes three `shared-cpu-1x` machines. With `min_machines_running = 0` and `auto_stop_machines = 'stop'`, the machine costs nothing while idle. On first request the machine **resumes** (not cold-boots) in ~2s — the process restores from its last state. This 2s overhead is imperceptible alongside 3–6s LLM call latency.

`fly deploy` builds and deploys the existing `infra/Dockerfile` with no additional packaging. HTTPS is automatic. Health checks are first-class.

---

## Consequences

**Positive:**
- Effective monthly cost: $0 for portfolio use.
- HTTPS, health checks, and deployment configuration are declarative in `fly.toml`.
- Docker-native workflow — same `infra/Dockerfile` runs locally and in production.
- Machine suspend/resume (~2s) is imperceptible vs. LLM API call latency.
- Region `iad` (US East) is low-latency to Qdrant Cloud and Google AI services.

**Negative:**
- Not AWS — engineers unfamiliar with Fly.io face a small learning curve for `flyctl` and `fly.toml`.
- No native VPC/IAM — secrets managed with `fly secrets set`, not AWS Secrets Manager.
- Single machine, single region. No horizontal scaling or multi-region failover.
- Free tier terms can change.

---

## When would we revisit this?

- If the project needs real user traffic (hundreds of requests/day): upgrade to `min_machines_running = 1` (~$1.94/month) or add a second region.
- If the project needs to integrate with AWS services (RDS, S3, SQS): consider AWS App Runner or ECS Fargate where VPC peering is native.
- If Fly.io free tier policies change materially.
