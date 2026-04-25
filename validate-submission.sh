#!/usr/bin/env bash
set -uo pipefail

# Remote validation script executed by judges / CI against a deployed
# Hugging Face Space. It checks that:
#   1. The deployed Space responds to /reset and /healthz.
#   2. The Dockerfile is present in the submitted repo.
#   3. `openenv validate` passes locally on the submitted source tree.

if [ -t 1 ]; then
  RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m' BOLD='\033[1m' NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: ./validate-submission.sh <hf_space_url> [repo_dir]\n"
  exit 1
fi

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; }
fail() { log "${RED}FAILED${NC} -- $1"; }

log "${BOLD}Step 1/4: Pinging HF Space ${NC}($PING_URL/reset) ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30 || printf "000")
if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live"
else
  fail "HF Space /reset returned $HTTP_CODE"
  exit 1
fi

log "${BOLD}Step 2/4: Checking /healthz endpoint...${NC}"
HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$PING_URL/healthz" --max-time 20 || printf "000")
if [ "$HEALTH_CODE" = "200" ]; then
  pass "/healthz is reachable"
else
  fail "/healthz returned $HEALTH_CODE"
fi

log "${BOLD}Step 3/4: Verifying Dockerfile presence${NC} ..."
if [ -f "$REPO_DIR/server/Dockerfile" ] || [ -f "$REPO_DIR/Dockerfile" ]; then
  pass "Dockerfile found"
else
  fail "Dockerfile missing"
  exit 1
fi

log "${BOLD}Step 4/4: Running openenv validate${NC} ..."
if (cd "$REPO_DIR" && openenv validate); then
  pass "openenv validate passed"
else
  fail "openenv validate failed"
  exit 1
fi

printf "\n${GREEN}${BOLD}All 4/4 checks passed! Ready to submit.${NC}\n"
