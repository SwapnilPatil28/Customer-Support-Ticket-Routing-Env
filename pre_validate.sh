#!/usr/bin/env bash
set -euo pipefail

# Pre-submission checklist runner. Prints a short PASS/FAIL summary.

echo "Starting Pre-Validation..."

fail=0
pass_msg() { printf "  \033[0;32m✓\033[0m %s\n" "$1"; }
fail_msg() { printf "  \033[0;31m✗\033[0m %s\n" "$1"; fail=1; }

echo "[1/5] Checking OpenEnv files..."
[ -f "openenv.yaml" ] && pass_msg "openenv.yaml found" || fail_msg "openenv.yaml missing"

echo "[2/5] Validating OpenEnv Spec..."
if openenv validate; then
  pass_msg "openenv validate passed"
else
  fail_msg "openenv validate failed"
fi

echo "[3/5] Checking inference + training scripts..."
[ -f "inference.py" ] && pass_msg "inference.py found" || fail_msg "inference.py missing"
[ -f "train_trl.py" ] && pass_msg "train_trl.py found" || fail_msg "train_trl.py missing"

echo "[4/5] Checking domain modules..."
[ -d "server/domain" ] && pass_msg "server/domain package present" || fail_msg "server/domain missing"

echo "[5/5] Running unit tests (domain-only)..."
if python -m pytest tests/test_reward.py tests/test_incidents.py -q 2>/dev/null; then
  pass_msg "pytest (domain suite) passed"
else
  fail_msg "pytest (domain suite) failed"
fi

if [ "$fail" -eq 0 ]; then
  printf "\n\033[0;32m========================================\n"
  printf "  Ready for Submission!\n"
  printf "========================================\033[0m\n"
  exit 0
else
  printf "\n\033[0;31mPre-validation failed. Fix the issues above before submitting.\033[0m\n"
  exit 1
fi
