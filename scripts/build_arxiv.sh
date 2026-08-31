#!/usr/bin/env bash
# Build and verify the arXiv submission tarball.
#
# arXiv compiles the source itself, so the tarball must be self-contained and
# must build in a directory that has none of our local state. This script
# therefore copies only what is needed into a scratch dir, compiles there
# twice from scratch, and fails loudly on anything arXiv would reject.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-arxiv_submission.tar.gz}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== staging"
mkdir -p "$TMP/pkg/figs" "$TMP/pkg/tables"
cp paper/main.tex "$TMP/pkg/"
cp paper/figs/*.pdf "$TMP/pkg/figs/"
cp paper/tables/*.tex "$TMP/pkg/tables/"

echo "== preflight checks"
fail=0
head -5 "$TMP/pkg/main.tex" | grep -q '\\pdfoutput=1' \
  || { echo "  FAIL: \\pdfoutput=1 must appear in the first five lines"; fail=1; }
if grep -q '\\todo{' "$TMP/pkg/main.tex"; then
  echo "  WARN: $(grep -c '\\todo{' "$TMP/pkg/main.tex") unresolved \\todo markers will render in the PDF:"
  grep -n '\\todo{' "$TMP/pkg/main.tex" | sed 's/^/    /' | cut -c1-110
fi
grep -qE '\\(input|include|includegraphics)\{[^}]*\.\./' "$TMP/pkg/main.tex" \
  && { echo "  FAIL: path escapes the package directory"; fail=1; }
grep -q 'write18\|shell-escape' "$TMP/pkg/main.tex" \
  && { echo "  FAIL: shell-escape is not allowed"; fail=1; }

echo "== clean compile (twice, in a scratch dir)"
( cd "$TMP/pkg" && for i in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error main.tex > pass$i.log 2>&1 \
      || { echo "  FAIL: pass $i"; tail -25 pass$i.log; exit 1; }
  done )

pages=$(pdfinfo "$TMP/pkg/main.pdf" | awk '/^Pages/{print $2}')
undef=$(grep -c 'undefined' "$TMP/pkg/pass2.log" || true)
echo "  built $pages pages, $undef undefined-reference warnings"
[ "$undef" -gt 0 ] && grep -m5 'undefined' "$TMP/pkg/pass2.log" | sed 's/^/    /'

rm -f "$TMP/pkg"/*.log "$TMP/pkg"/*.aux "$TMP/pkg"/*.out "$TMP/pkg"/main.pdf
tar czf "$OUT" -C "$TMP/pkg" .
echo "== wrote $OUT ($(du -h "$OUT" | cut -f1))"
tar tzf "$OUT" | sed 's/^/    /'
[ "$fail" -eq 0 ] || { echo "PREFLIGHT FAILED"; exit 1; }
echo "OK"
