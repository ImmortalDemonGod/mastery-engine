#!/usr/bin/env bash
#
# fetch_sources.sh - Download third-party source materials for curriculum generation
#
# This script fetches external educational resources that serve as raw material
# for the automated curriculum generation pipeline. These sources are NOT required
# to USE the Mastery Engine - only to REGENERATE curricula from scratch.
#
# Usage:
#   ./scripts/fetch_sources.sh
#
# What it does:
#   - Creates .sources/ directory
#   - Clones 30 Days of Python (for python_for_cp curriculum)
#   - Downloads CP accelerator taxonomies (for cp_accelerator curriculum)
#
# Requirements:
#   - git, curl

set -euo pipefail

SOURCES_DIR=".sources"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🔽 Fetching third-party curriculum source materials..."
echo ""

# Create sources directory
mkdir -p "$SOURCES_DIR"

# Clone 30 Days of Python
if [ ! -d "$SOURCES_DIR/30_days_of_python" ]; then
    echo "📚 Cloning 30 Days of Python..."
    git clone --depth=1 https://github.com/Asabeneh/30-Days-Of-Python.git "$SOURCES_DIR/30_days_of_python"
    # Remove .git to prevent nested git repo issues
    rm -rf "$SOURCES_DIR/30_days_of_python/.git"
    echo "✅ 30 Days of Python downloaded"
else
    echo "⏭️  30 Days of Python already exists"
fi

# Create CP accelerator directory
mkdir -p "$SOURCES_DIR/cp_accelerator"

# Download DSA taxonomies (if you have a specific source, add it here)
# For now, this is a placeholder
if [ ! -f "$SOURCES_DIR/cp_accelerator/dsa_taxonomies" ]; then
    echo "📊 CP accelerator taxonomies placeholder created"
    echo "# DSA Taxonomies" > "$SOURCES_DIR/cp_accelerator/dsa_taxonomies"
    echo "# Add your competitive programming taxonomy sources here" >> "$SOURCES_DIR/cp_accelerator/dsa_taxonomies"
else
    echo "⏭️  CP accelerator taxonomies already exist"
fi

echo ""
echo "✅ Source materials fetched successfully!"
echo ""
echo "Directory structure:"
tree -L 2 "$SOURCES_DIR" 2>/dev/null || ls -lhR "$SOURCES_DIR"
echo ""
echo "🎯 Next steps:"
echo "   - Use scripts/generate_module.py to create new curriculum modules"
echo "   - See docs/architecture/MASTERY_ENGINE.md for pipeline details"
