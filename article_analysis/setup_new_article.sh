#!/bin/bash
# Template for setting up new article analysis framework
# 
# Usage: bash setup_new_article.sh NewArticleName
# 
# This creates a new article analysis folder based on the KZ_article template

set -e

if [ $# -eq 0 ]; then
    echo "Usage: bash setup_new_article.sh <ArticleName>"
    echo ""
    echo "Examples:"
    echo "  bash setup_new_article.sh BEC_formation"
    echo "  bash setup_new_article.sh Soliton_dynamics"
    echo ""
    exit 1
fi

ARTICLE_NAME=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTICLE_DIR="$SCRIPT_DIR/$ARTICLE_NAME"

# Check if article already exists
if [ -d "$ARTICLE_DIR" ]; then
    echo "ERROR: Article directory already exists: $ARTICLE_DIR"
    exit 1
fi

echo "Creating new article analysis framework: $ARTICLE_NAME"
echo "Location: $ARTICLE_DIR"
echo ""

# Create directory structure
mkdir -p "$ARTICLE_DIR"
mkdir -p "$ARTICLE_DIR/results"

# Copy template files
echo "Copying template files..."
cp KZ_article/config.yaml "$ARTICLE_DIR/"
cp KZ_article/camera_settings.yaml "$ARTICLE_DIR/"
cp KZ_article/defect_config.yaml "$ARTICLE_DIR/"
cp KZ_article/run_kz_article_analysis.py "$ARTICLE_DIR/run_article_analysis.py"
cp KZ_article/batch_kz_article_analysis.py "$ARTICLE_DIR/batch_article_analysis.py"
cp KZ_article/README.md "$ARTICLE_DIR/"
cp KZ_article/QUICKSTART.md "$ARTICLE_DIR/"

# Update script names in comments
sed -i "s/KZ Article/$(echo $ARTICLE_NAME | tr '_' ' ') Article/g" "$ARTICLE_DIR/run_article_analysis.py"
sed -i "s/KZ Article/$(echo $ARTICLE_NAME | tr '_' ' ') Article/g" "$ARTICLE_DIR/batch_article_analysis.py"

# Make scripts executable
chmod +x "$ARTICLE_DIR/run_article_analysis.py"
chmod +x "$ARTICLE_DIR/batch_article_analysis.py"

echo ""
echo "✓ Article framework created successfully!"
echo ""
echo "Next steps:"
echo "1. Edit $ARTICLE_DIR/config.yaml"
echo "   - Update article_name and article_description"
echo "   - Add your datasets with (year, month, day, sequences)"
echo "   - Adjust initial affine_correction and integration limits"
echo ""
echo "2. Edit $ARTICLE_DIR/camera_settings.yaml if needed"
echo "   - Usually no changes required for standard analysis"
echo ""
echo "3. Edit $ARTICLE_DIR/defect_config.yaml if needed"
echo "   - Adjust domain wall finding parameters for your data"
echo ""
echo "4. Run interactive analysis:"
echo "   cd $ARTICLE_DIR"
echo "   python run_article_analysis.py config.yaml camera_settings.yaml defect_config.yaml"
echo ""
echo "5. Review results in:"
echo "   $ARTICLE_DIR/results/YYYY/MM/DD/iteration_1/"
echo ""
echo "For more details, see: $ARTICLE_DIR/QUICKSTART.md"
