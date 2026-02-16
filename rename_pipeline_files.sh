#!/bin/bash
# filepath: git_code/rename_pipeline_files.sh

set -e  # Exit on any error

echo "=== Renaming RothC Pipeline Files ==="
echo ""

# 1. Rename files using git mv (preserves history)
echo "Step 1: Renaming files..."
git mv main_calib.py pipeline_main.py
git mv step2_c_inputs.py calc_carbon_inputs.py
git mv step3_c_initial.py calc_initial_pools.py
git mv step4_plant_cover.py calc_plant_cover.py
git mv step5_run_rothc.py run_rothc_model.py
git mv step6_calc_deltas.py calc_soc_deltas.py
git mv rothc.py rothc_model.py

echo "✓ Files renamed"
echo ""

# 2. Update imports in pipeline_main.py
echo "Step 2: Updating imports in pipeline_main.py..."
sed -i 's/import step2_c_inputs as step2/import calc_carbon_inputs as step2/g' pipeline_main.py
sed -i 's/import step3_c_initial as step3/import calc_initial_pools as step3/g' pipeline_main.py
sed -i 's/import step4_plant_cover as step4/import calc_plant_cover as step4/g' pipeline_main.py
sed -i 's/import step5_run_rothc as step5/import calc_soc_deltas as step5/g' pipeline_main.py
sed -i 's/import step6_calc_deltas as step6/import calc_soc_deltas as step6/g' pipeline_main.py

# Also update any direct imports (no alias)
sed -i 's/import step2_c_inputs/import calc_carbon_inputs/g' pipeline_main.py
sed -i 's/import step3_c_initial/import calc_initial_pools/g' pipeline_main.py
sed -i 's/import step4_plant_cover/import calc_plant_cover/g' pipeline_main.py
sed -i 's/import step5_run_rothc/import run_rothc_model/g' pipeline_main.py
sed -i 's/import step6_calc_deltas/import calc_soc_deltas/g' pipeline_main.py

echo "✓ pipeline_main.py imports updated"
echo ""

# 3. Update rothc imports in all Python files
echo "Step 3: Updating rothc imports in all files..."
find . -name "*.py" -type f -exec sed -i 's/from rothc import/from rothc_model import/g' {} +
find . -name "*.py" -type f -exec sed -i 's/import rothc$/import rothc_model as rothc/g' {} +

echo "✓ rothc imports updated"
echo ""

# 4. Update step module imports in other files
echo "Step 4: Updating cross-references in other modules..."

# In run_rothc_model.py (was step5)
sed -i 's/from rothc import/from rothc_model import/g' run_rothc_model.py
sed -i 's/import rothc$/import rothc_model as rothc/g' run_rothc_model.py

# In optimization.py
sed -i 's/import step2_c_inputs as step2/import calc_carbon_inputs as step2/g' optimization.py
sed -i 's/import step3_c_initial as step3/import calc_initial_pools as step3/g' optimization.py
sed -i 's/import step4_plant_cover as step4/import calc_plant_cover as step4/g' optimization.py
sed -i 's/import step5_run_rothc as step5/import run_rothc_model as step5/g' optimization.py
sed -i 's/import step6_calc_deltas as step6/import calc_soc_deltas as step6/g' optimization.py

sed -i 's/from step2_c_inputs import/from calc_carbon_inputs import/g' optimization.py
sed -i 's/from step3_c_initial import/from calc_initial_pools import/g' optimization.py
sed -i 's/from step4_plant_cover import/from calc_plant_cover import/g' optimization.py
sed -i 's/from step5_run_rothc import/from run_rothc_model import/g' optimization.py
sed -i 's/from step6_calc_deltas import/from calc_soc_deltas import/g' optimization.py

echo "✓ Cross-references updated"
echo ""

# 5. Update any test files if they exist
if ls test_*.py 1> /dev/null 2>&1; then
    echo "Step 5: Updating test files..."
    find . -name "test_*.py" -type f -exec sed -i 's/import step2_c_inputs/import calc_carbon_inputs/g' {} +
    find . -name "test_*.py" -type f -exec sed -i 's/import step3_c_initial/import calc_initial_pools/g' {} +
    find . -name "test_*.py" -type f -exec sed -i 's/import step4_plant_cover/import calc_plant_cover/g' {} +
    find . -name "test_*.py" -type f -exec sed -i 's/import step5_run_rothc/import run_rothc_model/g' {} +
    find . -name "test_*.py" -type f -exec sed -i 's/import step6_calc_deltas/import calc_soc_deltas/g' {} +
    find . -name "test_*.py" -type f -exec sed -i 's/from rothc import/from rothc_model import/g' {} +
    echo "✓ Test files updated"
else
    echo "Step 5: No test files found (skipped)"
fi
echo ""

# 6. Verify no old imports remain
echo "Step 6: Verification - checking for any remaining old imports..."
OLD_PATTERNS=(
    "import step2_c_inputs"
    "import step3_c_initial"
    "import step4_plant_cover"
    "import step5_run_rothc"
    "import step6_calc_deltas"
    "from step2_c_inputs"
    "from step3_c_initial"
    "from step4_plant_cover"
    "from step5_run_rothc"
    "from step6_calc_deltas"
    "from rothc import"
    "import rothc$"
)

FOUND_OLD=0
for pattern in "${OLD_PATTERNS[@]}"; do
    if grep -r "$pattern" *.py 2>/dev/null; then
        echo "⚠ WARNING: Found old import pattern: $pattern"
        FOUND_OLD=1
    fi
done

if [ $FOUND_OLD -eq 0 ]; then
    echo "✓ No old import patterns found"
fi
echo ""

# 7. Git status and commit suggestion
echo "Step 7: Git status..."
git status --short
echo ""
echo "=== Renaming Complete ==="
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff --cached"
echo "  2. Test the pipeline: python pipeline_main.py"
echo "  3. Commit changes: git commit -m 'Refactor: Rename pipeline files to verb-noun pattern'"
echo ""
echo "To undo: git reset --hard HEAD"
