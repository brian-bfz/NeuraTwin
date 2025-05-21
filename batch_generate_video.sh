CASE_NAME="single_push_rope"
start_time="20250521_133000"
for dir in generated_data/${CASE_NAME}_*; do
    base=$(basename "$dir")
    ts_part=${base#${CASE_NAME}_}
    ts_clean=$(echo "$ts_part" | sed 's/[^0-9_]//g' | cut -d_ -f1-2)
    echo "<$ts_clean> <$start_time>"
    if [ "$ts_clean" > "$start_time" ]; then
        python -W ignore v_from_d.py --case_name $CASE_NAME --timestamp "$ts_clean"
    fi
done