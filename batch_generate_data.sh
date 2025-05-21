CASE_NAME="single_push_rope"
# export "WARP_SILENCE_WARNINGS=1"
# export "WARP_LOG_LEVEL=ERROR"
exec 1>/dev/null # If I see a single WARP warning again I will kill myself

for i in $(seq 1 1); do
    echo "$CASE_NAME Run $i"
    python -W ignore generate_data.py --case_name $CASE_NAME
done

