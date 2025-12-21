#!/bin/bash
# Wrapper script to run run_uitars.py for each domain in a single tmux session with multiple windows

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# =========================================
# CONFIGURE DOMAINS TO RUN
# =========================================
# Available domains: chrome, gimp, libreoffice_calc, libreoffice_impress,
#                    libreoffice_writer, multi_apps, os, thunderbird, vlc, vs_code
# Comment/uncomment the domains you want to run

DOMAINS_TO_RUN=(
    "chrome"
    "gimp"
    "libreoffice_calc"
    "libreoffice_impress"
    "libreoffice_writer"
    "multi_apps"
    "os"
    "thunderbird"
    "vlc"
    "vs_code"
)

# =========================================
# DEFAULT RUN PARAMETERS
# =========================================
RUN_NAME="1p_verify_every_5"
MAX_STEPS=50
VERIFIER="one_pass"
VERIFY_EVERY_N_STEPS=0
TEST_ALL_META_PATH="evaluation_examples/test_all.json"
MAX_PARALLEL=3 # If 0, run all domains in parallel without limit
VLLM=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --verifier)
            VERIFIER="$2"
            shift 2
            ;;
        --verify_every_n_steps)
            VERIFY_EVERY_N_STEPS="$2"
            shift 2
            ;;
        --test_all_meta_path)
            TEST_ALL_META_PATH="$2"
            shift 2
            ;;
        --vllm)
            VLLM=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--run_name NAME] [--max_steps N] [--verifier TYPE] [--verify_every_n_steps N] [--test_all_meta_path PATH] [--vllm]"
            exit 1
            ;;
    esac
done

# Read domains from test_all.json
if [[ ! -f "$TEST_ALL_META_PATH" ]]; then
    echo "Error: Test file not found at $TEST_ALL_META_PATH"
    exit 1
fi

if [[ ${#DOMAINS_TO_RUN[@]} -eq 0 ]]; then
    echo "Error: No domains configured to run"
    exit 1
fi

echo "==========================================="
echo "Starting tmux session"
echo "Run name: $RUN_NAME"
echo "Max steps: $MAX_STEPS"
echo "Verifier: $VERIFIER"
echo "Verify every N steps: $VERIFY_EVERY_N_STEPS"
echo "Test meta path: $TEST_ALL_META_PATH"
echo "Use vLLM: $VLLM"
if [[ $MAX_PARALLEL -gt 0 ]]; then
    echo "Max parallel domains: $MAX_PARALLEL"
else
    echo "Max parallel domains: unlimited"
fi
echo "==========================================="
echo ""

echo "Total domains to run: ${#DOMAINS_TO_RUN[@]}"
echo "Domains: ${DOMAINS_TO_RUN[*]}"
echo ""

# Create a single tmux session with one window per domain
TMUX_NAME="uitars_${RUN_NAME}"

echo "Creating tmux session: $TMUX_NAME"
echo ""

tmux new-session -d -s "$TMUX_NAME" -c "$SCRIPT_DIR"
tmux send-keys -t "$TMUX_NAME" "conda activate osw" C-m
sleep 1

# Create a window for each domain
DOMAIN_INDEX=0
for DOMAIN in "${DOMAINS_TO_RUN[@]}"; do
    if [[ $DOMAIN_INDEX -eq 0 ]]; then
        # Rename the first window
        tmux rename-window -t "$TMUX_NAME:0" "$DOMAIN"
    else
        # Create a new window for subsequent domains
        tmux new-window -t "$TMUX_NAME" -n "$DOMAIN" -c "$SCRIPT_DIR"
        tmux send-keys -t "$TMUX_NAME:$DOMAIN_INDEX" "conda activate osw" C-m
        sleep 0.5
    fi
    DOMAIN_INDEX=$((DOMAIN_INDEX + 1))
done

# If MAX_PARALLEL is set and positive, run domains in batches
if [[ $MAX_PARALLEL -gt 0 ]]; then
    echo "Running domains in batches of $MAX_PARALLEL..."
    echo ""
    
    # Now run domains in batches
    BATCH_NUM=1
    RUNNING_PIDS=()
    
    for ((i=0; i<${#DOMAINS_TO_RUN[@]}; i++)); do
        DOMAIN="${DOMAINS_TO_RUN[$i]}"
        
        # Wait if we've reached max parallel
        while [[ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ]]; do
            echo "  Batch $BATCH_NUM: Waiting for a slot (${#RUNNING_PIDS[@]}/$MAX_PARALLEL running)..."
            sleep 5
            
            # Check which processes are still running
            NEW_PIDS=()
            for pid_info in "${RUNNING_PIDS[@]}"; do
                pid=$(echo "$pid_info" | cut -d: -f1)
                if ps -p "$pid" > /dev/null 2>&1; then
                    NEW_PIDS+=("$pid_info")
                else
                    domain_name=$(echo "$pid_info" | cut -d: -f2)
                    echo "  ✓ Completed: $domain_name"
                fi
            done
            RUNNING_PIDS=("${NEW_PIDS[@]}")
            
            if [[ ${#RUNNING_PIDS[@]} -lt $MAX_PARALLEL ]]; then
                BATCH_NUM=$((BATCH_NUM + 1))
            fi
        done
        
        # Start the next domain
        CMD="python run_uitars.py --run_name \"${RUN_NAME}\" --max_steps ${MAX_STEPS} --verifier ${VERIFIER} --verify_every_n_steps ${VERIFY_EVERY_N_STEPS} --test_all_meta_path ${TEST_ALL_META_PATH} --domain ${DOMAIN}"
        if [[ "$VLLM" == "true" ]]; then
            CMD="${CMD} --vllm"
        fi
        
        echo "  Starting ($((i+1))/${#DOMAINS_TO_RUN[@]}): $DOMAIN"
        tmux send-keys -t "$TMUX_NAME:$i" "$CMD" C-m
        
        # Get the PID of the python process
        sleep 2
        PID=$(tmux list-panes -t "$TMUX_NAME:$i" -F "#{pane_pid}" | xargs pgrep -P 2>/dev/null | tail -1)
        if [[ -n "$PID" ]]; then
            RUNNING_PIDS+=("$PID:$DOMAIN")
        fi
    done
    
    # Wait for all remaining processes to complete
    echo ""
    echo "Waiting for remaining domains to complete..."
    for pid_info in "${RUNNING_PIDS[@]}"; do
        pid=$(echo "$pid_info" | cut -d: -f1)
        domain_name=$(echo "$pid_info" | cut -d: -f2)
        wait "$pid" 2>/dev/null || true
        echo "  ✓ Completed: $domain_name"
    done
    
    echo ""
    echo "All domains completed!"   
    echo ""
    
    echo ""
    echo "All domains started!"
fi

# Select the first window
tmux select-window -t "$TMUX_NAME:0"

echo ""
echo "==========================================="
echo "Tmux session started!"
echo "==========================================="
echo ""
echo "Session name: $TMUX_NAME"
echo ""
echo "To attach to the session: tmux attach -t ${TMUX_NAME}"
echo "To switch between windows: Ctrl+b then 'n' (next) or 'p' (previous) or '0-9'"
echo "To list all windows: Ctrl+b then 'w'"
echo "To kill the session: tmux kill-session -t ${TMUX_NAME}"
echo ""
echo "Active session:"
tmux ls | grep "${TMUX_NAME}"
