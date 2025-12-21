#!/bin/bash
#==============================================================================
# Auxiliary functions
#==============================================================================
# Create flags with `true` `false` parameter
get_flag() {
    local flag=$1
    local condition=$2
    if [ "$condition" = true ]; then
        echo "--$flag"
    else
        echo ""
    fi
}


#===============================================================================
# Set environment variables
#===============================================================================
export TF_CPP_MIN_LOG_LEVEL=2   # Suppress some tensorflow warnings.

#===============================================================================
# Parametrization for end to end eval
#===============================================================================

#-------------------------------------------------------------------------------
# Tasks
#-------------------------------------------------------------------------------
env="vwa"                                                   # Benchmark to run the evaluation. Options: vwa (visualwebarena), wa (webarena)
json_config_file="config_files/vwa/test_vwa.raw.json"       # Path to the json file with test config for each task.
test_start_idx=-1                                           # Start ID of tasks to evaluate. If lower than initial ID, default to first task.
test_end_idx=1000                                           # End ID of tasks to evaluate. If greater than the number of tasks, will default to the last task. 
task_list=''                                                # Path to a .txt file containing a list of task IDs to evaluate. Overwrites `test_start_idx` and `test_end_idx`.
max_tasks=-1                                                # Max number of tasks to evaluate. If <0, or none,  evaluate all tasks in list or range.
shuffle_tasks=$(get_flag 'shuffle_tasks' true)             # If true, shuffle the order of tasks before evaluation.

#-------------------------------------------------------------------------------
# Models and Agent Configuration
#-------------------------------------------------------------------------------
agent_config_file='noverifier.yaml'  # Path to the YAML configuration for the Agent.
manual_input=$(get_flag 'manual_input' true) # If true, actions are inputted manually by the user.

# Device to hold captioner models, if any. server-cuda:<id>: hosts a server on GPU with <id>;  cuda:<id>: loads the model on the GPU with <id>; cpu: loads the model on the CPU.
agent_captioning_model_device='server-cuda:0' # Augment text observations and task intent with captions for images in a webpage or user intent.
eval_captioning_model_device='server-cuda:0'  # Used by the oracle for evaluation.

#-------------------------------------------------------------------------------
# Evaluation
#-------------------------------------------------------------------------------
max_steps=30                                   # Default:30. Max number of environment steps allowed. If exceeded, FAIL.
parsing_failure_th=3                           # Default: 3. Number of action parsing failures allowed. If exceeded, FAIL.
repeating_action_failure_th=30                 # Default: 5. Max number of repeated actions allowed. If exceeded, FAIL.
fuzzy_match_model='gemini-2.5-flash'           # Default: gpt-5-2025-08-07. Defines the provider for fuzzy match evals

#-------------------------------------------------------------------------------
# Observation and action config
#-------------------------------------------------------------------------------
observation_type=image_som                     # Environment observation type. Options: accessibility_tree, accessibility_tree_with_captioner, image_som
viewport_width=1280                            # Width of the browser viewport. Default: 1280
viewport_height=2048                           # Height of the browser viewport. Default: 720 for small context window models | 2048 for large context window models
current_viewport_only=$(get_flag 'current_viewport_only' true)   # Default: true
show_scroll_bar=$(get_flag 'show_scroll_bar' true)  # Renders the browser's scroll bar in screenshots.

#-------------------------------------------------------------------------------
# Execution config
#-------------------------------------------------------------------------------
sleep_after_execution=0.5                                      # Seconds. If > 0, automatic wait for the page to stabilize then sleep for up to this number of seconds.
docker_instance_id='90'                                       # Instance ID for homepage websites and env containers.
skip_env_reset=$(get_flag 'skip_env_reset' true)             # If true, skip resetting the environment at the beginning of each task. Otherwise, reset environments required by the corresponding task config.
force_reset=$(get_flag 'force_reset' false)                   # If true, force reset of all websites before execution of each task. Otherwise, only reset environments required by the corresponding task config.
skip_cookie_reset=$(get_flag 'skip_cookie_reset' false)       # If true, skip renewing cookies at the beginning of each task.
render=$(get_flag 'render' false)                             # Displays the browser.
[ "$render" = '--render' ] && slow_mo=100 || slow_mo=0        # Display the browser in slow motion. 100 if displaying the browser, else 0.
require_all_sites_up=1                                        # If 1 (default), require all sites to be up before executing a task. If 0, only require the sites included in the task config file.

#-------------------------------------------------------------------------------
# Logging config
#-------------------------------------------------------------------------------
result_dir=''  # Save execution data to this directory; if empty, automatically creates a directory with the current date and time.
log_trajectory=$(get_flag 'log_trajectory' true)              # Save trajectories in a json file with for each task.
log_html=$(get_flag 'log_html' true)                         # Save trajectories in HTML files for each task (VWA original format).
render_screenshot=$(get_flag 'render_screenshot' true)        # Add screenshots to HTML files with trajectories.
save_trace_enabled=$(get_flag 'save_trace_enabled' false)     # Save playwright traces in the result directory.


#===============================================================================
# Command line arguments - overwrite parameters if provided in the terminal
#===============================================================================

while getopts ":s:e:t:c:r:k:d:m:a:i:f:x:" opt; do
    case $opt in
        s) test_start_idx=$OPTARG ;;                        # start index of tasks to evaluate
        e) test_end_idx=$OPTARG ;;                          # end index of tasks to evaluate
        t) task_list=$OPTARG ;;                             # path to a .txt file containing a list of task IDs to evaluate
        c) json_config_file=$OPTARG ;;                      # path to the json file with test config for each task
        d) result_dir=$OPTARG ;;                            # directory to save execution results
        m) agent_captioning_model_device=$OPTARG ;;         # device to hold captioner models
        a) agent_config_file=$OPTARG ;;                     # path to the YAML configuration for the Agent
        i) docker_instance_id=$OPTARG ;;                    # docker instance ID
        f) force_reset=$(get_flag 'force_reset' true) ;;    # if true, force reset of all websites before execution of each task
        x) max_steps=$OPTARG ;;                             # max number of environment steps allowed; if exceeded, FAIL
        \?) echo "Invalid option -$OPTARG" >&2 ;;
    esac
done
#===============================================================================
# Evaluation
#===============================================================================
echo -e "\n==================================\nStart of evaluation\n=================================="
python3 run.py \
    --env $env \
    --agent_config_file $agent_config_file \
    --result_dir $result_dir \
    --test_start_idx $test_start_idx \
    --test_end_idx $test_end_idx \
    --test_config_json_file $json_config_file \
    --task_list $task_list \
    --viewport_width $viewport_width \
    --viewport_height $viewport_height \
    --max_steps $max_steps \
    --parsing_failure_th $parsing_failure_th \
    --repeating_action_failure_th $repeating_action_failure_th \
    --observation_type $observation_type \
    --agent_captioning_model_device $agent_captioning_model_device \
    --eval_captioning_model_device $eval_captioning_model_device \
    --max_tasks $max_tasks \
    --sleep_after_execution $sleep_after_execution \
    --docker_instance_id $docker_instance_id \
    --fuzzy_match_model $fuzzy_match_model \
    --slow_mo $slow_mo \
    $skip_env_reset \
    $skip_cookie_reset \
    $force_reset \
    $render \
    $render_screenshot \
    $save_trace_enabled \
    $current_viewport_only \
    $show_scroll_bar \
    $shuffle_tasks \
    $log_trajectory \
    $log_html \
    $manual_input \



