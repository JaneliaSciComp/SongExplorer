#!/bin/bash

# launch the graphical user interface
 
# gui.sh <configuration-file> <port>
# http://<hostname>:<port>/gui

# e.g.
# $SONGEXPLORER_BIN gui.sh `pwd`/configuration.pysh 5006

configuration_file=$1
port=$2

source $configuration_file

local_ncpu_cores=$(nproc)
which nvidia-smi &> /dev/null
if [[ "$?" == 0 ]] ; then
  nvidia_output=$(nvidia-smi -L)
  local_ngpu_cards=$(echo "$nvidia_output" | wc -l)
else
  local_ngpu_cards=0
fi
nbytes=$(free -b | tail -2 | head -1 | tr -s ' ' | cut -d' ' -f2)
local_ngigabytes_memory=$(dc -e "$nbytes 1024 / 1024 / 512 + 1024 / p")
echo INFO: detected $local_ncpu_cores local_ncpu_cores, \
                    $local_ngpu_cards local_ngpu_cards, \
                    $local_ngigabytes_memory local_ngigabytes_memory

if [[ -n "$server_ipaddr" ]] ; then
    server_ncpu_cores=$(ssh $server_ipaddr nproc)
    ssh $server_ipaddr which nvidia-smi &> /dev/null
    if [[ "$?" == 0 ]] ; then
      nvidia_output=$(ssh $server_ipaddr nvidia-smi -L)
      server_ngpu_cards=$(echo "$nvidia_output" | wc -l)
    else
      server_ngpu_cards=0
    fi
    nbytes=$(ssh $server_ipaddr free -b | tail -2 | head -1 | tr -s ' ' | cut -d' ' -f2)
    server_ngigabytes_memory=$(dc -e "$nbytes 1024 / 1024 / 512 + 1024 / p")
    echo INFO: detected $server_ncpu_cores server_ncpu_cores, \
                        $server_ngpu_cards server_ngpu_cards, \
                        $server_ngigabytes_memory server_ngigabytes_memory
fi

isinteger() {
  number_re='^-?[0-9]+$'
  [[ "${!1}" =~ $number_re ]] || echo WARNING: $1 is not set or is not an integer
}

isinteger audio_tic_rate
isinteger audio_nchannels
isinteger gui_snippets_width_ms
isinteger gui_snippets_nx
isinteger gui_snippets_ny
isinteger gui_nlabels
isinteger gui_gui_width_pix
isinteger gui_context_width_ms
isinteger gui_context_offset_ms
isinteger gui_context_waveform_height_pix
isinteger gui_context_spectrogram_height_pix
isinteger models_per_job
isinteger pca_batch_size
isinteger nprobabilities
isinteger nwindows
isinteger detect_ncpu_cores
isinteger detect_ngpu_cards
isinteger detect_ngigabytes_memory
isinteger misses_ncpu_cores
isinteger misses_ngpu_cards
isinteger misses_ngigabytes_memory
isinteger train_gpu_ncpu_cores
isinteger train_gpu_ngpu_cards
isinteger train_gpu_ngigabytes_memory
isinteger train_cpu_ncpu_cores
isinteger train_cpu_ngpu_cards
isinteger train_cpu_ngigabytes_memory
isinteger generalize_gpu_ncpu_cores
isinteger generalize_gpu_ngpu_cards
isinteger generalize_gpu_ngigabytes_memory
isinteger generalize_cpu_ncpu_cores
isinteger generalize_cpu_ngpu_cards
isinteger generalize_cpu_ngigabytes_memory
isinteger xvalidate_gpu_ncpu_cores
isinteger xvalidate_gpu_ngpu_cards
isinteger xvalidate_gpu_ngigabytes_memory
isinteger xvalidate_cpu_ncpu_cores
isinteger xvalidate_cpu_ngpu_cards
isinteger xvalidate_cpu_ngigabytes_memory
isinteger mistakes_ncpu_cores
isinteger mistakes_ngpu_cards
isinteger mistakes_ngigabytes_memory
isinteger activations_gpu_ncpu_cores
isinteger activations_gpu_ngpu_cards
isinteger activations_gpu_ngigabytes_memory
isinteger activations_cpu_ncpu_cores
isinteger activations_cpu_ngpu_cards
isinteger activations_cpu_ngigabytes_memory
isinteger cluster_ncpu_cores
isinteger cluster_ngpu_cards
isinteger cluster_ngigabytes_memory
isinteger accuracy_ncpu_cores
isinteger accuracy_ngpu_cards
isinteger accuracy_ngigabytes_memory
isinteger freeze_ncpu_cores
isinteger freeze_ngpu_cards
isinteger freeze_ngigabytes_memory
isinteger classify1_gpu_ncpu_cores
isinteger classify1_gpu_ngpu_cards
isinteger classify1_gpu_ngigabytes_memory
isinteger classify1_cpu_ncpu_cores
isinteger classify1_cpu_ngpu_cards
isinteger classify1_cpu_ngigabytes_memory
isinteger classify2_ncpu_cores
isinteger classify2_ngpu_cards
isinteger classify2_ngigabytes_memory
isinteger ethogram_ncpu_cores
isinteger ethogram_ngpu_cards
isinteger ethogram_ngigabytes_memory
isinteger compare_ncpu_cores
isinteger compare_ngpu_cards
isinteger compare_ngigabytes_memory
isinteger congruence_ncpu_cores
isinteger congruence_ngpu_cards
isinteger congruence_ngigabytes_memory
isinteger accuracy_parallelize
isinteger cluster_parallelize
isinteger congruence_parallelize

isbinary() {
  binary_re='^[0-1]$'
  [[ "${!1}" =~ $binary_re ]] || echo WARNING: $1 is not set or is not 0 or 1
}

isbinary gui_snippets_waveform
isbinary gui_snippets_spectrogram
isbinary gui_context_waveform
isbinary gui_context_spectrogram
isbinary activations_gpu
isbinary classify_gpu
isbinary generalize_gpu
isbinary train_gpu
isbinary xvalidate_gpu

[[ "$gui_context_spectrogram_units" == mHz || "$gui_context_spectrogram_units" == Hz || "$gui_context_spectrogram_units" == kHz || "$gui_context_spectrogram_units" == MHz ]] || \
      echo WARNING: gui_context_spectrogram_units should be "mHz", "Hz", "kHz", or "MHz"

resource_kinds=(ncpu_cores ngpu_cards ngigabytes_memory)
for resource_kind in "${resource_kinds[@]}"; do
  readarray job_resources < <(set | grep ${resource_kind}= | grep -v local_ | grep -v server_)
  for job_resource in "${job_resources[@]}"; do
    job_resource_name=${job_resource%=*}
    job_resource_value=${job_resource##*=}
    local_resource=local_$resource_kind
    (( $job_resource_value > ${!local_resource} )) && \
          echo WARNING: $job_resource_name exceeds ${!local_resource} $local_resource
    if [[ -n "$server_ipaddr" ]] ; then
      server_resource=server_$resource_kind
      (( $job_resource_value > ${!server_resource} )) && \
            echo WARNING: $job_resource_name exceeds ${!server_resource} $server_resource
    fi
  done
  unset job_resource
  unset job_resources
done
unset resource_kind
unset resource_kinds

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo SongExplorer version: $(cat $DIR/../VERSION.txt)

allow_websocket=--allow-websocket-origin=localhost:$port
ipaddr=noIPv4s

flags=(-i -I)
for flag in ${flags[*]} ; do
  read -a theseips <<< $(hostname $flag)
  for thisip in ${theseips[*]} ; do
      [[ $thisip == *':'* ]] && continue
      allow_websocket=${allow_websocket}' '--allow-websocket-origin=$thisip:$port
      ipaddr=$thisip
  done
done

thisip=$(hostname)
if [ "$thisip" != '(none)' ] ; then
    allow_websocket=${allow_websocket}' '--allow-websocket-origin=$thisip:$port
    ipaddr=$thisip
fi

echo $ipaddr:$port

trap "local_njobs=\`hetero njobs\`; \
      if [[ \\\$\? && (( \"\$local_njobs\" > 0 )) ]] ; then \
          echo WARNING: jobs are still queued locally; \
          echo to kill them execute \\\`\\\$SONGEXPLORER_BIN hetero stop force\\\`; \
          echo to stop SongExplorer\'s scheduler, wait until they are done and execute \\\`\\\$SONGEXPLORER_BIN hetero stop\\\`; \
      else \
          hetero stop; \
      fi; \
      if [[ -n \"$server_ipaddr\" ]] ; then \
        server_njobs=\`ssh $server_ipaddr \"export SINGULARITYENV_PREPEND_PATH=$source_path; $SONGEXPLORER_BIN hetero njobs\"\`; \
        if [[ \\\$\? && (( \"\$server_njobs\" > 0 )) ]] ; then \
            echo WARNING: jobs are still queued on the server; \
            echo to kill them execute \\\`ssh $server_ipaddr \\\$SONGEXPLORER_BIN hetero stop force\\\`; \
            echo to stop SongExplorer\'s scheduler, wait until they are done and execute \\\`ssh $server_ipaddr \\\$SONGEXPLORER_BIN hetero stop\\\`; \
        else \
            ssh $server_ipaddr \"export SINGULARITYENV_PREPEND_PATH=$source_path; $SONGEXPLORER_BIN hetero stop\"; \
        fi; \
      fi" INT TERM KILL STOP HUP

hetero_nslots=`hetero nslots`
hetero_isrunning=$?
if [[ "$hetero_isrunning" != 0 ]] ; then
    hetero start $local_ncpu_cores $local_ngpu_cards $local_ngigabytes_memory
elif [[ "$hetero_nslots" != "$local_ncpu_cores $local_ngpu_cards $local_ngigabytes_memory" ]] ; then

    echo WARNING: SongExplorer\'s scheduler is already running with local_ncpu_cores,
    echo local_ngpu_cards, and local_ngigabytes_memory set to $hetero_nslots,
    echo respectively, which is different than specified in the configuration
    echo file.  To use the latter instead, quit SongExplorer, execute
    echo \`\$SONGEXPLORER_BIN hetero stop\`, and restart SongExplorer.
fi

if [[ -n "$server_ipaddr" ]] ; then
    hetero_nslots=`ssh $server_ipaddr "export SINGULARITYENV_PREPEND_PATH=$source_path; $SONGEXPLORER_BIN hetero nslots"`
    hetero_isrunning=$?
    if [[ "$hetero_isrunning" != 0 ]] ; then
        ssh $server_ipaddr "export SINGULARITYENV_PREPEND_PATH=$source_path; $SONGEXPLORER_BIN hetero start \
                            $server_ncpu_cores $server_ngpu_cards $server_ngigabytes_memory" &
    elif [[ "$hetero_nslots" != "$server_ncpu_cores $server_ngpu_cards $server_ngigabytes_memory" ]] ; then

        echo WARNING: SongExplorer\'s scheduler is already running on
        echo $server_ipaddr with server_ncpu_cores, server_ngpu_cards, and
        echo server_ngigabytes_memory set to $hetero_nslots, respectively, which
        echo is different than specified in the configuration file.  To use
        echo the latter instead, quit SongExplorer, execute \`ssh $server_ipaddr
        echo \$SONGEXPLORER_BIN hetero stop\`, and restart SongExplorer.

    fi
fi

bokeh serve \
      $allow_websocket \
      --show $DIR/gui \
      --port $port \
      --args "$(cat $DIR/../VERSION.txt)" $configuration_file
