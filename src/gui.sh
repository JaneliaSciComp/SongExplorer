#!/bin/bash

# launch the graphical user interface
 
# deepsong.sh <configuration-file> <port>
# http://<hostname>:<port>/deepsong

# e.g.
# $DEEPSONG_BIN gui.sh `pwd`/configuration.sh 5006

configuration_file=$1
port=$2

source $configuration_file

isinteger() {
  number_re='^[0-9]+$'
  [[ "${!1}" =~ $number_re ]] || echo WARNING: $1 is not set or is not an integer
}

isinteger audio_tic_rate
isinteger audio_nchannels
isinteger gui_snippet_ms
isinteger gui_snippet_nx
isinteger gui_snippet_ny
isinteger gui_nlabels
isinteger gui_gui_width_pix
isinteger gui_context_width_ms
isinteger gui_context_offset_ms
isinteger models_per_job
isinteger pca_batch_size
isinteger accuracy_nprobabilities
isinteger nstrides
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

isbinary() {
  binary_re='^[0-1]$'
  [[ "${!1}" =~ $binary_re ]] || echo WARNING: $1 is not set or is not 0 or 1
}

isbinary activations_gpu
isbinary classify_gpu
isbinary generalize_gpu
isbinary train_gpu
isbinary xvalidate_gpu
isbinary accuracy_parallelize
isbinary cluster_parallelize
isbinary congruence_parallelize

IFS=$'\n' read -d '' -a where_vars < <(set | grep _where=  )
for var in "${where_vars[@]}"; do
  lhs=${var%%=*}
  (( "$lhs" == local )) || (( "$lhs" == server )) || (( "$lhs" == cluster )) || \
        echo WARNING: $var should be "local", "server", or "cluster"
done
unset where_vars

resource_kinds=(ncpu_cores ngpu_cards ngigabytes_memory)
for resource_kind in "${resource_kinds[@]}"; do
  IFS=$'\n' read -d '' -a job_resources < <(set | grep ${resource_kind}= | grep -v local_ | grep -v server_)
  for job_resource in "${job_resources[@]}"; do
    job_resource_name=${job_resource%=*}
    job_resource_value=${job_resource##*=}
    local_resource=local_$resource_kind
    server_resource=server_$resource_kind
    (( $job_resource_value > ${!local_resource} )) && \
          echo WARNING: $job_resource_name exceeds ${!local_resource} $local_resource
    (( $job_resource_value > ${!server_resource} )) && \
          echo WARNING: $job_resource_name exceeds ${!server_resource} $server_resource
  done
  unset job_resource
  unset job_resources
done
unset resource_kind
unset resource_kinds

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

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
          echo to kill them execute \\\`\\\$DEEPSONG_BIN hetero stop force\\\`; \
          echo to stop DeepSong\'s scheduler, wait until they are done and execute \\\`\\\$DEEPSONG_BIN hetero stop\\\`; \
      else \
          hetero stop; \
      fi; \
      if [[ -n \"$server_ipaddr\" ]] ; then \
        server_njobs=\`ssh $server_ipaddr \"export SINGULARITYENV_PREPEND_PATH=$source_path; $DEEPSONG_BIN hetero njobs\"\`; \
        if [[ \\\$\? && (( \"\$server_njobs\" > 0 )) ]] ; then \
            echo WARNING: jobs are still queued on the server; \
            echo to kill them execute \\\`ssh $server_ipaddr \\\$DEEPSONG_BIN hetero stop force\\\`; \
            echo to stop DeepSong\'s scheduler, wait until they are done and execute \\\`ssh $server_ipaddr \\\$DEEPSONG_BIN hetero stop\\\`; \
        else \
            ssh $server_ipaddr \"export SINGULARITYENV_PREPEND_PATH=$source_path; $DEEPSONG_BIN hetero stop\"; \
        fi; \
      fi" INT TERM KILL STOP HUP

hetero_nslots=`hetero nslots`
hetero_isrunning=$?
if [[ "$hetero_isrunning" != 0 ]] ; then
    hetero start $local_ncpu_cores $local_ngpu_cards $local_ngigabytes_memory
elif [[ "$hetero_nslots" != "$local_ncpu_cores $local_ngpu_cards $local_ngigabytes_memory" ]] ; then

    echo WARNING: DeepSong\'s scheduler is already running with local_ncpu_cores,
    echo local_ngpu_cards, and local_ngigabytes_memory set to $hetero_nslots,
    echo respectively, which is different than specified in the configuration
    echo file.  To use the latter instead, quit DeepSong, execute
    echo \`\$DEEPSONG_BIN hetero stop\`, and restart DeepSong.
fi

if [[ -n "$server_ipaddr" ]] ; then
    hetero_nslots=`ssh $server_ipaddr "export SINGULARITYENV_PREPEND_PATH=$source_path; $DEEPSONG_BIN hetero nslots"`
    hetero_isrunning=$?
    if [[ "$hetero_isrunning" != 0 ]] ; then
        ssh $server_ipaddr "export SINGULARITYENV_PREPEND_PATH=$source_path; $DEEPSONG_BIN hetero start \
                            $server_ncpu_cores $server_ngpu_cards $server_ngigabytes_memory" &
    elif [[ "$hetero_nslots" != "$server_ncpu_cores $server_ngpu_cards $server_ngigabytes_memory" ]] ; then

        echo WARNING: DeepSong\'s scheduler is already running on
        echo $server_ipaddr with server_ncpu_cores, server_ngpu_cards, and
        echo server_ngigabytes_memory set to $hetero_nslots, respectively, which
        echo is different than specified in the configuration file.  To use
        echo the latter instead, quit DeepSong, execute \`ssh $server_ipaddr
        echo \$DEEPSONG_BIN hetero stop\`, and restart DeepSong.

    fi
fi

bokeh serve \
      $allow_websocket \
      --show $DIR/gui \
      --port $port \
      --args $configuration_file
