#!/bin/bash

# launch the graphical user interface
 
# deepsong.sh <configuration-file> <port>
# http://<hostname>:<port>/deepsong

# e.g.
# $DEEPSONG_BIN gui.sh `pwd`/configuration.sh 5006

configuration_file=$1
port=$2

source $configuration_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

thishost=$(hostname)
if [ "$thishost" == '(none)' ] ; then
    thishost=$(hostname -I)
    thishost=${thishost% }
fi
echo $thishost:$port
thisip=$(hostname -i)

trap "local_njobs=\`hetero njobs\`; \
      if [[ \\\$\? && (( \"\$local_njobs\" > 0 )) ]] ; then \
          echo WARNING: jobs are still queued locally; \
          echo to kill them execute \\\`\\\$DEEPSONG_BIN hetero stop force\\\`; \
          echo to stop DeepSong\'s scheduler, wait until they are done and execute \\\`\\\$DEEPSONG_BIN hetero stop\\\`; \
      else \
          hetero stop; \
      fi; \
      if [[ -n \"$server_ipaddr\" ]] ; then \
        server_njobs=\`ssh $server_ipaddr \"$server_export $DEEPSONG_BIN hetero njobs\"\`; \
        if [[ \\\$\? && (( \"\$server_njobs\" > 0 )) ]] ; then \
            echo WARNING: jobs are still queued on the server; \
            echo to kill them execute \\\`ssh $server_ipaddr \\\$DEEPSONG_BIN hetero stop force\\\`; \
            echo to stop DeepSong\'s scheduler, wait until they are done and execute \\\`ssh $server_ipaddr \\\$DEEPSONG_BIN hetero stop\\\`; \
        else \
            ssh $server_ipaddr \"$server_export $DEEPSONG_BIN hetero stop\"; \
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
    hetero_nslots=`ssh $server_ipaddr "$server_export $DEEPSONG_BIN hetero nslots"`
    hetero_isrunning=$?
    if [[ "$hetero_isrunning" != 0 ]] ; then
        ssh $server_ipaddr "$server_export $DEEPSONG_BIN hetero start \
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
      --allow-websocket-origin=${thishost}:$port \
      --allow-websocket-origin=${thisip}:$port \
      --allow-websocket-origin=localhost:$port \
      --show $DIR/gui \
      --port $port \
      --args $configuration_file $audio_tic_rate $audio_nchannels $gui_snippet_ms $gui_nx_snippets $gui_ny_snippets $gui_nlabels $gui_gui_width_pix $gui_context_width_ms $gui_context_offset_ms $gui_cluster_background_color $gui_cluster_dot_colormap $gui_cluster_hex_colormap $gui_cluster_hex_range_low $gui_cluster_hex_range_high $gui_snippet_colormap