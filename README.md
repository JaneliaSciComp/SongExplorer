Table of Contents
=================

   * [Description](#description)
   * [Public Domain Annotations](#public-domain-annotations)
   * [Citations and Repositories](#citations-and-repositories)
   * [Installation](#installation)
      * [Notation](#notation)
      * [Singularity for Linux, Mac, and Windows](#singularity-for-linux-mac-and-windows)
      * [Docker for Windows, Mac, and Linux](#docker-for-windows-mac-and-linux)
      * [System Configuration](#system-configuration)
      * [Batching Remotely](#batching-remotely)
         * [Another Workstation](#another-workstation)
         * [An On-Premise Cluster](#an-on-premise-cluster)
      * [Reporting Problems](#reporting-problems)
   * [Tutorial](#tutorial)
      * [Detecting Sounds](#detecting-sounds)
      * [Visualizing Clusters](#visualizing-clusters)
      * [Manually Annotating](#manually-annotating)
      * [Training a Classifier](#training-a-classifier)
      * [Quantifying Accuracy](#quantifying-accuracy)
      * [Making Predictions](#making-predictions)
      * [Correcting False Alarms](#correcting-false-alarms)
      * [Correcting Misses](#correcting-misses)
      * [Minimizing Annotation Effort](#minimizing-annotation-effort)
      * [Measuring Generalization](#measuring-generalization)
      * [Searching Hyperparameters](#searching-hyperparameters)
      * [Discovering Novel Sounds](#discovering-novel-sounds)
      * [Examining Errors](#examining-errors)
      # [Dense Congruence](#dense-congruence)
   * [Development](#development)
      * [Singularity](#singularity)
      * [Docker](#docker)

# Description #

You have an audio recording, and you want to know where certain classes of
sounds are.  DeepSong is trained to recognize such words by manually giving
it a few examples.  It will then automatically calculate the probability,
over time, of when those words occur in all of your recordings.

Applications suitable for DeepSong include quantifying the rate or pattern
of words emitted by a particular species, distinguishing a recording of one
species from another, and discerning whether individuals of the same species
produce different song.

Underneath the hood is a deep convolutional neural network.  The audio stream
is first converted into a spectrogram using the mel-frequency cepstrum.
The output is a set of mutually-exclusive probability waveforms corresponding
to each word of interest.

Training begins by first thresholding one of your recordings in the
time- and frequency-domains to find sounds that exceed the ambient noise.
These sounds are then clustered into similar categories for you to manually
annotate with however many word labels naturally occur.  A classifier is
then trained on this corpus of ground truth, and a new recording is analyzed
by it.  The words it automatically finds are then clustered as before, but
this time are displayed with predicted labels.  You manually correct the
mistakes, both re-labeling words that it got wrong, as well as labeling
words it missed.  These new annotations are added to the ground truth,
and the process of retraining the classifier and analyzing and correcting
new recordings is repeated until the desired accuracy is reached.


# Public Domain Annotations #

DeepSong is open source and free for you to use.  However, DeepSong is not
a static piece of software.  It’s performance is improved with additional
high-quality annotations.

Therefore, when you publish results based on DeepSong, we request that you make
all of your primary data and annotations freely available in a recognized data
repository, such as [figshare](http://figshare.com),
[Dryad](http://datadryad.org), or [Zenodo](http://zenodo.org).  Many journals
already require deposition of raw data, but we strongly encourage you to also
provide your manual annotations.  These manual annotations will serve to
improve the performance of DeepSong over time, helping both your own work and
that of everyone using DeepSong.

Please let us know where you have deposited your raw
data and annotations by posting an issue to the [DeepSong
repository](https://github.com/JaneliaSciComp/DeepSong).  We will endeavor to
maintain a database of these recordings and annotations and will periodically
re-train DeepSong with the new data.

In addition, consider donating your recordings to library or museum, like the
Cornell Lab of Ornithology's Macauley Library (www.macaulaylibrary.org) or the
Museo de Ciencias Naturales de Madrid's Fonoteca Zoológica (www.fonozoo.com).


# Citations and Repositories

BJ Arthur, Y Ding, M Sosale, F Khalif, S Turaga, DL Stern (in prep)  
DeepSong: A machine-learning classifier to segment and discover animal acoustic communication signals   
[BioRxiv]()  
[datadryad]()


# Installation #

DeepSong can be run on all three major platforms.  The installation procedure
is different on each due to various support of the technologies used.  We
recommend using Singularity on Linux and Apple Macintosh, and Docker on
Microsoft Windows.  Training your own classifier is fastest with an Nvidia
graphics processing unit (GPU).

TensorFlow, the machine learning framework from Google that DeepSong uses,
supports Ubuntu, Mac and Windows.  The catch is that TensorFlow doesn't
(currently) support GPUs on Macs.  So while using a pre-trained classifier
would be fine on a Mac, because inference is just as fast on the CPU,
training your own would be ~10x slower.

Docker, a popular container framework which provides an easy way to deploy
software across platforms, supports Linux, Mac and Windows, but only supports
GPUs on Linux.  Moreover, on Mac and Windows it runs within a heavy-weight
virtual machine, and on all platforms it requires administrator privileges to
both install and run.

Singularity is an alternative to Docker that does not require root access.  For
this reason it is required in certain high-performance computing (HPC)
environments.  Currently it only supports Mac and Linux out of the box; you can
run it on Windows within a virtual environment like Docker, but would have to
set that up yourself.  As with Docker, GPUs are only accessible on Linux.

## Notation ##

Throughout this document the dollar sign ($) in code snippets signifies your
computer terminal's command line.  Square brackets ([]) in code indicate
optional components, and angle brackets (<>) represent sections which you much
customize.

## Singularity for Linux, Mac, and Windows ##

Platform-specific installation instructions can be found at
[Sylabs](https://www.sylabs.io).  DeepSong has been tested with version 3.2.

You'll also need to install the CUDA and CUDNN drivers from nvidia.com.
The latter requires you to register for an account.  DeepSong was tested with
version 10.1.

Next download the DeepSong image from the
cloud.  You can either go to [DeepSong's cloud.sylabs.io
page](https://cloud.sylabs.io/library/_container/5ccca72a800ca26aa6ccf008)
and click the Download button, or equivalently use the command line (for
which you might need an access token):

    $ singularity pull library://bjarthur/default/deepsong:latest
    INFO:    Container is signed
    Data integrity checked, authentic and signed by:
      ben arthur (deepsong) <arthurb@hhmi.org>, Fingerprint EE89861064C45947BC8954B51ADBD926D9B48B0D

    $ ls -lht | head -n 2
    total 16G
    -rwxr-xr-x  1 arthurb scicompsoft 1.5G Sep  2 08:16 deepsong_latest.sif*

Finally, put these definitions in your .bashrc file:

    export DEEPSONG_BIN='singularity exec -B `pwd` --nv <path-to-deepsong_latest.sif>'
    alias deepsong="$DEEPSONG_BIN gui.sh `pwd`/configuration.sh 5006"

Note that the current directory is mounted in the `export` above with the `-B`
flag.  If you want to access any other directories, you'll have to add additional
flags (e.g. `-B /groups:/groups`).

If you don't have an Nvidia GPU or want to perform certain tasks using
just a CPU (which might be more cost effective; e.g. classification), pull
`library://bjarthur/default/deepsong:latest_cpu`, omit the `--nv` flag,
and export DEEPSONG_CPU_BIN as well.

## Docker for Windows, Mac, and Linux ##

Platform-specific installation instructions can be found at
[Docker](http://www.docker.com).  Once you have it installed, download the [DeepSong image
from
cloud.docker.com](https://cloud.docker.com/u/bjarthur/repository/docker/bjarthur/deepsong):

    $ docker pull bjarthur/deepsong
    Using default tag: latest
    latest: Pulling from bjarthur/deepsong
    Digest: sha256:466674507a10ae118219d83f8d0a3217ed31e4763209da96dddb03994cc26420
    Status: Image is up to date for bjarthur/deepsong:latest

    $ docker image ls
    REPOSITORY                   TAG                 IMAGE ID            CREATED             SIZE
    bjarthur/deepsong            latest              b63784a710bb        20 hours ago        2.27GB

Finally, put these definitions in your .bashrc file:

    export DEEPSONG_BIN='docker run -w `pwd` -v `pwd`:`pwd` --env DEEPSONG_BIN -h=`hostname` -p 5006:5006 bjarthur/deepsong'
    alias deepsong="$DEEPSONG_BIN gui.sh `pwd`/configuration.sh 5006"

Note that the current directory is mounted in the `export` above with the
`-v` flag.  If you want to access any other directories, you'll have to
add additional flags (e.g. `-v /Volumes:/Volumes`).

Should docker ever hang, or run for an interminably long time, and you
want to kill it, you'll need to open another terminal window and issue the
`stop` command:

    $ docker ps
    CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                NAMES
    6a26ad9d005e        bjarthur/deepsong   "detect.sh /src/p..."   3 seconds ago       Up 2 seconds        6006/tcp, 8888/tcp   heuristic_galois

    $ docker stop 6a26ad9d005e

If you have to do this often, consider putting this short cut in your
.bashrc file:

    $ alias dockerkill='docker stop $(docker ps --latest --format "{{.ID}}")'

The virtual machine that docker runs within on Mac and Windows is configured
by default with only 2 GB of memory.  You will probably want to increase
this limit.

## System Configuration ##

DeepSong is capable of training a classifier and making predictions on recordings
either locally on the host computer, remotely on a workstation, or remotely on
a cluster.  You specify how you want this to work by editing "configuration.sh".

Copy an exemplar configuration file out of the container and into your home directory:

    $ eval $DEEPSONG_BIN cp /opt/deepsong/configuration.sh .

Inside you'll find many shell variables and functions which control where DeepSong
does its work:

    $ grep _where= configuration.sh 
    default_where=local
    deepsong_where=local
    detect_where=$default_where
    misses_where=$default_where
    train_where=$default_where
    generalize_where=server #$default_where
    xvalidate_where=server #$default_where
    hidden_where=$default_where
    cluster_where=$default_where
    accuracy_where=$default_where
    freeze_where=$default_where
    classify_where=$default_where
    ethogram_where=$default_where
    compare_where=$default_where

    $  grep -A13 GENERIC configuration.sh 
    # GENERIC HOOK
    generic_it () {  # 1=cmd, 2=logfile, 3=jobname, 4=where, 5=deepsongbin, 6=bsubflags
        if [ "$4" == "local" ] ; then
            bash -c "$1" &> $2  #&
        elif [ "$4" == "server" ] ; then
            ssh c03u14 "$5 bash -c \"$1\" &> $2" #&
        elif [ "$4" == "cluster" ] ; then
            ssh login1 bsub \
                    -P stern \
                    -J $3 \
                    "$6" \
                    -oo $2 <<<"$5 bash -c \"$1\""
        fi
    }

    $  grep -A6 train_gpu  configuration.sh 
    train_gpu=1
    train_where=$default_where
    train_it () {
        if [ "$train_gpu" -eq "1" ] ; then
            generic_it "$1" "$2" "$3" "$train_where" "$DEEPSONG_BIN" "-n 2 -W 1440 -gpu \"num=1\" -q gpu_rtx" #&
        else
            generic_it "$1" "$2" "$3" "$train_where" "$DEEPSONG_CPU_BIN" "-n 24 -W 1440" #&
        fi
    }

Each operation (e.g. detect, train, classify, generalize, etc.) is dispatched
by an eponymous function ending in "\_it".  In the example above, `train_it` is
called when you train a model.  This function hook references a variable called
`train_where` that switches between using the "local" host computer, a remote
"server", or an on-premise "cluster".

In this example, each `_where` variable is set to "local" via the
`default\_where` variable at the top of the configuration file.  You can configure
which computer is used either globally through this variable, or by changing
the operation specific ones later in the file.

Whereas cluster jobs always run in the background, those on local and remote
workstations do not by default.  To make the latter asynchronous too, uncomment
the trailing ampersands (&).  Doing so in `generic_it()` will run all jobs in
the background; doing so just within, for example, `train_it()`, will run just
those specific operations asynchronously.  Whereas jobs which run in the
foreground signal their completion by returning the DoIt! button to its resting
color, for those in the background you'll have to check the directory structure
for the output files.

## Batching Remotely ##

The aforementioned eponymous hook functions can be changed to support a cluster
with a different scheduler (SGE, PBS, Slurm, etc.), to ssh into another
computer, or even to batch jobs out to the cloud.

### Another Workstation ###

Using a lab or departmental server, or perhaps a colleague's workstation
remotely, is easiest if you run DeepSong on it directly and then view the GUI
in your own personal workstation's internet browser.  To do this, simply `ssh`
into the server and install DeepSong as described above.

Alternatively, you can run the GUI code (in addition to viewing its output) on
your own personal workstation and batch compute jobs to the remote server.
This is easiest if there is a shared file system between the two computers.
The advantage here is that less compute intensive jobs (e.g. freeze, accuracy)
can be run on your workstation.  In this case:

* Store all DeepSong related files on the share, including the container image,
"configuration.sh", and all of your data.

* Make the remote and local file paths match by creating a symbolic link.
For example, if on a Mac you use SMB to mount as "/Volumes/sternlab"
an NSF drive whose path is "/groups/stern/sternlab", then add `-[v|B]
/groups/stern/sternlab` to `DEEPSONG_BIN` and `mkdir -p /groups/stern &&
ln -s /Volumes/sternlab/ /groups/stern/sternlab`.  With Docker you'll
additionally need to open the preferences panel and configure file sharing
to bind "/groups"

* Set the `DEEPSONG` environment variable plus the `deepsong` alias on both
your workstation and server to point to this same image.

* You might need an RSA key pair.  If so, you'll need to add `-[v|B]
~/.ssh:/ssh` to `DEEPSONG_BIN`.

* You might need to use ssh flags `-i /ssh/id_rsa -o "StrictHostKeyChecking
no"` in "configuration.sh".

If you do not have a shared file system, the DeepSong image and configuration file
must be separately installed on both computers, and you'll need to do all of
the compute jobs remotely.

### An On-Premise Cluster ###

Submitting jobs to a cluster is similar to using a remote workstation, so read
the above section first.  You might want to even try batching to a another
workstation first, as it can be easier to debug problems than doing so on a
cluster.

You use own workstation to view the GUI in a browser, and can either run the
GUI code locally or on the cluster.  With the former you have the option to
submit only a portion of the compute jobs to the cluster, whereas with the
latter they must all be performed by the cluster.  Running the GUI code on the
cluster also requires that the cluster be configured to permit hosting a web
page.  Moreover, if your cluster charges a use fee, you'll be charged even when
the GUI is sitting idle.

As before, it is easiest if there is a shared file system, and if so, all files
need to be on it, and the local and remote file paths must be the same or made
to be the same with links.  The environment variables and aliases must also be
the same.

You'll likely need an RSA key pair, possibly need special `ssh` flags, and
almost assuredly need to change the `bsub` command and/or its flags.  The best
person to ask for help here is your system administrator.

## Reporting Problems ##

The code is hosted on [github](https://github.com/JaneliaSciComp/DeepSong).
Please file an issue there for all bug reports and feature requests.
Pull requests are also welcomed!  For major changes it is best to file an
issue first so we can discuss implementation details.  Please work with me
to improve DeepSong instead instead of forking your own version.


# Tutorial #

Let's walk through the steps needed to train a classifier completely from
scratch.

Recordings need to be monaural 16-bit little-endian PCM-encoded WAVE files.
They should all be sampled at the same rate, which can be anything.  For this
tutorial we supply you with *Drosophila melanogaster* data sampled at 2500 Hz.

First, let's get some data bundled with DeepSong into your home directory.

    $ eval $DEEPSONG_BIN ls /opt/deepsong/data
    20161207T102314_ch1_p1.wav     PS_20130625111709_ch3_p2.wav
    20161207T102314_ch1_p2.wav     PS_20130625111709_ch3_p3.wav
    20161207T102314_ch1_p3.wav     my_frozen_graph_1k_0.pb     
    PS_20130625111709_ch3_p1.wav   vgg_labels.txt              

    $ mkdir -p groundtruth-data/round1

    $ eval $DEEPSONG_BIN cp /opt/deepsong/data/PS_20130625111709_ch3_p1.wav ./groundtruth-data/round1

## Detecting Sounds ##

Now that we have some data, let's extract the timestamps of some sounds from
one of these as-of-yet unannotated audio recordings.

First, start DeepSong's GUI:

    $ deepsong

Then in your favorite internet browser navigate to
`http://localhost:5006/deepsong`.  If you are running the DeepSong GUI on
a remote computer, replace `localhost` with that computer's hostname or IP
address:

    $ hostname [-i]
    arthurb-ws2

On the left you'll see three empty panels (one purple, two white) in which the
sound recordings are displayed and annotated.  In the middle are buttons and
text boxes used to train the classifier and make predictions with it.  On the
right is this instruction manual for easy reference.

Click on the Detect button and it will turn blue and all of the parameters
below that it does *not* use will be disabled.  If all of the required
parameters are filled in, the DoIt! button in the upper right will in addition
be enabled and turn red.

The first time you use DeepSong all of the parameters will need to be
specified.  In the File Dialog browser immediately below, navigate to the
"configuration.sh" file that you just copied from the container and then click on the
Parameters button.  Notice that the large text box to the right of the File
Dialog browser now contains the text of this file.  Similarly, navigate to the
WAV file in the "round1/" directory and click on the WAV,TF,CSV Files button.
Lastly you'll need to specify the six numeric parameters that control the
algorithm used to find sounds:  In the time domain, subtract the median, take
the absolute value, threshold by the median absolute deviation times `σ time`,
and morphologically close gaps shorter than `time smooth` milliseconds.
Separately, in the frequency domain, create a spectrogram using a window of
length `freq N` milliseconds (`freq N` / 1000 * sampling_rate should be a power
of two) and twice `freq NW` Slepian tapers, adjust the default threshold of the
F-test by a factor of `ρ freq`, and open islands and close gaps shorter than
`freq smooth`.  Sound events are considered to be periods of time which pass
either of these two criteria.

Once all the needed parameters are specified, click on the red DoIt! button to
start the detecting sounds.  It will turn orange until the the computations are
finished, unless you have configured it to run asynchronously ([see
System Configuration](#system-configuration)).

The result is a file of comma-separated values with the start and stop times
(in tics) of sounds which exceeded a threshold in either the time or frequency
domain.

    $ head -3 groundtruth-data/round1/PS_20130625111709_ch3_p1-detected.csv
    PS_20130625111709_ch3_p1,4501,4503,detected,time
    PS_20130625111709_ch3_p1,4628,4631,detected,time
    PS_20130625111709_ch3_p1,4807,4810,detected,time

    $ tail -3 groundtruth-data/round1/PS_20130625111709_ch3_p1-detected.csv
    PS_20130625111709_ch3_p1,945824,946016,detected,frequency
    PS_20130625111709_ch3_p1,947744,947936,detected,frequency
    PS_20130625111709_ch3_p1,962720,962912,detected,frequency

## Visualizing Clusters ##

To cluster these detected sounds we're going to use the same method that we'll
later use to cluster the hidden state activations of a trained classifier.

Click on the Train button to create a randomly initialized network.  Before
clicking a second time though, we need to change `# steps`, `validation
period`, and `validation %` to all be 0.  You'll also need to use the File
Dialog browser to choose directories in which to put the log files and to find
the ground truth data.  Lastly you'll need to specify "time,frequency" as the
wanted words and "detected" as the label types to match what is in the CSV file
you just created above.  Now press the Train button a second time!  Output into
the log file directory are "train.log", "train_1k.log", and "train_1k/".  The
former two files contain error transcripts should any problems arise, and the
latter folder contains checkpoint files prefixed with "vgg.ckpt-" which save
the weights of the neural network at regular intervals.

Use the Hidden button to save the mel-frequency cepstrograms and hidden state
activations by mock-classifying these detected sounds with this untrained
network.  You'll need to tell it which model to use by selecting the last
checkpoint file in the untrained classifier's log files with the File Dialog
browser.  Output are three files in the ground truth directory beginning with
"hidden":  the two ending in ".log" report any errors, and "hidden.npz"
contains the actual data in binary format.

Now cluster the hidden state activations with the Cluster button.  The time and
amount of memory this takes depends directly on the number and dimensionality
of detected sounds.  To limit the problem to a manageable size one can use
`max_samples` to randomly choose a subset of samples to cluster.  (The `σ time`
and `ρ freq` variables can also be used limit how many sound events were
detected in the first place.)  So that words with few samples are not obscured
by those with many, you might also want to randomly subsample the former using
`equalize ratio`.  Principal Component Analysis (PCA) is used to reduce the
dimensionality of the vector of hidden unit activations describing each word,
and the fraction of variance retained is specified by `PCA fraction`.  Lastly,
there are also two parameters which directly control the clustering algorithm:
`perplexity` and `exaggeration`; see [van der Maaten and
Hinton](http://www.jmlr.org/papers/v9/vandermaaten08a.html) (2008; J. machine
Learning Res.) for a description.  Output are four files in the ground truth
directory beginning with "cluster":  a ".log" file with errors and a ".npz"
file with data as before, plus two PDF files showing the results of the
principal components analysis (PCA) that precedes the t-SNE clustering.

Finally, click on the Visualize button to render the clusters as hexagonally
binned densities in the left-most panel.  It is normal to see some flickering
while this happens.  Once it is done you should see a cloud of blue, green,
and yellow patches, nominally with some structure.

To browse through your recordings first click on the Detected button (as
distinct from the Detect button above).  The large plot on the left will then
show the detected sounds as their mel-cepstrograms cluster in t-SNE space.
Click on one of the more yellowish hexagons and a fuchsia circle will appear.
Adjust its size with the `radius` variable.  In the right panel are now
displayed snippets of waveforms which are nominally similar to one another.
They will each be labeled "d. time" or "d. frequency" to indicate which
threshold criterion they passed and that they were detected (as opposed to
annotated, predicted, or missed; see below).  The color is the scale bar--
yellow is loud and purple is quiet.  Clicking on a snippet will show it in
greater temporal context in the wide panel below.  Pan and zoom with the
buttons labeled "<", ">", "+", "-", and "0".

## Manually Annotating ##

To record a manual annotation, first pick a waveform snippet that contains an
unambiguous example of a particular word.  Type the word's name into one of the
text boxes at the bottom and hit return to activate the corresponding counter
to the left.  Hopefully the gray box in the upper half of the wide context
window nicely demarcates the temporal extent of the word.  If so, all you have
to do is to double click the grey box and it will be extended to the bottom
half and your chosen label will be applied.  If not, either double-click or
click-and-drag in the bottom half of the wide context window to create a custom
time span for a new annotation.  In all cases, annotations can be deleted by
double clicking any of the gray boxes.

For this tutorial, choose the words "mel-pulse", "mel-sine", "ambient", and
"other".  We use the syntax "\<A\>-\<B\>" here, where A is the species (mel
is short for *D.  melanogaster*) and B is the song type, but that is not
strictly required.  The word syntax could nominally be anything.  The GUI
does have a feature, however, to split labels at the hyphen and display
groups of words that share a common prefix or suffix.

## Training a Classifier ##

Once you have a few tens of examples for each word, it's time to train a
classifier.  First, confirm that the annotations you just made were saved into
an "-annotated.csv" file in the ground truth folder.

    $ tree groundtruth-data
    groundtruth-data
    ├── cluster.npz
    ├── cluster.log
    ├── cluster-pca-hidden.log
    ├── cluster-pca-hidden-zoomed.log
    ├── hidden.npz
    ├── hidden.log
    ├── hidden-samples.log
    └── round1
        ├── PS_20130625111709_ch3_p1-annotated.csv
        ├── PS_20130625111709_ch3_p1-detected.csv
        ├── PS_20130625111709_ch3_p1-threshold.log
        └── PS_20130625111709_ch3_p1.wav

    $ tail -5 groundtruth-data/round1/PS_20130625111709_ch3_p1-annotated.csv
    PS_20130625111709_ch3_p1.wav,771616,775264,annotated,mel-sine
    PS_20130625111709_ch3_p1.wav,864544,870112,annotated,mel-sine
    PS_20130625111709_ch3_p1.wav,898016,910276,annotated,ambient
    PS_20130625111709_ch3_p1.wav,943493,943523,annotated,mel-pulse
    PS_20130625111709_ch3_p1.wav,943665,943692,annotated,mel-pulse

Now train a classifier on your annotations using the Train button.  Fifty steps
suffices for this amount of ground truth.  So we can accurately monitor the
progress, withhold 40% of the annotations to validate on, and do so every 10 steps.
You'll also need to change wanted words to "mel-pulse,mel-sine,ambient,other"
and label types to "annotated" so that it will ignore the detected annotations
in the ground truth directory.  It's important to include "other" as a wanted
word here, even if you haven't labeled any sounds as such, as it will be used
later by DeepSong to highlight false negatives ([see
Correcting Misses](#correcting-misses)).  Note that the total number of annotations must
exceed the the size of the mini-batches, which is specified by "mini-batch".

With small data sets the network should just take a minute or so to train.
As your example set grows, you might want to monitor the training progress
as it goes:

    $ watch tail trained-classifier/train_1k.log
    Every 2.0s: tail trained-classifier1/train_1k.log                            Mon Apr 22 14:37:31 2019

    INFO:tensorflow:Elapsed 39.697532, Step #9: rate 0.000200, accuracy 75.8%, cross entropy 0.947476
    INFO:tensorflow:Elapsed 43.414184, Step #10: rate 0.000200, accuracy 84.4%, cross entropy 0.871244
    INFO:tensorflow:Saving to "/groups/scicompsoft/home/arthurb/projects/turaga/stern/deepsong/trained-classifier1/train_1k/vgg.ckpt-10"
    INFO:tensorflow:Confusion Matrix:
     ['mel-pulse', 'mel-sine', 'ambient']
     [[26  9  9]
     [ 0  4  0]
     [ 0  0  4]]
    INFO:tensorflow:Elapsed 45.067488, Step 10: Validation accuracy = 65.4% (N=52)
    INFO:tensorflow:Elapsed 48.786851, Step #11: rate 0.000200, accuracy 79.7%, cross entropy 0.811077

It is common for the accuracy, as measured on the withheld data and reported as
"Validation accuracy" in the log file above, to be worse than the training
accuracy.  If so, it is an indication that the classifier does not generalize
well at that point.  With more training steps and more ground truth data though
the validation accuracy should become well above chance.

## Quantifying Accuracy ##

Measure the classifier's performance using the Accuracy button.  Output are the
following charts and tables in the Logs Folder and the `train_*` subdirectories
therein:

* "training.pdf" shows the training and validation accuracies as a function of
the number of training steps and wall-clock time.

* "confusion-matrix.pdf" shows in aggregate which word each annotation was
classified as.  The upper right triangle in each square is normalized to the
row and is called the recall, while the lower left is to the column-normalized
precision.

* "precision-recall.pdf" and `sensitivity-specificity.pdf` show how the ratio
of false positives to false negatives changes as the threshold used to call an
event changes.  The areas underneath these curves are widely-cited metrics of
performance.

* "thresholds.csv" lists the word-specific probability thresholds that one can
use to achieve a specified precision-recall ratio.  This file is used when
creating ethograms ([see Making Predictions](#making-predictions)).

* "probability-density.pdf" shows, separately for each word, histograms of the
values of the classifier's output taps across all of that word's annotations.
The difference between a given word's probability distribution and the second
most probable word can be used as a measure of the classifier's confidence.

* The CSV files in the "predictions" directory list the specific annotations
which were mis-classified (plus those that were correct).  The WAV files and
time stamps therein can be used to look for patterns in the raw data ([see
Examining Errors](#examining-errors)).


## Making Predictions##

For the next round of manual annotations, we're going to have this newly
trained classifier find sounds for us instead of using a simple threshold.  And
we're going to do so with a different recording so that the classifier learns
to be insensitive to experimental conditions.

First let's get some more data bundled with DeepSong into your home directory:

    $ mkdir groundtruth-data/round2

    $ eval $DEEPSONG_BIN cp /opt/deepsong/data/20161207T102314_ch1_p1.wav /
            groundtruth-data/round2

Then use the Freeze button to save the classifier's neural network graph
structure and weight parameters into the single file that TensorFlow needs for
inference.  You'll need to choose a checkpoint to use with the File Dialog
browser as before.  Output into the log files directory are two ".log" files
for errors, and a file ending with ".pb" containing the binary data.

Now use the Classify button to generate probabilities over time for each
annotated word.  Specify which recordings using the File Dialog browser and
the WAV,TF,CSV Files button.  These are first stored in a log file ending in
".tf", and then converted to WAV files for easy viewing.

    $ ls groundtruth-data/round2/
    20161207T102314_ch1_p1-ambient.wav   20161207T102314_ch1_p1-mel-pulse.wav
    20161207T102314_ch1_p1-classify.log  20161207T102314_ch1_p1.tf
    20161207T102314_ch1_p1-mel-sine.wav  20161207T102314_ch1_p1.wav

Discretize these probabilities using thresholds based on a set of
precision-recall ratios using the Ethogram button.  The ratios used are those
in the "thresholds.csv" file in the log files folder, which is created by the
Accuracy button and controlled by "P/Rs".  You'll need to specify which ".tf"
files to threshold using the File Dialog browser and the WAV,TF,CSV button.

    $ ls -t1 groundtruth-data/round2/ | head -4
    20161207T102314_ch1_p1-ethogram.log
    20161207T102314_ch1_p1-predicted-0.5pr.csv
    20161207T102314_ch1_p1-predicted-1.0pr.csv
    20161207T102314_ch1_p1-predicted-2.0pr.csv

    $ head -5 groundtruth-data/round2/20161207T102314_ch1_p1-predicted-1.0pr.csv 
    20161207T102314_ch1_p1.wav,19976,20008,predicted,mel-pulse
    20161207T102314_ch1_p1.wav,20072,20152,predicted,mel-sine
    20161207T102314_ch1_p1.wav,20176,20232,predicted,mel-pulse
    20161207T102314_ch1_p1.wav,20256,20336,predicted,mel-sine
    20161207T102314_ch1_p1.wav,20360,20416,predicted,mel-pulse

The resulting CSV files are in the same format as those generated when we
detected sounds in the time and frequency domains as well as when we manually
annotated words earlier using the GUI.  Note that the fourth column
distinguishes whether these words were detected, annotated, or predicted.

## Correcting False Alarms ##

In the preceding section we generated three sets of detected sounds by
applying three sets of word-specific thresholds to the probability waveforms:

    $ cat trained-classifier/thresholds.csv 
    precision/recall,2.0,1.0,0.5
    mel-pulse,0.9977890984593017,0.508651224000211,-1.0884193525904096
    mel-sine,0.999982304641803,0.9986744484433365,0.9965472849431617
    ambient,0.999900757998532,0.9997531463467944,0.9996660975683063

Higher thresholds result in fewer false positives and more false negatives.
A precision-recall ratio of one means these two types of errors occur at
equal rates.  Your experimental design drives this choice.

Let's manually check whether our classifier in hand accurately calls sounds
using these thresholds.  First, choose one of the predicted CSV files that has
a good mix of the labels and either delete or move outside the ground truth
directory the other ones.  Then cluster and visualize the neural network's
hidden state activations as we did before using the Hidden, Cluster, and
Visualize buttons.  The only difference here is that "annotated,predicted"
should be used as the wanted words instead of just "annotated" or "detected".
The "PS_20130625111709_ch3_p1-detected.csv" file can still be in the
ground truth folder as it is ignored.

Now let's correct the mistakes!  Click on the Predicted and Ambient buttons, set
the radius to say 10, and then click on a yellow hexagon.  Were the classifier
perfect, all the snippets now displayed would look like background noise.
Click on the ones that don't and manually annotate them appropriately.
Similarly click on "mel-" and "-pulse" and correct any mistakes, and then
"mel-" and "-sine".

Keep in mind that the only words which show up in the clustered t-SNE space are
those that exceed the chosen threshold.  Any mistakes you find in the snippets
are hence strictly false positives.

## Correcting Misses ##

It's important that false negatives are corrected as well.  One way find them
is to click on random snippets and look at the surrounding context in the
window below.  A better way is to look at detected sounds that don't exceed the
probability threshold.  To do this, first detect sounds in the recording you
just classified, using the Detect button as before, and create a list of the
subset of these sounds which were not assigned a label using the Misses button.
For the latter, you'll need to specify both the detected and predicted CSV
files with the File Dialog browser and the WAV,TF,CSV button.  The result is
another CSV file, this time ending in "missed.csv":

    $ head -5 groundtruth-data/round2/20161207T102314_ch1_p1-missed.csv 
    20161207T102314_ch1_p1.wav,12849,13367,missed,other
    20161207T102314_ch1_p1.wav,13425,13727,missed,other
    20161207T102314_ch1_p1.wav,16105,18743,missed,other
    20161207T102314_ch1_p1.wav,18817,18848,missed,other
    20161207T102314_ch1_p1.wav,19360,19936,missed,other

Now cluster and visualize the hidden state activations as before, using
the Hidden, Cluster, and Visualize buttons in turn, making sure to specify
"annotated,missed" this time as the label types.

Click on "missed" and a yellow hexagon.  Were the classifier perfect, none of
the snippets would be an unambiguous example of any of the labels you trained
upon earlier.  Annotate any of them that are, and add new label types for sound
events which fall outside the current categories.

## Minimizing Annotation Effort

From here, we just keep alternating between annotating false positives and
false negatives, using a new recording for each iteration, until mistakes
become sufficiently rare.  The most effective annotations are those that
correct the classifier's mistakes, so don't spend much, if any, time annotating
what it got right.

Each time you train a new classifier, all of the existing "predicted.csv",
"missed.csv", ".tf", and word-probability WAV files are moved to an "oldfiles"
sub-folder as they will be out of date.  You might want to occasionally delete
these folders to conserve disk space:

    $ rm groundtruth-data/*/oldfiles*

Ideally a new model would be trained after each new annotation is made, so that
subsequent time is not spent correcting a prediction (or lack thereof) that
would no longer be made in error.  Training a classifier takes time though, so
a balance must be struck with how quickly you alternate between annotating and
training.

Since there are more annotations each time you train, use a proportionately
smaller percentage of them for validations and proportionately larger number of
training steps.  You don't need more than ten-ish annotations for each word to
confirm that the learning curves converged.  And since the learning curves
generally don't converge until the entire data set has been sampled many times
over, set "# steps" to be several fold greater than the number of annotations
(shown in the table near the labels) divided by "mini-batch", and check that it
actually converges with "training.pdf" (generated by the Accuracy button).

As the wall-clock time spent training is generally shorter with larger
mini-batches, set it as high as the memory in your GPU will permit.  Multiples
of 32 are generally faster.  The caveat here is that exceedingly large
mini-batches can reduce accuracy, so make sure to compare it with smaller ones.

One should make an effort to choose a recording at each step that is most
different from the ones trained upon so far.  Doing so will produce a
classifier that generalizes better.

Once an acceptable number of errors in the ethograms is achieved, train a model
for use in your experiments with nearly all of your annotations (`validation % ~= 1`)
as this will produce the most accurate predictions.  Report its accuracy using
cross validation (see [Searching Hyperparameters](#searching-hyperparameters)),
and/or by leaving entire recordings out for validation ([see Measuring
Generalization](#measuring-generalization)).  The former estimate is a minimum,
and asymptotically converges to the true accuracy as the number of folds
approaches the number of annotations, while the latter estimates your model's
ability to generalize to new recordings.

## Measuring Generalization ##

Up to this point we have validated on a small portion of each recording.  Once you
have annotated many recordings though, it is better to set aside entire WAV
files to validate on.  In this way we measure the classifier's ability to
generalize.

To train one classifier with a single recording or set of recordings withheld for
validation, either (1) select those file(s) in the File Dialog browser and press the
Withhold Files button; (2) store a list of WAV file(s) you'd like to withhold in
a text file (either comma separated or one per line), and select this file; (3)
or, organize the subddirectories in the ground truth folder such that all of the
file(s) you'd like to withhold are in just one of them, and then select this
sub-directory with the Withhold Files Button.  In all cases, use the Train
button as before.

To train multiple classifiers, each of which withholds a single recording in
your ground truth data, set Withheld Files to the ground truth folder.  The
Train button will then iteratively launch a job for each WAV file that has been
annotated, withholding it for validation and training on the rest of the entire
ground truth data set.  In this case, separate files and subdirectories are
created in the Logs Folder that are suffixed by the name of the withheld WAV
file.

Of course, training multiple classifiers is quickest when done simultaneously
instead of sequentially.  In practice this is only possible if your machine has
multiple GPUs, or you have access to a cluster.

A simple jitter plot of the accuracies on withheld recordings is included in
the output of the Accuracy button.  Their average is the most appropriate to
report, as it will likely be worse than a model trained on a portion of each
recording.

## Searching Hyperparameters ##

Achieving high accuracy is not just about annotating lots of data, it also
depends on choosing the right model.  While DeepSong is (currently) set up
solely for convolutional neural networks, there are many free parameters by
which to tune its architecture.  You configure them by editing the variables
itemized below, and then use cross-validation to compare different choices.
One could of course also modify the source code to permit radically different
neural architectures, or even something other than neural networks.

* `context` is the temporal duration, in milliseconds, that the classifier
inputs

* `shift by` is the asymmetry, in milliseconds, of `context` with respect to the
point in time that is annotated or being classified.  `shift by` divided by
`stride` (see below) should be an integer.  Positive values move `context` to
left relative to the annotation.

* `window` is the length of the temporal slices, in milliseconds, that
constitute the spectrogram.  `window` / 1000 * `sample rate` should round down
to a power of two.

* `mel,DCT` specifies how many taps to use in the mel-frequency cepstrum.  The
first number is for the mel-frequency resampling and the second for the
discrete cosine transform.  Modifying these is tricky as valid values depend on
`sample rate` and `window`.  The table below show the maximum permissable
values for each, and are what is recommended.  See the code in
"tensorflow/contrib/lite/kernels/internal/mfcc.cc" for more details.

|sample rate|window|mel,DCT|
|:---------:|:----:|:-----:|
|10000      |12.8  |28,28  |
|10000      |6.4   |15,15  |
| 5000      |6.4   |11,11  |
| 2500      |6.4   |7,7    |
| 1250      |6.4   |3,3    |
|10000      |3.2   |7,7    |

* `stride` is the time, in milliseconds, by which the `window`s in the
spectrogram are shifted.  1000/`stride` must be an integer.

* `dropout` is the fraction of hidden units on each forward pass to omit
during training.

* `optimizer` can be one of stochastic gradient descent (SGD),
[Adam](https://arxiv.org/abs/1412.6980),
[AdaGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf), or
[RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

* `learning rate` specifies the fraction of the gradient to change each weight
by at each training step.

* `kernel sizes` is a 3-vector of the size of the convolutional kernels.  The
first value is used for each layer until the tensor height in the frequency
axis is smaller than it.  Then the second value is then repeately used until
the height is again smaller than it.  Finally the third value is used until the
width is less than `last conv width`.

* `nfeatures` is the number of feature maps at each of the corresponding stages
in `kernel_sizes`.

To search for the optimal value for a particular hyperparameter, first choose
how many folds you want to partition your ground truth data into using
`k-fold`.  Then set the hyperparameter of interest to the first value you want
to try and choose a name for the Logs Folder such that its prefix will be
shared across all of the hyperparameters values you plan to validate.  Click the
X-Validate button and then DoIt!  One classifier will be trained for each fold,
using it as the validation set and the remaining folds for training.  Separate
files and subdirectories are created in the Logs Folder that are suffixed by
the fold number.  Plot overlayed training curves with the Accuracy button, as
before.  Repeat the above procedure for each of remaining hyperparameter values
you want to try.  Then use the Compare button to create a figure of the
cross-validation data over the hyperparameter values, specifying the prefix
that the logs folders have in common.

## Discovering Novel Sounds ##

After amassing a sizeable amount of ground truth one might wonder whether one
has manually annotated all types of words that exist in the recordings.  One
way to check for any missed types is to look for hot spots in the clusters of
detected sounds that have no corresponding annotations.  Annotating known types
in these spots should improve generalization too.

First, set label types to "annotated" and train a model than includes "time"
and "frequency" plus all of your existing wanted words
("mel-pulse,mel-sine,ambient,other").  Then, use the Detect button to threshold
*all* of the recordings that you have annotated.  Save their hidden state
activations, along with those of the manually annotated sounds, using the
Hidden button by setting the label types to "annotated,detected".  Cluster and
visualize as before.  Now rapidly and alternately click the Annotated and
Detected buttons to find any differences in the density distributions.  Click
on any new hot spots you find in the detected clusters, and annotate sounds
which are labeled as detected but not annotated.  Create new word types as
necessary.

## Examining Errors ##

Mistakes can possibly be corrected if more annotations are made of similar
sounds.  To find such sounds, cluster the predictions made on the ground
truth annotations with sounds detected in your recordings.

As [mentioned earlier](#quantifying-accuracy), the Accuracy button generates a
"predictions/" folder in the log files directory containing CSV files itemizing
whether the sounds in the validation set were correctly or incorrectly classified.
Each CSV file corresponds to a sub-folder within the ground truth folder.  First
copy these CSV files into their corresponding sub-folders.  Then detect sounds
in all of your ground truth recordings if you haven't done so already, save the
hidden state activations, and cluster using "detected,correct,mistake" as the
label types.

    $ tail -n 10 trained-classifier/predictions/round2.csv 
    PS_20130625111709_ch3_p1.wav,377778,377778,correct,me-pulse,me-pulse
    PS_20130625111709_ch3_p1.wav,157257,157257,correct,me-pulse,me-pulse
    PS_20130625111709_ch3_p1.wav,164503,165339,correct,ambient,ambient
    PS_20130625111709_ch3_p1.wav,379518,379518,mistake,ambient,me-pulse
    PS_20130625111709_ch3_p1.wav,377827,377827,correct,me-pulse,me-pulse
    PS_20130625111709_ch3_p1.wav,378085,378085,correct,me-pulse,me-pulse
    PS_20130625111709_ch3_p1.wav,379412,379412,mistake,ambient,me-pulse
    PS_20130625111709_ch3_p1.wav,160474,161353,correct,ambient,ambient
    PS_20130625111709_ch3_p1.wav,207780,208572,correct,mel-sine,mel-sine
    PS_20130625111709_ch3_p1.wav,157630,157630,correct,me-pulse,me-pulse

The file format here is similar to DeepSong's other CSV files, with the
difference being that the penultimate column is the prediction and the
final one the annotation.

When visualizing the clusters, click on the Mistake button and look for a
localized density.  Click on that hot spot to examine the shapes of waveforms
that are mis-classified-- the ones whose text label, which is the prediction,
does not match the waveform.  Then click on the Detected button and manually
annotate similar waveforms.  Nominally they will cluster at the same location.

## Dense Congruence ##

The accuracy statistics reported in the confusion matrices described above are
limited to false negatives.  If a manual annotation withheld to validate upon
does not have the maximum probability across all labels, it is considered an
error.  A false positive is just the opposite-- a prediction that a word has
occurred without the corresponding annotation at that time.  To measure the
false positive rate then one must make predictions at all points in time, not
just at those that are annotated.  Moreover, checking that these dense
predictions are correct necessitates annotating all occurrences of a word.

To quantify the false positive rate, first put a set of recordings in a new
sub-directory within the Ground Truth Folder.  Then cluster the sounds in these
recordings by either detecting sounds or making predictions.  Annotate every
occurrence of each word of interest by scrolling to the beginning of each
recording and panning all the way to the end.  Suffix each "annotated.csv" file
with the name of the annotator (e.g. "annotated-YYYYMMDDTHHMMSS-ben.csv").  If
you don't already have a trained classifier, train a new one with this
directory withheld to test upon, using a small percentage of the rest of the
data for validation.  Make ethograms of these densely annotated recordings.
Finally, use the Dense button to plot Venn diagrams showing the fraction of
false positives and negatives.

The same procedure can be used to quantify the congruence between multiple
human annotators.  Simply create "annotated-<name>.csv" files for each one, and
another circle will be added to the Venn diagrams.


# Development #

## Singularity ##

To build an image, first download the latest CUDA and CUDNN drivers from
nvidia.com.  You'll need to register for the latter.  Then change to a local
(i.e. not NFS mounted; e.g. /opt/users) directory and:

    $ ls cud*
    cuda_10.1.243_418.87.00_linux.run  cudnn-10.1-linux-x64-v7.6.2.24.tgz

    $ git clone https://github.com/JaneliaSciComp/DeepSong.git
    $ sudo singularity build -s deepsong.img deepsong/containers/singularity-gpu.def
    $ sudo singularity exec -B `pwd`:/mnt -w deepsong.img sh /mnt/cuda_<version>-run --silent --toolkit
    $ sudo singularity exec -B `pwd`:/mnt -w deepsong.img tar xvzf /mnt/cudnn-<version>.tgz -C /usr/local

To confirm that the image works:

    $ singularity run --nv deepsong.img
    >>> import tensorflow as tf
    >>> hello = tf.constant('Hello, TensorFlow!')
    >>> sess = tf.Session()
    >>> sess.run(hello)

Optionally, compress the image into a single file:

    $ sudo singularity build deepsong.sif deepsong.img

To push an image to the cloud, first copy your access token from
cloud.sylabs.io to ~/.singularity/sylabs-token, then:

    $ singularity sign deepsong.sif
    $ singularity push deepsong.sif library://bjarthur/default/deepsong:<version>[_cpu]

To build an image without GPU support, use "singularity-cpu.def", skip the
`singularity exec` commands, omit the `--nv` flags, and add the "\_cpu" suffix.

To use the DeepSong source code outside of the container, set
SINGULARITYENV_PREPEND_PATH to the full path to DeepSong's `src` directory
in your shell environment.

## Docker ##

To start docker on linux and set permissions:

    $ service docker start
    $ setfacl -m user:$USER:rw /var/run/docker.sock

To build a docker image and push it to docker hub:

    $ cd deepsong
    $ docker build --file=containers/dockerfile-cpu --tag=bjarthur/deepsong [--no-cache=true] .
    $ docker login
    $ docker {push,pull} bjarthur/deepsong

To monitor resource usage:

    $ docker stats

To run a container interactively add "-i --tty".
