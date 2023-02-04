#sudo sh scripts/screens-monitoring.sh "/home/alireza/PycharmProjects/IO_monitoring/venv/bin/python"
#    "src/tensorflow_apps" "cifar-100-classification-with-keras.py" "nvme0n1" "keras-classification-cifar-100"


echo none > /sys/block/nvme0n1/queue/scheduler
sync
echo 3 > /proc/sys/vm/drop_caches

mkdir -p $2/$5

screen -dmS monitoring-screen-iostat bash -c "time iostat -tx /dev/$4 1 > $2/$5/iostat.txt"
screen -dmS monitoring-screen-blktrace bash -c "time blktrace -d /dev/$4 -a complete -o - > $2/$5/trace.txt"

$1 $2/$3

screen -XS monitoring-screen-iostat quit
screen -XS monitoring-screen-blktrace quit

time cat $2/$5/trace.txt | blkparse -i - > $2/$5/parsed_trace.txt

$1 src/blktrace_monitoring/blktrace_plot.py $2/$5
$1 src/iostat_monitoring/iostat/main.py --data $2/$5/iostat.txt --disk $4 --output $2/$5/iostat.csv csv
$1 src/iostat_monitoring/iostat/main.py --data $2/$5/iostat.txt --disk $4 --fig-output $2/$5/iostat-plot.png plot
$1 src/iostat_monitoring/intensive.py  $2/$5/iostat_cpu.csv  $2/$5/iostat_devices.csv $2/$5

rm -f $2/$5/trace.txt

chmod -R 777 $2/$5/iostat-plot.png
chmod -R 777 $2/$5/iostat_cpu.csv
chmod -R 777 $2/$5/iostat_devices.csv
chmod -R 777 $2/$5
chmod -R 777 $2/$5/iostat.txt
chmod -R 777 $2/$5/parsed_trace.txt


