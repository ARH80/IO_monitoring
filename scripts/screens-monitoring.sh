echo none > /sys/block/nvme0n1/queue/scheduler
sync
echo 3 > /proc/sys/vm/drop_caches

screen -dmS monitoring-screen-iostat bash -c "time iostat -mx /dev/$2 60 > ./iostat.txt"
screen -dmS monitoring-screen-blktrace bash -c "time blktrace -d /dev/$2 -a complete -o - > ./trace.txt"

$1

screen -XS monitoring-screen-iostat quit
screen -XS monitoring-screen-blktrace quit

time cat ./trace.txt | blkparse -i - > ./parsed_trace.txt

rm -f ./trace.txt

python3 src/iostat_monitoring/iostat/main.py --data ./iostat.txt --disk $2 --fig-output iostat-plot.png plot
