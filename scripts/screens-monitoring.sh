screen -dmS monitoring-screen-iostat bash -c "time iostat -mx sda 60 > ./iostat.txt"
screen -dmS monitoring-screen-blktrace bash -c "time blktrace -d /dev/sda -a complete -o - > ./blktrace.txt"

$1 $2 $3

screen -XS monitoring-screen-iostat quit
screen -XS monitoring-screen-blktrace quit
