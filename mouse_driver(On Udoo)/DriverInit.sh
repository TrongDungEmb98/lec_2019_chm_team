#!/bin/bash
sudo insmod driver.ko
sudo mknod /dev/mousek c 247 0
sudo chmod 666 /dev/mousek
