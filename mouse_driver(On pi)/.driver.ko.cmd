cmd_/home/pi/lec_2019_chm_team/mouse_driver/driver.ko := ld -r  -EL -T ./scripts/module-common.lds -T ./arch/arm/kernel/module.lds  --build-id  -o /home/pi/lec_2019_chm_team/mouse_driver/driver.ko /home/pi/lec_2019_chm_team/mouse_driver/driver.o /home/pi/lec_2019_chm_team/mouse_driver/driver.mod.o ;  true