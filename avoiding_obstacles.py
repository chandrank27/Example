from picarx import Picarx
import time
from picarx import Picarx
import readchar
import sys
import cv2

POWER = 50
SafeDistance = 40   # > 30 safe
DangerDistance = 20 # > 20 && < 30 turn around, 
                    # < 20 backward
                    


def main():

    try:
        px = Picarx()
        # px = Picarx(ultrasonic_pins=['D2','D3']) # tring, echo
        key= readchar.readkey().lower()      
        while True:
            distance = round(px.ultrasonic.read(), 2)
            print("distance: ",distance)
            if key == "q":
                px.stop()
                print("quite")
            if distance >= SafeDistance:
                px.set_dir_servo_angle(0)
                px.forward(POWER)
            elif distance >= DangerDistance:
                px.set_dir_servo_angle(40)
                px.forward(POWER)
                time.sleep(0.1)
            else:
                px.set_dir_servo_angle(-40)
                px.backward(POWER)
                time.sleep(0.5)
            

    finally:
        px.forward(0)


if __name__ == "__main__":
    main()

