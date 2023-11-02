from picarx import Picarx


px = Picarx()
px.set_dir_servo_angle(-5)
# px = Picarx(ultrasonic_pins=['D2','D3']) # tring, echo
px.forward(30)


while True:
    distance = px.ultrasonic.read()
    print("distance: ",distance)
    if distance > 0 and distance < 300:
        if distance < 50:
            px.forward(0)
            break

print('quit ...') 
