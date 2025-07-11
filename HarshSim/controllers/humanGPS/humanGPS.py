from controller import Robot, GPS, Emitter

# Constants
TIME_STEP = 32

# Initialize the robot
human_robot = Robot()

# Initialize the GPS sensor
human_gps = human_robot.getDevice('h_gps')
human_gps.enable(TIME_STEP)

# Initialize the Emitter to send the GPS position
emitter = human_robot.getDevice('emitter')
emitter.setChannel(1)  # Ensure the channel matches the receiver's channel

while human_robot.step(TIME_STEP) != -1:
    # Get the GPS position of the human
    human_position = human_gps.getValues()
    # print(f"Human GPS Position: {human_position}")

    # Send the human position as a comma-separated string
    message = f"{human_position[0]},{human_position[1]},{human_position[2]}"
    emitter.send(message.encode('utf-8'))
