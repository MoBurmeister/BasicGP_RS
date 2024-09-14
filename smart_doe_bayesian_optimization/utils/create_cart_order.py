# true command: A55000150033005A010000000000000005DC00003205001111

# A5 50 00 15 00 33 00 5A 01 00 00 00 00 00 00 00 05 DC 00 00 32 05 00 11 11
# A5 50 00 15 00 05 00 5A 01 00 00 00 00 00 00 00 05 DC 00 00 32 05 00 11 11



import random

def generate_robot_command(distance, max_speed, acceleration, deceleration):
    # Base command structure (task ID will be replaced and calculated later)
    base_command = "A5 50 00 15 00 XX 00 5A 01 00 00 00 00 00 "
    
    # Generate a random task ID between 00 and 99 in hex
    task_id = f"{random.randint(0, 99):02}"

    # Print initial inputs
    print(f"Initial Inputs:")
    print(f"  Distance (mm): {distance}")
    print(f"  Max Speed (m/s): {max_speed}")
    print(f"  Acceleration (m/s²): {acceleration}")
    print(f"  Deceleration (m/s²): {deceleration}")
    print(f"Generated Task ID: {task_id}\n")
    
    # Convert units as required
    # Distance stays in mm, no conversion required
    # Convert max speed from m/s to m/min
    max_speed_m_per_min = max_speed * 60
    # Convert acceleration and deceleration from m/s² to m/min²
    acceleration_m_per_min2 = acceleration * 3600
    deceleration_m_per_min2 = deceleration * 3600
    
    # Print converted values
    print(f"Converted Values:")
    print(f"  Max Speed (m/min): {max_speed_m_per_min}")
    print(f"  Acceleration (m/min²): {acceleration_m_per_min2}")
    print(f"  Deceleration (m/min²): {deceleration_m_per_min2}\n")
    
    # Convert input values to hexadecimal
    distance_hex = f"{distance:08X}".upper()  # 4 pairs of digits for distance
    max_speed_hex = f"{int(max_speed_m_per_min):04X}".upper()  # 2 pairs of digits for max speed
    acceleration_hex = f"{int(acceleration_m_per_min2):04X}".upper()  # 2 pairs of digits for acceleration
    deceleration_hex = f"{int(deceleration_m_per_min2):04X}".upper()  # 2 pairs of digits for deceleration
    
    # Print hex values
    print(f"Hexadecimal Values:")
    print(f"  Distance (hex): {distance_hex}")
    print(f"  Max Speed (hex): {max_speed_hex}")
    print(f"  Acceleration (hex): {acceleration_hex}")
    print(f"  Deceleration (hex): {deceleration_hex}\n")
    
    # Construct the final command by replacing placeholders and adding values
    command = base_command.replace("XX", task_id)
    
    # Adding distance, max speed, acceleration, and deceleration into the correct positions
    final_command = (
        f"{command} {distance_hex[:4]} {distance_hex[4:]} "
        f"00 {max_speed_hex} {acceleration_hex} {deceleration_hex}"
    )
    
    # Print final command
    print(f"Final Command: \n{final_command}\n")
    
    return final_command

# Input values
distance = 1500        # distance (in mm)
max_speed = 1.0        # max speed (in m/s) - Max: 90 m/min which is 1.5 m/s
acceleration = 3    # acceleration (in m/s^2) - Max is 18 000 m/min² which is 5 m/s² [0.0, 18000.0]
deceleration = 3    # deceleration (in m/s^2) - Max is 18 000 m/min² which is 5 m/s² [0.0, 18000.0]

# Generate command
command = generate_robot_command(distance, max_speed, acceleration, deceleration)
