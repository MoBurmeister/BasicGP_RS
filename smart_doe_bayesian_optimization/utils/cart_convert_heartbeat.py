import re

def extract_info(message):
    # Remove any spaces from the message
    clean_message = message.replace(" ", "")
    
    # Find the section between 'AA' and 'BB' (time in milliseconds)
    time_hex_match = re.search(r"AA(.*?)BB", clean_message)
    if time_hex_match:
        time_hex = time_hex_match.group(1)
        time_in_ms = int(time_hex, 16)  # Convert from hex to decimal
    else:
        time_in_ms = None
        print("Time not found between AA and BB")

    # Find the section between 'CC' and 'DD' (max distance)
    dist_hex_match = re.search(r"CC(.*?)DD", clean_message)
    if dist_hex_match:
        dist_hex = dist_hex_match.group(1)
        max_dist = int(dist_hex, 16)  # Convert from hex to decimal
    else:
        max_dist = None
        print("Max distance not found between CC and DD")

    return time_in_ms, max_dist

# TODO: fix Additional C error, where it gets too high hex value

# Example message
message = "F5 01 00 4D 43 FE 00 5A 00 02 64 19 19 07 00 FF FF 87 00 00 00 00 00 00 00 00 00 00 F9 1D 00 00 00 02 01 00 00 00 00 00 00 FF FF 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 AA 00 00 07 66 BB 00 00 90 94 CC 00 00 05 FF DD 00 00 05 DC 55 B0 "


# true distance in mm
true_distance = 1500

# Extract information
time_in_ms, max_dist = extract_info(message)

# difference from true distance to max distance

diff = max_dist - true_distance

absolute_in_mm = abs(diff)

print(f"\nTime (ms): {time_in_ms}")
print(f"Max Distance (mm): {max_dist}")
print(f"Absolute difference from true distance (mm): {absolute_in_mm}\n")
