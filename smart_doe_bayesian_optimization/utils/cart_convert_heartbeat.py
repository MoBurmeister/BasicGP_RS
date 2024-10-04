import re

def extract_info(message):
    # Remove any spaces from the message
    clean_message = message.replace(" ", "")
    
    # Find the section between 'AA' and 'BB' (time in milliseconds), exactly 8 hex characters
    time_hex_match = re.search(r"AA([0-9A-Fa-f]{8})BB", clean_message)
    if time_hex_match:
        time_hex = time_hex_match.group(1)
        time_in_ms = int(time_hex, 16)  # Convert from hex to decimal
    else:
        time_in_ms = None
        print("Time not found between AA and BB")

    # Find the section between 'CC' and 'DD' (max distance), exactly 8 hex characters
    dist_hex_match = re.search(r"CC([0-9A-Fa-f]{8})DD", clean_message)
    if dist_hex_match:
        dist_hex = dist_hex_match.group(1)
        max_dist = int(dist_hex, 16)  # Convert from hex to decimal
    else:
        max_dist = None
        print("Max distance not found between CC and DD")

    return time_in_ms, max_dist

# Example message
message = "F5 01 00 4D 21 EE 00 5A 00 03 93 19 19 07 00 FF FF 82 00 00 00 00 00 00 00 00 00 00 F9 1D 00 00 00 02 01 00 00 00 00 00 00 FF FF 03 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 AA 00 00 07 99 BB 00 2B 64 37 CC 00 00 06 AB DD 00 00 06 A7 75 4B "


# true distance in mm - fixed during the optimization run
true_distance = 1703

# Extract information
time_in_ms, max_dist = extract_info(message)

# difference from true distance to max distance

diff = max_dist - true_distance

absolute_in_mm = abs(diff)

print(f"\nTime (ms): {time_in_ms}")
print(f"Max Distance (mm): {max_dist}")
print(f"Absolute difference from true distance (mm): {absolute_in_mm}\n")
