
global_counter = 1
divisor = 1

times_divised = 0

step_distant = 10

num_iterations = 100

for iteration in range(num_iterations):
    
    if global_counter % divisor == 0:
        times_divised = times_divised + 1
        print(f"Retraining code at iteration: {iteration + 1} and {times_divised} times divised")

        if times_divised % step_distant == 0:
            divisor = divisor * 2
            times_divised = 0
    
    global_counter = global_counter + 1