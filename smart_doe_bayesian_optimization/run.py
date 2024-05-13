from flask import Flask, render_template, redirect, url_for, request
from web_main import setup_first_model, setup_optimization_loop, get_next_optimization_iteration

#template folder needs to be defined
app = Flask(__name__, template_folder='flask_app/templates', static_folder='flask_app/static')

#global variable to store the first model
first_gp = None
#global variable to store the optimizer
optimizer = None
#global variable to store the next value
next_value = "-"
#global variable to display next_value
next_display_value = "-"
#global variable to store the next y answer
next_y_value = None

@app.route('/')
def home():
    return render_template('index.html', next_display_value=next_display_value)

@app.route('/start_doe', methods=['POST'])
def start_doe():
    #call to use the global variable first_gp
    global first_gp 
    first_gp = setup_first_model() 
    print(f"global gp initialized with {first_gp}") 
    return redirect(url_for('home'))  

@app.route('/start_optimization', methods=['POST'])
def start_optimization():
    global optimizer, next_value, next_display_value
    optimizer, next_value = setup_optimization_loop(first_gp)
    next_display_value = round(next_value.item(), 2)
    print(f"optimization loop initiated with {first_gp}")
    return redirect(url_for('home'))

@app.route('/update_next_y_value', methods=['POST'])
def update_next_y_value():
    global next_y_value
    next_y_value = request.form['next_y_value']
    print(f"Updated next_y_value: {next_y_value}") 
    return redirect(url_for('home'))

@app.route('/get_next_optimization_iteration', methods=['POST'])
def get_next_optimization_iteration_route():
    global next_value, optimizer, next_y_value, first_gp, next_display_value
    print(next_y_value)
    print(type(next_y_value))
    next_value = get_next_optimization_iteration(optimizer, input_value=float(next_y_value), original_x=next_value)
    next_display_value = round(next_value.item(), 2)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)