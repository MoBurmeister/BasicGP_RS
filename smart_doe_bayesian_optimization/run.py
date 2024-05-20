from flask import Flask, render_template, redirect, url_for, request, jsonify
from web_main import setup_first_model, setup_optimizer, perform_optimization_iteration

#template folder needs to be defined
app = Flask(__name__, template_folder='flask_app/templates', static_folder='flask_app/static')

#global variable to store the first model
first_gp = None
#global variable to store the optimizer
optimizer = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/activate_model', methods=['POST'])
def activate_model():
    try:
        global first_gp
        first_gp = setup_first_model()
        num_parameters = first_gp.train_X.shape[1]
        return jsonify({'status': 'success', 'message': 'Model activated successfully!', 'num_parameters': num_parameters})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
@app.route('/activate_optimizer', methods=['POST'])
def activate_optimizer():
    try:
        global optimizer, first_gp
        optimizer = setup_optimizer(first_gp)
        next_parameter_setting = optimizer.next_proposed_parameter_setting.tolist()
        flattened_list = [item for sublist in next_parameter_setting for item in sublist]
        print(next_parameter_setting)
        print(type(next_parameter_setting))
        return jsonify({'status': 'success', 'message': 'Optimizer initiated successfully!', 'next_parameter_setting': flattened_list})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
@app.route('/perform_optimization', methods=['POST'])
def perform_optimization():
    try:
        global optimizer
        user_input = request.json.get('userInput')
        # Perform your function with the input value here
        user_input = float(user_input)
        perform_optimization_iteration(optimizer, user_input)
        print(f"User input received: {user_input}")
        next_parameter_setting = optimizer.next_proposed_parameter_setting.tolist()
        flattened_list = [item for sublist in next_parameter_setting for item in sublist]
        return jsonify({'status': 'success', 'message': f'Input value {user_input} processed by the AI Model!', 'next_parameter_setting': flattened_list})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == "__main__":
    app.run(debug=True)